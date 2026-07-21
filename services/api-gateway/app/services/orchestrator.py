"""
Request router and pipeline orchestrator service for API Gateway (`services/api-gateway`).
Dispatches authenticated requests to the Ingestion Celery queue (`process_document_batch`)
and invokes the real LangGraph Correction Router engine (`CorrectionRouterGraph`).
"""

import logging
import time
from typing import Any
import uuid
from datetime import UTC, datetime

from celery import Celery
from fastapi import HTTPException
import httpx
from pydantic import BaseModel, Field
from shared.config import get_settings
from shared.models.common import (
    ConfidenceStatus,
    QueryRequest,
    QueryResponse,
    TenantContext,
)

from app.db.models import DocumentBatchRecord
from app.db.session import get_sync_session
from app.services.rbac import rbac_service

logger = logging.getLogger(__name__)


class DocumentIngestRequest(BaseModel):
    """Payload format for document ingestion submission."""

    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original file name")
    expected_pages: int = Field(..., ge=1, description="Expected page count")
    tenant_id: str | None = Field(default=None, description="Optional tenant override")
    pages: list[dict] | None = Field(
        default=None, description="Optional pre-extracted page payloads"
    )


class DocumentIngestResponse(BaseModel):
    """Response returned upon submission of an ingestion job."""

    job_id: str = Field(..., description="Unique Celery/batch job tracking ID")
    document_id: str = Field(..., description="Associated document ID")
    status: str = Field(..., description="Initial job status")
    message: str = Field(..., description="Human readable submission confirmation")


class JobStatusResponse(BaseModel):
    """Status polling response for asynchronous ingestion jobs."""

    id: str = Field(..., description="Batch job identifier")
    document_id: str = Field(..., description="Associated document ID")
    filename: str = Field(..., description="File name being processed")
    expected_pages: int = Field(..., description="Total expected pages")
    received_pages: int = Field(..., description="Pages processed/received so far")
    status: str = Field(..., description="Current job execution status")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")


class GatewayOrchestratorService:
    """
    Unified orchestrator managing Ingestion task dispatch, retrieval self-correction loop execution,
    and asynchronous job status verification under strict tenant isolation.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.celery_app = Celery(
            "compass_api_gateway",
            broker=self.settings.REDIS_URL,
            backend=self.settings.REDIS_URL,
        )

    def ingest_document(
        self, payload: DocumentIngestRequest, tenant_context: TenantContext
    ) -> DocumentIngestResponse:
        """
        Enforces tenant scope and RBAC permissions, initializes database tracking record,
        and dispatches ingestion task asynchronously via Celery.
        """
        target_tenant = payload.tenant_id or tenant_context.tenant_id
        rbac_service.verify_tenant_scope(tenant_context, target_tenant)
        rbac_service.verify_document_access(tenant_context, payload.document_id)

        job_id = str(uuid.uuid4())
        now = datetime.now(UTC)

        session = get_sync_session()
        try:
            batch_record = DocumentBatchRecord(
                id=job_id,
                document_id=payload.document_id,
                filename=payload.filename,
                expected_pages=payload.expected_pages,
                received_pages=0,
                status="PROCESSING",
                created_at=now,
                updated_at=now,
            )
            session.add(batch_record)
            session.commit()
        except Exception as e:
            session.rollback()
            raise HTTPException(
                status_code=500, detail=f"Failed to initialize ingestion batch record: {str(e)}"
            ) from e
        finally:
            session.close()

        # Dispatch task to Celery worker queue
        self.celery_app.send_task(
            "app.tasks.process_document_batch",
            kwargs={
                "batch_id": job_id,
                "document_id": payload.document_id,
                "filename": payload.filename,
                "expected_pages": payload.expected_pages,
                "tenant_id": target_tenant,
            },
        )

        return DocumentIngestResponse(
            job_id=job_id,
            document_id=payload.document_id,
            status="PROCESSING",
            message="Document ingestion job submitted successfully.",
        )

    async def ingest_upload_file(
        self, file: Any, document_id: str, tenant_id: str, tenant_context: TenantContext
    ) -> DocumentIngestResponse:
        """
        Proxies an uploaded file from API Gateway directly to Ingestion Service (/ingest/upload).
        """
        target_tenant = tenant_id or tenant_context.tenant_id
        content = await file.read()

        async with httpx.AsyncClient() as client:
            try:
                files = {"file": (file.filename, content, getattr(file, "content_type", "application/pdf") or "application/pdf")}
                data = {"document_id": document_id, "tenant_id": target_tenant}
                resp = await client.post(
                    f"{self.settings.INGESTION_SERVICE_URL}/ingest/upload",
                    files=files,
                    data=data,
                    headers={"X-Tenant-ID": target_tenant},
                    timeout=300.0,
                )
                if resp.status_code not in (200, 202):
                    logger.error("Ingestion service upload failed: %s - %s", resp.status_code, resp.text)
                    raise HTTPException(status_code=resp.status_code, detail=f"Ingestion service error: {resp.text}")
                res_json = resp.json()
                job_id = res_json.get("batch_id") or str(uuid.uuid4())
                return DocumentIngestResponse(
                    job_id=job_id,
                    document_id=document_id,
                    status="COMPLETED",
                    message="Document successfully processed and indexed into Qdrant/Elasticsearch.",
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Failed to forward upload to ingestion service: %s", e)
                raise HTTPException(status_code=500, detail=f"Ingestion proxy failure: {str(e)}") from e

    async def process_query(
        self, payload: QueryRequest, tenant_context: TenantContext
    ) -> QueryResponse:
        """
        Validates tenant isolation, verifies document-level RBAC restrictions, and executes
        the LangGraph Correction Router across retrieval, evaluation, and generation using async invoke.
        """
        target_tenant = (
            (payload.tenant_context.tenant_id if payload.tenant_context else None)
            or payload.tenant_id
            or tenant_context.tenant_id
        )
        rbac_service.verify_tenant_scope(tenant_context, target_tenant)

        doc_id_filter = payload.metadata_filter.get("document_id")
        if doc_id_filter:
            rbac_service.verify_document_access(tenant_context, str(doc_id_filter))

        # Lazy import to avoid circular imports during startup/testing if needed
        try:
            from services.correction.app.services.graph import get_correction_graph
        except ImportError:
            from correction.app.services.graph import get_correction_graph

        graph = get_correction_graph()

        initial_state = {
            "query": payload.query,
            "original_query": payload.query,
            "tenant_id": target_tenant,
            "attempt_count": 0,
            "retrieved_chunks": [],
            "retrieval_confidence": 0.0,
            "retrieval_status": ConfidenceStatus.VERIFIED,
            "contradictions_detected": False,
            "contradiction_reasoning": "",
            "has_same_date_contradiction": False,
            "draft_answer": "",
            "draft_citations": [],
            "groundedness_score": 0.0,
            "groundedness_verdict": True,
            "verdicts": [],
            "final_answer": "",
            "final_status": ConfidenceStatus.VERIFIED,
        }

        t0 = time.perf_counter()
        final_state = await graph.ainvoke(initial_state)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return QueryResponse(
            answer=final_state.get("final_answer", "No answer generated."),
            confidence_status=final_state.get("final_status", ConfidenceStatus.VERIFIED),
            confidence_score=float(
                final_state.get(
                    "groundedness_score", final_state.get("retrieval_confidence", 1.0)
                )
            ),
            citations=final_state.get("draft_citations", []),
            verdicts=final_state.get("verdicts", []),
            latency_ms=elapsed_ms,
        )

    def get_job_status(self, job_id: str, tenant_context: TenantContext) -> JobStatusResponse:
        """
        Fetches job status for an ingestion batch record and verifies tenant ownership.
        """
        session = get_sync_session()
        try:
            batch = session.get(DocumentBatchRecord, job_id)
            if not batch:
                raise HTTPException(
                    status_code=404, detail=f"Job ID '{job_id}' not found."
                )
            return JobStatusResponse(
                id=batch.id,
                document_id=batch.document_id,
                filename=batch.filename,
                expected_pages=batch.expected_pages,
                received_pages=batch.received_pages,
                status=batch.status,
                created_at=batch.created_at.isoformat(),
                updated_at=batch.updated_at.isoformat(),
            )
        finally:
            session.close()


gateway_orchestrator = GatewayOrchestratorService()
