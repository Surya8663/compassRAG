"""
Request Router endpoints (`/v1/ingest`, `/v1/query`, `/v1/status/{job_id}`) for API Gateway.
Enforces strict JWT authentication and delegates to `GatewayOrchestratorService`.
"""

from typing import Annotated
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from shared.models.common import QueryRequest, QueryResponse, TenantContext

from app.services.auth import get_current_tenant_context
from app.services.orchestrator import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    JobStatusResponse,
    gateway_orchestrator,
)

router = APIRouter(prefix="/v1", tags=["gateway"])
TenantContextDep = Annotated[TenantContext, Depends(get_current_tenant_context)]


@router.post("/ingest", response_model=DocumentIngestResponse, status_code=200)
async def submit_ingestion_job(
    request: Request,
    tenant_context: TenantContextDep,
) -> DocumentIngestResponse:
    """
    Submit a document for ingestion (supports both multipart/form-data UploadFile from UI and JSON DocumentIngestRequest).
    """
    content_type = request.headers.get("content-type", "").lower()
    if "multipart/form-data" in content_type:
        form = await request.form()
        file = form.get("file")
        if not file or not hasattr(file, "filename"):
            raise HTTPException(status_code=400, detail="Missing file in form upload")
        document_id = str(form.get("document_id") or f"doc_{uuid.uuid4().hex[:8]}")
        target_tenant = str(form.get("tenant_id") or tenant_context.tenant_id)
        return await gateway_orchestrator.ingest_upload_file(file, document_id, target_tenant, tenant_context)
    else:
        try:
            data = await request.json()
            payload = DocumentIngestRequest(**data)
            return gateway_orchestrator.ingest_document(payload, tenant_context)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc}") from exc



@router.post("/query", response_model=QueryResponse, status_code=200)
async def process_rag_query(
    payload: QueryRequest,
    tenant_context: TenantContextDep,
) -> QueryResponse:
    """
    Execute the self-correcting RAG pipeline across Retrieval, Correction, and Generation.
    Enforces JWT authentication and document-level RBAC restrictions before retrieval.
    """
    return await gateway_orchestrator.process_query(payload, tenant_context)


@router.get("/status/{job_id}", response_model=JobStatusResponse, status_code=200)
async def check_job_status(
    job_id: str,
    tenant_context: TenantContextDep,
) -> JobStatusResponse:
    """
    Poll the execution status and progress of an asynchronous ingestion batch job.
    Enforces JWT authentication and tenant ownership verification.
    """
    return gateway_orchestrator.get_job_status(job_id, tenant_context)
