"""
Request Router endpoints (`/v1/ingest`, `/v1/query`, `/v1/status/{job_id}`) for API Gateway.
Enforces strict JWT authentication and delegates to `GatewayOrchestratorService`.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
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
    payload: DocumentIngestRequest,
    tenant_context: TenantContextDep,
) -> DocumentIngestResponse:
    """
    Submit a multi-page document for asynchronous ingestion via Celery worker queue.
    Enforces JWT authentication and tenant scope.
    """
    return gateway_orchestrator.ingest_document(payload, tenant_context)


@router.post("/query", response_model=QueryResponse, status_code=200)
async def process_rag_query(
    payload: QueryRequest,
    tenant_context: TenantContextDep,
) -> QueryResponse:
    """
    Execute the self-correcting RAG pipeline across Retrieval, Correction, and Generation.
    Enforces JWT authentication and document-level RBAC restrictions before retrieval.
    """
    return gateway_orchestrator.process_query(payload, tenant_context)


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
