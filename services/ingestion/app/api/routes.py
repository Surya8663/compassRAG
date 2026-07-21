import os
import shutil
import uuid
from typing import Any

import fitz
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.db.models import DocumentBatch, DocumentPage
from app.db.session import get_sync_session
from app.workers.tasks import process_document_page
from shared.tenant import resolve_tenant_id

router = APIRouter(prefix="/ingest", tags=["ingestion"])

STAGING_DIR = os.path.join(os.getcwd(), ".staging")
os.makedirs(STAGING_DIR, exist_ok=True)


class IngestionRequest(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the document")
    text: str = Field(..., description="Raw document text to ingest")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class IngestionResponse(BaseModel):
    status: str = Field(..., description="Status of ingestion job")
    document_id: str = Field(..., description="Ingested document identifier")
    chunks_indexed: int = Field(default=0, description="Number of chunks created and indexed")


class BatchUploadResponse(BaseModel):
    batch_id: str = Field(..., description="Unique identifier for the ingestion batch")
    document_id: str = Field(..., description="Ingested document identifier")
    filename: str = Field(..., description="Original filename of uploaded document")
    expected_pages: int = Field(..., description="Total number of pages detected")
    received_pages: int = Field(default=0, description="Pages processed so far")
    status: str = Field(..., description="Current status of the batch job")


class PageStatusResponse(BaseModel):
    page_number: int
    status: str
    classification: str | None = None
    ocr_confidence: float | None = None
    error_message: str | None = None


class BatchStatusResponse(BaseModel):
    batch_id: str
    document_id: str
    filename: str
    expected_pages: int
    received_pages: int
    status: str
    pages: list[PageStatusResponse] = Field(default_factory=list)


@router.post("", response_model=IngestionResponse, status_code=202)
async def ingest_document(payload: IngestionRequest) -> IngestionResponse:
    """
    Scaffolding endpoint for direct text ingestion.
    """
    return IngestionResponse(
        status="accepted",
        document_id=payload.document_id,
        chunks_indexed=0,
    )


@router.post("/upload", response_model=BatchUploadResponse, status_code=202)
async def upload_document_batch(
    file: UploadFile = File(...),  # noqa: B008
    document_id: str = Form(...),  # noqa: B008
    tenant_id: str | None = Form(default=None),  # noqa: B008
) -> BatchUploadResponse:
    """
    Real ingestion endpoint accepting a multi-page PDF upload.
    Initializes a DocumentBatch in Postgres and processes each page synchronously or via Celery.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename")

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are currently supported")

    batch_id = str(uuid.uuid4())
    staging_filename = f"{batch_id}_{file.filename}"
    file_path = os.path.join(STAGING_DIR, staging_filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to save uploaded file: {exc}"
        ) from exc

    try:
        with fitz.open(file_path) as doc:
            expected_pages = len(doc)
            if expected_pages == 0:
                raise HTTPException(status_code=400, detail="Uploaded PDF contains zero pages")
    except Exception as exc:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=400, detail=f"Invalid or corrupted PDF file: {exc}"
        ) from exc

    session = get_sync_session()
    try:
        batch = DocumentBatch(
            id=batch_id,
            document_id=document_id,
            filename=file.filename,
            expected_pages=expected_pages,
            received_pages=0,
            status="PROCESSING",
        )
        session.add(batch)

        for p_num in range(1, expected_pages + 1):
            page_record = DocumentPage(
                batch_id=batch_id,
                page_number=p_num,
                status="PROCESSING",
            )
            session.add(page_record)

        session.commit()
    except Exception as exc:
        session.rollback()
        raise HTTPException(
            status_code=500, detail=f"Database error registering batch: {exc}"
        ) from exc
    finally:
        session.close()

    target_tenant = resolve_tenant_id(explicit_tenant_id=tenant_id)

    # Execute processing right inside the route when running locally without a separate Celery worker process
    import asyncio
    for p_num in range(1, expected_pages + 1):
        await asyncio.to_thread(process_document_page, batch_id, p_num, file_path, document_id, target_tenant)

    return BatchUploadResponse(
        batch_id=batch_id,
        document_id=document_id,
        filename=file.filename,
        expected_pages=expected_pages,
        received_pages=expected_pages,
        status="COMPLETED",
    )


@router.get("/batches/{batch_id}", response_model=BatchStatusResponse)
async def get_batch_status(batch_id: str) -> BatchStatusResponse:
    """
    Retrieves current progress and status of a document ingestion batch.
    """
    session = get_sync_session()
    try:
        batch = session.query(DocumentBatch).filter(DocumentBatch.id == batch_id).first()
        if not batch:
            raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

        page_statuses = [
            PageStatusResponse(
                page_number=p.page_number,
                status=p.status,
                classification=p.classification,
                ocr_confidence=p.ocr_confidence,
                error_message=p.error_message,
            )
            for p in sorted(batch.pages, key=lambda x: x.page_number)
        ]

        return BatchStatusResponse(
            batch_id=batch.id,
            document_id=batch.document_id,
            filename=batch.filename,
            expected_pages=batch.expected_pages,
            received_pages=batch.received_pages,
            status=batch.status,
            pages=page_statuses,
        )
    finally:
        session.close()
