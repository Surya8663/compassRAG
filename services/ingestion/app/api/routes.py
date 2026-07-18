from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/ingest", tags=["ingestion"])


class IngestionRequest(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the document")
    text: str = Field(..., description="Raw document text to ingest")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class IngestionResponse(BaseModel):
    status: str = Field(..., description="Status of ingestion job")
    document_id: str = Field(..., description="Ingested document identifier")
    chunks_indexed: int = Field(default=0, description="Number of chunks created and indexed")


@router.post("", response_model=IngestionResponse, status_code=202)
async def ingest_document(payload: IngestionRequest) -> IngestionResponse:
    """
    Scaffolding endpoint for document ingestion.
    TODO: Implement document chunking, embedding generation via EMBEDDING_MODEL_NAME,
    and indexing into Qdrant and Elasticsearch.
    """
    return IngestionResponse(
        status="accepted",
        document_id=payload.document_id,
        chunks_indexed=0,
    )
