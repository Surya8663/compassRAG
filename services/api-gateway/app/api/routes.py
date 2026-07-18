from fastapi import APIRouter
from pydantic import BaseModel, Field
from shared.models.common import GenerationResponse

router = APIRouter(prefix="/query", tags=["gateway"])


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question or query to answer")
    top_k: int = Field(default=10, description="Number of chunks to retrieve")


@router.post("", response_model=GenerationResponse, status_code=200)
async def process_rag_query(payload: QueryRequest) -> GenerationResponse:
    """
    Scaffolding endpoint orchestrating the end-to-end self-correcting RAG pipeline via HTTP calls:
    1. Retrieval -> 2. Correction/Evaluation -> 3. Generation (with query rewrite retry loop).
    """
    return GenerationResponse(
        answer="Scaffolding API Gateway placeholder: pipeline workflow orchestrator initialized.",
        sources=[],
        confidence=1.0,
    )
