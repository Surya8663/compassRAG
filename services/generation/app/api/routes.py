from fastapi import APIRouter
from shared.models.common import GenerationRequest, GenerationResponse

from app.services import get_generation_service

router = APIRouter(prefix="/generate", tags=["generation"])


@router.post("", response_model=GenerationResponse, status_code=200)
async def generate_answer(payload: GenerationRequest) -> GenerationResponse:
    """
    Endpoint for grounded LLM answer generation and citation tracking.
    Executes circuit breaker resilience and mandatory groundedness checking.
    """
    service = get_generation_service()
    return service.generate(query=payload.query, chunks=payload.chunks)
