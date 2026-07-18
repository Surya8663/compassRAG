from fastapi import APIRouter

from shared.models.common import GenerationRequest, GenerationResponse

router = APIRouter(prefix="/generate", tags=["generation"])


@router.post("", response_model=GenerationResponse, status_code=200)
async def generate_answer(payload: GenerationRequest) -> GenerationResponse:
    """
    Scaffolding endpoint for grounded LLM answer generation.
    TODO: Implement LLM synthesis using LLM_MODEL_NAME grounded strictly on
    payload.chunks with citation tracking and confidence estimation.
    """
    return GenerationResponse(
        answer="Scaffolding generation placeholder: answer will be generated from context chunks.",
        sources=[chunk.id for chunk in payload.chunks],
        confidence=1.0,
    )
