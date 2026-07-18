from fastapi import APIRouter

from shared.config import get_settings
from shared.models.common import HealthCheckResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthCheckResponse, status_code=200)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint returning 200 OK and service status details.
    """
    settings = get_settings()
    return HealthCheckResponse(
        status="ok",
        service="compass-rag-generation",
        environment=settings.ENVIRONMENT,
        version="0.1.0",
    )
