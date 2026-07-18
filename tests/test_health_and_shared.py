import pytest

from shared.config import get_settings
from shared.models.common import HealthCheckResponse


def test_settings_load() -> None:
    """
    Verify that pydantic-settings loads strictly from environment variables without hardcoding.
    """
    settings = get_settings()
    assert settings.ENVIRONMENT == "testing"
    assert settings.EMBEDDING_DIMENSION == 1024
    assert settings.SIMILARITY_THRESHOLD == 0.75
    assert "postgresql+asyncpg" in settings.POSTGRES_DSN


@pytest.mark.asyncio
async def test_health_check_response_schema() -> None:
    """
    Verify shared HealthCheckResponse model instantiation and validation.
    """
    response = HealthCheckResponse(
        status="ok",
        service="compass-rag-shared-test",
        environment="testing",
        version="0.1.0",
    )
    assert response.status == "ok"
    assert response.service == "compass-rag-shared-test"
