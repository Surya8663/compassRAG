# Shared module for Compass RAG services
from shared.config import Settings, get_settings
from shared.logging import RequestIdMiddleware, setup_logging
from shared.models.common import (
    CorrectionRequest,
    CorrectionResult,
    DocumentChunk,
    GenerationRequest,
    GenerationResponse,
    HealthCheckResponse,
    RetrievalQuery,
    RetrievalResult,
)

__all__ = [
    "Settings",
    "get_settings",
    "setup_logging",
    "RequestIdMiddleware",
    "HealthCheckResponse",
    "DocumentChunk",
    "RetrievalQuery",
    "RetrievalResult",
    "CorrectionRequest",
    "CorrectionResult",
    "GenerationRequest",
    "GenerationResponse",
]
