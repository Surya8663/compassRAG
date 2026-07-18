from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.routes import router as correction_router
from shared.config import get_settings
from shared.logging import RequestIdMiddleware, setup_logging

# Initialize structured logging and settings before app startup
settings = get_settings()
setup_logging(
    service_name="compass-rag-correction",
    log_level=settings.LOG_LEVEL,
    json_logs=settings.JSON_LOGS,
)

app = FastAPI(
    title="Compass RAG - Correction Service",
    description="Self-correcting evaluator and query refinement service",
    version="0.1.0",
)

# Add distributed tracing middleware
app.add_middleware(RequestIdMiddleware, service_name="compass-rag-correction")

# Include API routes
app.include_router(health_router)
app.include_router(correction_router)
