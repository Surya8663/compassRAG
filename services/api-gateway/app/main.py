from fastapi import FastAPI
from shared.config import get_settings
from shared.logging import RequestIdMiddleware, setup_logging

from app.api.health import router as health_router
from app.api.routes import router as gateway_router

# Initialize structured logging and settings before app startup
settings = get_settings()
setup_logging(
    service_name="compass-rag-api-gateway",
    log_level=settings.LOG_LEVEL,
    json_logs=settings.JSON_LOGS,
)

app = FastAPI(
    title="Compass RAG - API Gateway",
    description="Main entrypoint orchestrating the self-correcting RAG pipeline across services",
    version="0.1.0",
)

# Add distributed tracing middleware
app.add_middleware(RequestIdMiddleware, service_name="compass-rag-api-gateway")

# Include API routes
app.include_router(health_router)
app.include_router(gateway_router)
