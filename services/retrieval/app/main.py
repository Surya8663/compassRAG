from fastapi import FastAPI
from shared.config import get_settings
from shared.logging import RequestIdMiddleware, setup_logging
from shared.metrics import setup_metrics
from shared.telemetry import instrument_fastapi, setup_telemetry

from app.api.health import router as health_router
from app.api.routes import router as retrieval_router

# Initialize structured logging and settings before app startup
settings = get_settings()
setup_logging(
    service_name="compass-rag-retrieval", log_level=settings.LOG_LEVEL, json_logs=settings.JSON_LOGS
)

# Initialize OpenTelemetry distributed tracing
setup_telemetry(
    service_name="compass-rag-retrieval",
    otlp_endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
    enabled=settings.OTEL_ENABLED,
)

app = FastAPI(
    title="Compass RAG - Retrieval Service",
    description="Service for dense vectors and sparse keyword search over indexed document chunks",
    version="0.1.0",
)

# Apply FastAPI auto-instrumentation (must run before routes are registered)
instrument_fastapi(app)

# Add request-id / structlog / Prometheus middleware
app.add_middleware(RequestIdMiddleware, service_name="compass-rag-retrieval")

# Include API routes
app.include_router(health_router)
app.include_router(retrieval_router)

# Expose /metrics endpoint
if settings.PROMETHEUS_ENABLED:
    setup_metrics(app, service_name="compass-rag-retrieval")
