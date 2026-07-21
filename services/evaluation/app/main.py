"""
Evaluation Service entry point (`services/evaluation/app/main.py`).
Exposes the FastAPI application and `/v1/evaluate` endpoints.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from shared.config import get_settings
from shared.metrics import setup_metrics
from shared.telemetry import instrument_fastapi, setup_telemetry

from services.evaluation.app.api.routes import router as evaluation_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("compass_rag_evaluation")

settings = get_settings()

# Initialize OpenTelemetry distributed tracing
setup_telemetry(
    service_name="compass-rag-evaluation",
    otlp_endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
    enabled=settings.OTEL_ENABLED,
)

app = FastAPI(
    title="Compass RAG Evaluation Service",
    version="0.1.0",
    description="Benchmarks baseline RAG vs self-correcting RAG across golden datasets.",
)

# Apply FastAPI auto-instrumentation
instrument_fastapi(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(evaluation_router)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Health check endpoint for the evaluation service."""
    return {"status": "ok", "service": "evaluation-service"}


# Expose /metrics endpoint
if settings.PROMETHEUS_ENABLED:
    setup_metrics(app, service_name="compass-rag-evaluation")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8005, reload=True)
