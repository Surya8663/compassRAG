from fastapi import FastAPI
from shared.config import get_settings
from shared.logging import RequestIdMiddleware, setup_logging

from app.api.health import router as health_router
from app.api.routes import router as ingestion_router
from app.db.session import init_db

# Initialize structured logging and settings before app startup
settings = get_settings()
setup_logging(
    service_name="compass-rag-ingestion", log_level=settings.LOG_LEVEL, json_logs=settings.JSON_LOGS
)

# Initialize database tables
try:
    init_db()
except Exception as e:
    # In local testing or without active DB server, log warning rather than crashing startup
    print(f"[WARN] Could not initialize database on startup: {e}")

app = FastAPI(
    title="Compass RAG - Ingestion Service",
    description="Service for parsing, chunking, and embedding indexing",
    version="0.1.0",
)

# Add distributed tracing middleware (`X-Request-ID` propagation)
app.add_middleware(RequestIdMiddleware, service_name="compass-rag-ingestion")

# Include API routes
app.include_router(health_router)
app.include_router(ingestion_router)
