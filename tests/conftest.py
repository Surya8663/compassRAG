import os
import sys
from collections.abc import Generator
from pathlib import Path

# Ensure monorepo root, shared/src, and services are on sys.path before imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "shared" / "src"))
sys.path.insert(0, str(ROOT_DIR / "services" / "correction"))
sys.path.insert(0, str(ROOT_DIR / "services" / "retrieval"))
sys.path.insert(0, str(ROOT_DIR / "services" / "ingestion"))
sys.path.insert(0, str(ROOT_DIR / "services" / "generation"))
sys.path.insert(0, str(ROOT_DIR / "services" / "api-gateway"))

import pytest  # noqa: E402
from shared.config import get_settings  # noqa: E402

# Set default required environment variables at import time before modules are collected
TEST_ENV_VARS = {
    "ENVIRONMENT": "testing",
    "LOG_LEVEL": "DEBUG",
    "JSON_LOGS": "False",
    "POSTGRES_DSN": "postgresql+asyncpg://postgres:postgres@localhost:5432/compass_rag_test",
    "REDIS_URL": "redis://localhost:6379/1",
    "QDRANT_URL": "http://localhost:6333",
    "ELASTICSEARCH_URL": "http://localhost:9200",
    "EMBEDDING_MODEL_NAME": "BAAI/bge-large-en-v1.5",
    "EMBEDDING_DIMENSION": "1024",
    "LLM_MODEL_NAME": "gpt-4o-mini",
    "SIMILARITY_THRESHOLD": "0.75",
    "RETRIEVAL_TOP_K": "10",
    "CORRECTION_CONFIDENCE_THRESHOLD": "0.80",
    "OCR_CONFIDENCE_THRESHOLD": "0.85",
    "MAX_RETRIES": "3",
    "INGESTION_SERVICE_URL": "http://localhost:8001",
    "RETRIEVAL_SERVICE_URL": "http://localhost:8002",
    "CORRECTION_SERVICE_URL": "http://localhost:8003",
    "GENERATION_SERVICE_URL": "http://localhost:8004",
    "CELERY_BROKER_URL": "redis://localhost:6379/0",
    "CELERY_RESULT_BACKEND": "redis://localhost:6379/0",
    "TESSERACT_CMD": r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    "AUTH_ENABLED": "True",
    "JWT_SECRET_KEY": "compass-rag-dev-secret-key-32bytes-long!!",
    "JWT_ALGORITHM": "HS256",
    "KEYCLOAK_JWKS_URL": "http://localhost:8080/realms/compass-rag/protocol/openid-connect/certs",
    # Observability — disable OTel exporting in tests; Prometheus metrics still work
    "OTEL_ENABLED": "False",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318",
    "OTEL_SERVICE_NAME": "compass-rag-test",
    "PROMETHEUS_ENABLED": "True",
}
for k, v in TEST_ENV_VARS.items():
    os.environ[k] = v


@pytest.fixture(scope="session", autouse=True)
def mock_env_variables() -> Generator[None, None, None]:
    """
    Provide default required environment variables for all test suites
    so pydantic-settings loads cleanly during tests.
    """
    for k_var, v_var in TEST_ENV_VARS.items():
        os.environ[k_var] = v_var
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
