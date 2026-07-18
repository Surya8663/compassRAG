import os
from collections.abc import Generator

import pytest


@pytest.fixture(scope="session", autouse=True)
def mock_env_variables() -> Generator[None, None, None]:
    """
    Provide default required environment variables for all test suites
    so pydantic-settings loads cleanly during tests.
    """
    env_vars = {
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
        "MAX_RETRIES": "3",
        "INGESTION_SERVICE_URL": "http://localhost:8001",
        "RETRIEVAL_SERVICE_URL": "http://localhost:8002",
        "CORRECTION_SERVICE_URL": "http://localhost:8003",
        "GENERATION_SERVICE_URL": "http://localhost:8004",
    }
    for k, v in env_vars.items():
        os.environ[k] = v

    yield
