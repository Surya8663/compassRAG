from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Strict configuration settings loaded exclusively from environment variables or .env file.
    No connection strings, model names, or thresholds are hardcoded.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --------------------------------------------------------------------------
    # 1. GENERAL SERVICE CONFIGURATION
    # --------------------------------------------------------------------------
    ENVIRONMENT: str = Field(
        ...,
        description="Environment name (e.g., development, staging, production)",
    )
    LOG_LEVEL: str = Field(
        ...,
        description="Global log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    JSON_LOGS: bool = Field(
        ...,
        description="Whether to output structured JSON logs",
    )

    # --------------------------------------------------------------------------
    # 2. DATABASE & DATA STORE CONNECTION STRINGS
    # --------------------------------------------------------------------------
    POSTGRES_DSN: str = Field(
        ...,
        description="PostgreSQL Data Source Name (asyncpg driver)",
    )
    REDIS_URL: str = Field(
        ...,
        description="Redis connection URL for caching and messaging",
    )
    QDRANT_URL: str = Field(
        ...,
        description="Qdrant vector database HTTP endpoint URL",
    )
    ELASTICSEARCH_URL: str = Field(
        ...,
        description="Elasticsearch cluster URL for BM25 and hybrid search",
    )

    # --------------------------------------------------------------------------
    # 3. AI MODELS & EMBEDDINGS CONFIGURATION
    # --------------------------------------------------------------------------
    EMBEDDING_MODEL_NAME: str = Field(
        ...,
        description="Embedding model name used for indexing and queries",
    )
    EMBEDDING_DIMENSION: int = Field(
        ...,
        description="Dimension size of vector embeddings produced by the model",
    )
    LLM_MODEL_NAME: str = Field(
        ...,
        description="Primary Large Language Model name used for generation and evaluation",
    )

    # --------------------------------------------------------------------------
    # 4. RAG PIPELINE & SELF-CORRECTION THRESHOLDS
    # --------------------------------------------------------------------------
    SIMILARITY_THRESHOLD: float = Field(
        ...,
        description="Minimum cosine similarity score required for chunk inclusion",
    )
    RETRIEVAL_TOP_K: int = Field(
        ...,
        description="Number of top candidate chunks to retrieve per search query",
    )
    CORRECTION_CONFIDENCE_THRESHOLD: float = Field(
        ...,
        description="Minimum confidence score required for self-correction validation",
    )
    MAX_RETRIES: int = Field(
        ...,
        description="Maximum number of query rewrite / self-correction attempts",
    )

    # --------------------------------------------------------------------------
    # 5. SERVICE-TO-SERVICE HTTP COMMUNICATION URLS
    # --------------------------------------------------------------------------
    INGESTION_SERVICE_URL: str = Field(
        ...,
        description="Internal base URL for the Ingestion service",
    )
    RETRIEVAL_SERVICE_URL: str = Field(
        ...,
        description="Internal base URL for the Retrieval service",
    )
    CORRECTION_SERVICE_URL: str = Field(
        ...,
        description="Internal base URL for the Correction service",
    )
    GENERATION_SERVICE_URL: str = Field(
        ...,
        description="Internal base URL for the Generation service",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Cached helper to retrieve application settings singleton.
    """
    return Settings()  # type: ignore[call-arg]
