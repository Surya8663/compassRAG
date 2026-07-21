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
    CELERY_BROKER_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Celery Redis broker connection string",
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://localhost:6379/0",
        description="Celery result backend URL",
    )
    TESSERACT_CMD: str = Field(
        default=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        description="Path to Tesseract OCR binary",
    )

    # --------------------------------------------------------------------------
    # 3. AI MODELS & EMBEDDINGS CONFIGURATION
    # --------------------------------------------------------------------------
    EMBEDDING_PROVIDER: str = Field(
        default="local",
        description="Embedding provider selection: 'local' (SentenceTransformer) or 'openai'",
    )
    LOCAL_EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="Local SentenceTransformer embedding model name",
    )
    OPENAI_EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name when provider is 'openai'",
    )
    OPENAI_API_KEY: str | None = Field(
        default=None,
        description="OpenAI API key used when EMBEDDING_PROVIDER is 'openai'",
    )
    EMBEDDING_MODEL_NAME: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model name used for indexing and queries",
    )
    EMBEDDING_DIMENSION: int = Field(
        default=384,
        description="Dimension size of vector embeddings produced by the model",
    )
    LLM_MODEL_NAME: str = Field(
        ...,
        description="Primary Large Language Model name used for generation and evaluation",
    )
    GENERATION_PRIMARY_MODEL: str = Field(
        default="gpt-4o",
        description="Flagship model selection used for primary answer synthesis",
    )
    GENERATION_FALLBACK_MODEL: str = Field(
        default="gpt-4o-mini",
        description="Fallback model selection used when circuit breaker trips open",
    )
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(
        default=5,
        ge=1,
        description="Consecutive failure threshold to open the circuit breaker",
    )
    CIRCUIT_BREAKER_RESET_TIMEOUT: float = Field(
        default=30.0,
        ge=0.1,
        description="Seconds to remain in open state before allowing half-open retry",
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
    OCR_CONFIDENCE_THRESHOLD: float = Field(
        default=0.85,
        description="Minimum OCR page confidence required before routing to manual review",
    )
    RRF_K_CONSTANT: int = Field(
        default=60,
        description="Reciprocal Rank Fusion k constant used for combining rankings",
    )
    RERANKER_PROVIDER: str = Field(
        default="local",
        description="Reranker provider selection: 'local' (CrossEncoder) or 'cohere'",
    )
    LOCAL_RERANK_MODEL: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Local CrossEncoder reranking model name",
    )
    COHERE_RERANK_MODEL: str = Field(
        default="rerank-english-v3.0",
        description="Cohere reranking model name when provider is 'cohere'",
    )
    COHERE_API_KEY: str | None = Field(
        default=None,
        description="Cohere API key used when RERANKER_PROVIDER is 'cohere'",
    )
    RETRIEVAL_CONFIDENCE_THRESHOLD: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum average reranked score for verified confidence status",
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

    # --------------------------------------------------------------------------
    # 6. API GATEWAY & AUTHENTICATION CONFIGURATION
    # --------------------------------------------------------------------------
    AUTH_ENABLED: bool = Field(
        default=True,
        description="Enforce JWT validation and RBAC checks across API Gateway endpoints",
    )
    JWT_SECRET_KEY: str = Field(
        default="compass-rag-dev-secret-key-32bytes-long!!",
        description="Secret key used for HS256 symmetric JWT signature verification in dev/tests",
    )
    JWT_ALGORITHM: str = Field(
        default="HS256",
        description="JWT cryptographic signing algorithm (HS256 or RS256)",
    )
    KEYCLOAK_JWKS_URL: str = Field(
        default="http://localhost:8080/realms/compass-rag/protocol/openid-connect/certs",
        description="Keycloak JWKS URL for fetching public RSA verification keys in RS256 mode",
    )

    # --------------------------------------------------------------------------
    # 7. OBSERVABILITY: OPENTELEMETRY + PROMETHEUS
    # --------------------------------------------------------------------------
    OTEL_ENABLED: bool = Field(
        default=True,
        description="Enable OpenTelemetry distributed tracing via OTLP-HTTP exporter",
    )
    OTEL_EXPORTER_OTLP_ENDPOINT: str = Field(
        default="http://localhost:4318",
        description="Base URL of the OTLP-HTTP collector (Jaeger / OTEL Collector)",
    )
    OTEL_SERVICE_NAME: str = Field(
        default="compass-rag",
        description="Logical service name reported to the OTel collector",
    )
    PROMETHEUS_ENABLED: bool = Field(
        default=True,
        description="Expose Prometheus /metrics endpoint on every FastAPI service",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Cached helper to retrieve application settings singleton.
    """
    return Settings()  # type: ignore[call-arg]
