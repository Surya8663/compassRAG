from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

# ==============================================================================
# ENUMS (Strict String Enums)
# ==============================================================================

class SignalType(StrEnum):
    """
    Self-correcting evaluation signal types.
    """
    GROUNDEDNESS = "GROUNDEDNESS"
    CONTRADICTION = "CONTRADICTION"


class ConfidenceStatus(StrEnum):
    """
    Status of the synthesized response confidence.
    """
    VERIFIED = "VERIFIED"
    CLARIFICATION_NEEDED = "CLARIFICATION_NEEDED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


# ==============================================================================
# TENANT & SECURITY CONTEXT
# ==============================================================================

class TenantContext(BaseModel):
    """
    Multi-tenant security context passed along with requests across services.
    Enforces tenant isolation and role-based access control.
    """
    tenant_id: str = Field(..., description="Unique identifier for the tenant")
    user_id: str = Field(..., description="Unique identifier for the user")
    roles: list[str] = Field(default_factory=list, description="Assigned user roles")
    permissions: list[str] = Field(default_factory=list, description="Granted permissions")


# ==============================================================================
# DOCUMENT & CHUNK MODELS
# ==============================================================================

class DocumentMetadata(BaseModel):
    """
    Strict metadata contract associated with ingested documents and chunks.
    Contains zero `Any` types to guarantee strict validation across pipelines.
    """
    source: str = Field(..., description="Source file path, URI, or origin identifier")
    page_number: int | None = Field(default=None, ge=1, description="1-indexed page number")
    ingestion_timestamp: datetime = Field(..., description="UTC timestamp of ingestion")
    ocr_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="OCR confidence (0.0 to 1.0)"
    )
    tenant_id: str = Field(..., description="Tenant organization identifier")
    version_id: str = Field(..., description="Document version identifier")


class Document(BaseModel):
    """
    Represents a full document before or during chunking.
    """
    id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Raw text content of the document")
    metadata: DocumentMetadata = Field(..., description="Strict metadata contract")


class DocumentChunk(BaseModel):
    """
    Discrete text chunk indexed into Qdrant and Elasticsearch.
    """
    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document identifier")
    content: str = Field(..., description="Text content of the chunk")
    chunk_index: int = Field(default=0, ge=0, description="Index position within document")
    metadata: DocumentMetadata = Field(..., description="Strict metadata contract")
    score: float | None = Field(default=None, description="Optional relevance score")


# ==============================================================================
# RETRIEVAL MODELS
# ==============================================================================

class RetrievalResult(BaseModel):
    """
    Detailed breakdown of a retrieved chunk along with vector, BM25, and fused scores.
    """
    chunk: DocumentChunk = Field(..., description="Retrieved document chunk")
    vector_score: float = Field(..., description="Vector similarity score from Qdrant")
    bm25_score: float = Field(..., description="Keyword relevance score from Elasticsearch")
    fused_score: float = Field(..., description="Reciprocal Rank Fusion or weighted score")
    rerank_score: float | None = Field(default=None, description="Cross-encoder rerank score")


class RetrievalQuery(BaseModel):
    """
    Query payload sent to the Retrieval service.
    """
    query: str = Field(..., description="User search or refined self-correction query")
    top_k: int = Field(default=10, ge=1, description="Maximum number of chunks to retrieve")
    tenant_id: str | None = Field(
        default=None, description="Optional tenant ID filter for vector isolation"
    )


class RetrievalResponse(BaseModel):
    """
    Response payload returned by the Retrieval service.
    """
    results: list[RetrievalResult] = Field(
        default_factory=list, description="Detailed retrieval results"
    )
    total_found: int = Field(default=0, ge=0, description="Total matching chunks found")


# ==============================================================================
# CORRECTION / SELF-EVALUATOR MODELS
# ==============================================================================

class CorrectionVerdict(BaseModel):
    """
    Evaluation verdict produced by the self-correcting RAG evaluator.
    """
    signal_type: SignalType = Field(
        ..., description="Signal evaluated (GROUNDEDNESS/CONTRADICTION)"
    )
    verdict: bool = Field(..., description="Whether check passed (True) or contradicted (False)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0 to 1.0)")
    reasoning: str = Field(..., description="Explanation of evaluation verdict")


class CorrectionRequest(BaseModel):
    """
    Request payload passed to Correction service to evaluate relevance or refine query.
    """
    query: str = Field(..., description="Original user query being evaluated")
    retrieved_chunks: list[DocumentChunk] = Field(..., description="Chunks from retrieval step")


class CorrectionResult(BaseModel):
    """
    Aggregate response payload returned by the self-evaluator/correction service.
    """
    is_valid: bool = Field(..., description="Whether context sufficiently answers query")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Evaluator confidence score (0.0 to 1.0)"
    )
    reasoning: str = Field(..., description="Explanation of evaluation decision")
    verdicts: list[CorrectionVerdict] = Field(
        default_factory=list, description="Granular evaluation verdicts"
    )
    refined_query: str | None = Field(
        default=None, description="Suggested refined query if invalid"
    )


# ==============================================================================
# GENERATION & QUERY GATEWAY MODELS
# ==============================================================================

class Citation(BaseModel):
    """
    Grounded citation attributing a portion of the generated answer to a DocumentChunk.
    """
    chunk_id: str = Field(..., description="Identifier of the source chunk")
    document_id: str = Field(..., description="Identifier of the parent document")
    source: str = Field(..., description="Source file path or URI")
    page_number: int | None = Field(default=None, ge=1, description="Page number where available")
    quote_snippet: str = Field(..., description="Relevant text snippet cited")


class QueryRequest(BaseModel):
    """
    Public query contract sent to the API Gateway.
    """
    query: str = Field(..., description="User question or query to answer")
    tenant_context: TenantContext = Field(..., description="Security context of user and tenant")
    top_k: int = Field(default=10, ge=1, description="Number of candidate chunks to retrieve")
    metadata_filter: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Strictly typed metadata key-value filters for retrieval isolation",
    )


class QueryResponse(BaseModel):
    """
    Public response contract returned by the API Gateway after pipeline execution.
    """
    answer: str = Field(..., description="Final synthesized answer")
    confidence_status: ConfidenceStatus = Field(..., description="Categorical confidence status")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Numerical confidence score (0.0 to 1.0)"
    )
    citations: list[Citation] = Field(default_factory=list, description="Grounded citations")
    verdicts: list[CorrectionVerdict] = Field(
        default_factory=list, description="Self-correction audit trail"
    )


class GenerationRequest(BaseModel):
    """
    Request payload passed to the Generation service.
    """
    query: str = Field(..., description="User query to answer")
    chunks: list[DocumentChunk] = Field(..., description="Verified chunks to ground generation")


class GenerationResponse(BaseModel):
    """
    Response payload returned by the Generation service.
    """
    answer: str = Field(..., description="Final synthesized answer")
    citations: list[Citation] = Field(default_factory=list, description="Grounded citations")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall generation confidence"
    )


# ==============================================================================
# COMMON HEALTH CHECK MODEL
# ==============================================================================

class HealthCheckResponse(BaseModel):
    """
    Standardized health check response returned by all services.
    """
    status: str = Field(..., description="Status of the service (e.g., 'ok')")
    service: str = Field(..., description="Name of the service responding")
    environment: str = Field(..., description="Running environment (e.g., dev, prod)")
    version: str = Field(default="0.1.0", description="Version of the service build")
