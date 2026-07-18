from typing import Any

from pydantic import BaseModel, Field


class HealthCheckResponse(BaseModel):
    """
    Standardized health check response returned by all services.
    """

    status: str = Field(..., description="Status of the service (e.g., 'ok')")
    service: str = Field(..., description="Name of the service responding")
    environment: str = Field(..., description="Running environment (e.g., development, production)")
    version: str = Field(default="0.1.0", description="Version of the service build")


class DocumentChunk(BaseModel):
    """
    Represents a discrete text segment with associated metadata and optional embedding.
    """

    id: str = Field(..., description="Unique identifier for the document chunk")
    content: str = Field(..., description="Text content of the chunk")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary key-value metadata"
    )
    score: float | None = Field(
        default=None, description="Optional relevance score assigned during retrieval"
    )


class RetrievalQuery(BaseModel):
    """
    Query request payload passed to the Retrieval service.
    """

    query: str = Field(..., description="User search query or refined self-correction query")
    top_k: int = Field(..., description="Maximum number of chunks to retrieve")


class RetrievalResult(BaseModel):
    """
    Response payload returned by the Retrieval service.
    """

    chunks: list[DocumentChunk] = Field(
        default_factory=list, description="List of retrieved chunks"
    )
    total_found: int = Field(default=0, description="Total number of matching chunks found")


class CorrectionRequest(BaseModel):
    """
    Request payload passed to the Correction service to evaluate context relevance
    or refine a query.
    """

    query: str = Field(..., description="Original user query being evaluated")
    retrieved_chunks: list[DocumentChunk] = Field(
        ..., description="Chunks returned by retrieval step"
    )


class CorrectionResult(BaseModel):
    """
    Response payload returned by the self-evaluator/correction service.
    """

    is_valid: bool = Field(
        ..., description="Whether retrieved context sufficiently answers the query"
    )
    confidence_score: float = Field(
        ..., description="Evaluator confidence score between 0.0 and 1.0"
    )
    reasoning: str = Field(..., description="Explanation of evaluation decision")
    refined_query: str | None = Field(
        default=None, description="Suggested refined query if not valid"
    )


class GenerationRequest(BaseModel):
    """
    Request payload passed to the Generation service.
    """

    query: str = Field(..., description="User query to answer")
    chunks: list[DocumentChunk] = Field(
        ..., description="Verified context chunks to ground generation"
    )


class GenerationResponse(BaseModel):
    """
    Response payload returned by the Generation service and API Gateway.
    """

    answer: str = Field(..., description="Final synthesized answer")
    sources: list[str] = Field(default_factory=list, description="List of source document IDs used")
    confidence: float = Field(..., description="Overall pipeline confidence score")
