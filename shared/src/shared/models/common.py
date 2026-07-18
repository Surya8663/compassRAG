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
