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
