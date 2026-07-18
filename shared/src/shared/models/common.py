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
