"""
LangGraph State definition (`CorrectionGraphState`) for the Correction Router.
Tracks query formulation across retry attempts, retrieval confidence,
contradictions, groundedness evaluation, and audit verdicts.
"""

from typing import TypedDict

from shared.models.common import Citation, ConfidenceStatus, CorrectionVerdict, RetrievalResult


class CorrectionGraphState(TypedDict):
    """
    State dictionary passed between LangGraph nodes across execution steps.
    """

    query: str
    tenant_id: str
    original_query: str
    attempt_count: int
    retrieved_chunks: list[RetrievalResult]
    retrieval_confidence: float
    retrieval_status: ConfidenceStatus
    contradictions_detected: bool
    contradiction_reasoning: str
    has_same_date_contradiction: bool
    draft_answer: str
    draft_citations: list[Citation]
    groundedness_score: float
    groundedness_verdict: bool
    verdicts: list[CorrectionVerdict]
    final_answer: str
    final_status: ConfidenceStatus
