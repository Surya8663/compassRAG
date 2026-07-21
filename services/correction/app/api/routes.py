"""
API routes for Correction Service (`compass-rag-correction`).
Exposes `/correct` endpoint executing the LangGraph StateGraph (`CorrectionRouterGraph`).
"""

import logging
from typing import Any

from fastapi import APIRouter
from shared.models.common import (
    ConfidenceStatus,
    CorrectionRequest,
    CorrectionResult,
    RetrievalResult,
)

from app.services.graph import get_correction_graph
from app.services.state import CorrectionGraphState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/correct", tags=["correction"])


@router.post("", response_model=CorrectionResult, status_code=200)
async def evaluate_and_correct(payload: CorrectionRequest) -> CorrectionResult:
    """
    Executes the Correction Router LangGraph StateGraph (`CorrectionRouterGraph`).
    Evaluates candidate chunks, checks for contradictions, verifies groundedness,
    and performs capped query reformulation or generates clarifying fallbacks.
    """
    graph = get_correction_graph()

    # Convert raw DocumentChunk items from payload into RetrievalResult wrappers
    wrapped_chunks = [
        RetrievalResult(
            chunk=c,
            vector_score=1.0,
            bm25_score=1.0,
            fused_score=1.0,
            rerank_score=1.0,
        )
        for c in payload.retrieved_chunks
    ]

    initial_state: CorrectionGraphState = {
        "query": payload.query,
        "tenant_id": getattr(payload, "tenant_id", "default_tenant"),
        "original_query": payload.query,
        "attempt_count": 0,
        "retrieved_chunks": wrapped_chunks,
        "retrieval_confidence": 1.0 if wrapped_chunks else 0.0,
        "retrieval_status": (
            ConfidenceStatus.VERIFIED if wrapped_chunks else ConfidenceStatus.LOW_CONFIDENCE
        ),
        "contradictions_detected": False,
        "contradiction_reasoning": "",
        "has_same_date_contradiction": False,
        "draft_answer": "",
        "draft_citations": [],
        "groundedness_score": 0.0,
        "groundedness_verdict": False,
        "verdicts": [],
        "final_answer": "",
        "final_status": ConfidenceStatus.VERIFIED,
    }

    try:
        result_state: dict[str, Any] = await graph.ainvoke(initial_state)
    except Exception as exc:
        logger.error("Correction Router graph execution failed: %s", exc)
        return CorrectionResult(
            is_valid=False,
            confidence_score=0.0,
            reasoning=f"Correction pipeline execution error: {exc}",
            verdicts=[],
            refined_query=None,
        )

    final_status = result_state.get("final_status", ConfidenceStatus.VERIFIED)
    is_valid = final_status == ConfidenceStatus.VERIFIED
    score = float(
        result_state.get("groundedness_score") or result_state.get("retrieval_confidence") or 0.0
    )

    reasoning_parts = []
    if result_state.get("contradiction_reasoning"):
        reasoning_parts.append(f"Contradictions: {result_state['contradiction_reasoning']}")
    if result_state.get("final_answer"):
        reasoning_parts.append(f"Final output status: {final_status.value}")
    if not reasoning_parts:
        reasoning_parts.append("Correction pipeline completed evaluation and synthesis.")

    reasoning = " | ".join(reasoning_parts)

    current_query = result_state.get("query", payload.query)
    refined = current_query if current_query != payload.query else None

    return CorrectionResult(
        is_valid=is_valid,
        confidence_score=score,
        reasoning=reasoning,
        verdicts=result_state.get("verdicts", []),
        refined_query=refined,
    )
