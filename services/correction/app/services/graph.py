"""
Correction Router LangGraph (`CorrectionRouterGraph`).
Compiles a StateGraph with 9 explicit nodes:
retrieve, evaluate_confidence, contradiction_check, generate_draft, groundedness_check,
reformulate_query, clarify, low_confidence_response, finalize_answer.
Routes dynamically based on computed confidence, contradiction verdicts, and retry budgets.
"""

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph
from shared.config import get_settings
from shared.models.common import Citation, ConfidenceStatus

from services.retrieval.app.services.evaluator import get_retrieval_evaluator
from services.retrieval.app.services.hybrid_retriever import get_hybrid_retriever

from .clarification import get_clarification_service
from .contradiction import get_contradiction_detector
from .groundedness import get_groundedness_checker
from .reformulator import get_query_reformulator
from .state import CorrectionGraphState

logger = logging.getLogger(__name__)


# ==============================================================================
# NODE DEFINITIONS
# ==============================================================================


async def retrieve_node(state: CorrectionGraphState) -> dict[str, Any]:
    """Node 1: Retrieves candidate chunks for `state['query']`."""
    settings = get_settings()

    # If state already has candidate chunks (e.g., passed from API payload on attempt 0),
    # use them directly before any query reformulation attempts
    if state.get("attempt_count", 0) == 0 and state.get("retrieved_chunks"):
        results = state["retrieved_chunks"]
        evaluator = get_retrieval_evaluator()
        avg_score, _, status, _ = evaluator.evaluate_confidence(
            results=results, threshold=settings.RETRIEVAL_CONFIDENCE_THRESHOLD
        )
        return {
            "retrieved_chunks": results,
            "retrieval_confidence": avg_score,
            "retrieval_status": status,
        }

    retriever = get_hybrid_retriever()
    query = state.get("query", "")
    tenant_id = state.get("tenant_id", "default_tenant")

    try:
        results, avg_score, status, _ = await retriever.retrieve(
            query=query, tenant_id=tenant_id, top_k=settings.RETRIEVAL_TOP_K
        )
    except Exception as exc:
        logger.warning("Retriever failed during graph traversal (`%s`); using empty.", exc)
        results, avg_score, status = [], 0.0, ConfidenceStatus.LOW_CONFIDENCE

    return {
        "retrieved_chunks": results,
        "retrieval_confidence": avg_score,
        "retrieval_status": status,
    }


def evaluate_confidence_node(state: CorrectionGraphState) -> dict[str, Any]:
    """Node 2: Evaluates retrieval confidence against threshold."""
    settings = get_settings()
    evaluator = get_retrieval_evaluator()
    chunks = state.get("retrieved_chunks", [])

    avg_score, is_conf, status, _ = evaluator.evaluate_confidence(
        results=chunks, threshold=settings.RETRIEVAL_CONFIDENCE_THRESHOLD
    )
    return {
        "retrieval_confidence": avg_score,
        "retrieval_status": status,
    }


def contradiction_check_node(state: CorrectionGraphState) -> dict[str, Any]:
    """Node 3: Checks for factual contradictions and reconciles temporal supersession."""
    detector = get_contradiction_detector()
    chunks = state.get("retrieved_chunks", [])

    detected, same_date, resolved, reasoning = detector.check_contradictions(chunks)
    return {
        "contradictions_detected": detected,
        "has_same_date_contradiction": same_date,
        "retrieved_chunks": resolved,
        "contradiction_reasoning": reasoning,
    }


def generate_draft_node(state: CorrectionGraphState) -> dict[str, Any]:
    """Node 4: Generates draft answer from verified and resolved candidate chunks."""
    chunks = state.get("retrieved_chunks", [])
    query = state.get("query", "")

    if not chunks:
        return {
            "draft_answer": f"No verified context available to answer: '{query}'.",
            "draft_citations": [],
        }

    # Draft synthesis based on top candidate content
    top_c = chunks[0].chunk
    citations = [
        Citation(
            chunk_id=top_c.id,
            document_id=top_c.document_id,
            source=top_c.metadata.source,
            page_number=top_c.metadata.page_number,
            quote_snippet=top_c.content[:100],
        )
    ]

    logger.debug("Generating draft answer for query: '%s'", query)
    draft = top_c.content
    return {
        "draft_answer": draft,
        "draft_citations": citations,
    }


def groundedness_check_node(state: CorrectionGraphState) -> dict[str, Any]:
    """Node 5: Decomposes draft answer into atomic claims and computes groundedness score."""
    checker = get_groundedness_checker()
    draft = state.get("draft_answer", "")
    chunks = state.get("retrieved_chunks", [])
    existing_verdicts = state.get("verdicts", [])

    score, is_grounded, verdicts, summary = checker.verify_groundedness(draft, chunks)
    return {
        "groundedness_score": score,
        "groundedness_verdict": is_grounded,
        "verdicts": existing_verdicts + verdicts,
    }


def reformulate_query_node(state: CorrectionGraphState) -> dict[str, Any]:
    """Node 6: Reformulates query using HyDE, Multi-Query, or Keyword Broadening."""
    reformulator = get_query_reformulator()
    query = state.get("query", "")
    attempts = state.get("attempt_count", 0)

    # Determine why we are reformulating
    if state.get("retrieval_status") == ConfidenceStatus.LOW_CONFIDENCE:
        reason = f"Low retrieval confidence ({state.get('retrieval_confidence', 0.0):.2f})"
    else:
        reason = f"Low groundedness score ({state.get('groundedness_score', 0.0):.2f})"

    new_query, strategy, _ = reformulator.reformulate(query, attempts, reason)
    return {
        "query": new_query,
        "attempt_count": attempts + 1,
    }


def clarify_node(state: CorrectionGraphState) -> dict[str, Any]:
    """Node 7: Generates clarifying question for unresolvable same-date contradictions."""
    service = get_clarification_service()
    orig_query = state.get("original_query") or state.get("query", "")
    chunks = state.get("retrieved_chunks", [])

    question = service.generate_clarification(orig_query, chunks)
    return {
        "final_answer": question,
        "final_status": ConfidenceStatus.CLARIFICATION_NEEDED,
    }


def low_confidence_response_node(state: CorrectionGraphState) -> dict[str, Any]:
    """Node 8: Generates transparent low-confidence response when retry limit reached."""
    service = get_clarification_service()
    orig_query = state.get("original_query") or state.get("query", "")
    attempts = state.get("attempt_count", 0)
    reasoning = (
        f"Retrieval score: {state.get('retrieval_confidence', 0.0):.2f}, "
        f"Groundedness score: {state.get('groundedness_score', 0.0):.2f}. "
        f"Contradiction check: {state.get('contradiction_reasoning', 'None')}"
    )

    resp = service.generate_low_confidence_response(orig_query, attempts, reasoning)
    return {
        "final_answer": resp,
        "final_status": ConfidenceStatus.LOW_CONFIDENCE,
    }


def finalize_answer_node(state: CorrectionGraphState) -> dict[str, Any]:
    """Node 9: Prepares final answer and status output."""
    if state.get("final_answer") and state.get("final_status") in (
        ConfidenceStatus.CLARIFICATION_NEEDED,
        ConfidenceStatus.LOW_CONFIDENCE,
    ):
        return {
            "final_answer": state["final_answer"],
            "final_status": state["final_status"],
        }

    return {
        "final_answer": state.get("draft_answer", ""),
        "final_status": ConfidenceStatus.VERIFIED,
    }


# ==============================================================================
# CONDITIONAL EDGE ROUTING FUNCTIONS
# ==============================================================================


def route_after_confidence(state: CorrectionGraphState) -> str:
    """Routes based on retrieval confidence score and retry budget."""
    settings = get_settings()
    score = state.get("retrieval_confidence", 0.0)
    attempts = state.get("attempt_count", 0)

    if score >= settings.RETRIEVAL_CONFIDENCE_THRESHOLD:
        return "contradiction_check"
    if attempts >= settings.MAX_RETRIES:
        return "low_confidence_response"
    return "reformulate_query"


def route_after_contradiction(state: CorrectionGraphState) -> str:
    """Routes based on same-date contradiction verdict."""
    if state.get("has_same_date_contradiction", False):
        return "clarify"
    return "generate_draft"


def route_after_groundedness(state: CorrectionGraphState) -> str:
    """Routes based on groundedness verification score and retry budget."""
    settings = get_settings()
    score = state.get("groundedness_score", 0.0)
    attempts = state.get("attempt_count", 0)

    if score >= settings.CORRECTION_CONFIDENCE_THRESHOLD:
        return "finalize_answer"
    if attempts >= settings.MAX_RETRIES:
        return "low_confidence_response"
    return "reformulate_query"


# ==============================================================================
# GRAPH COMPILATION & SINGLETON
# ==============================================================================


def build_correction_graph() -> Any:
    """Builds and compiles the Correction Router StateGraph."""
    graph = StateGraph(CorrectionGraphState)

    # Add all 9 explicit nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evaluate_confidence", evaluate_confidence_node)
    graph.add_node("contradiction_check", contradiction_check_node)
    graph.add_node("generate_draft", generate_draft_node)
    graph.add_node("groundedness_check", groundedness_check_node)
    graph.add_node("reformulate_query", reformulate_query_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("low_confidence_response", low_confidence_response_node)
    graph.add_node("finalize_answer", finalize_answer_node)

    # Define edges and conditional routing
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "evaluate_confidence")

    graph.add_conditional_edges(
        "evaluate_confidence",
        route_after_confidence,
        {
            "contradiction_check": "contradiction_check",
            "reformulate_query": "reformulate_query",
            "low_confidence_response": "low_confidence_response",
        },
    )

    graph.add_conditional_edges(
        "contradiction_check",
        route_after_contradiction,
        {
            "clarify": "clarify",
            "generate_draft": "generate_draft",
        },
    )

    graph.add_edge("generate_draft", "groundedness_check")

    graph.add_conditional_edges(
        "groundedness_check",
        route_after_groundedness,
        {
            "finalize_answer": "finalize_answer",
            "reformulate_query": "reformulate_query",
            "low_confidence_response": "low_confidence_response",
        },
    )

    graph.add_edge("reformulate_query", "retrieve")
    graph.add_edge("clarify", "finalize_answer")
    graph.add_edge("low_confidence_response", "finalize_answer")
    graph.add_edge("finalize_answer", END)

    return graph.compile()


_correction_router_graph: Any | None = None


def get_correction_graph() -> Any:
    """Singleton getter for the compiled Correction Router LangGraph."""
    global _correction_router_graph
    if _correction_router_graph is None:
        _correction_router_graph = build_correction_graph()
    return _correction_router_graph
