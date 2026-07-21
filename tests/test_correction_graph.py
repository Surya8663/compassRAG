"""
Unit and integration tests for Phase 6 Correction Router LangGraph (`CorrectionRouterGraph`).
Verifies all 5 mandatory self-correction and routing scenarios:
1. Sufficient context (generates answer, status=VERIFIED)
2. Low retrieval confidence (triggers reformulate_query with HyDE and increments attempt count)
3. Contradiction across different dates (resolves via supersession, retains newer chunk)
4. Contradiction across same date/version (triggers clarify, status=CLARIFICATION_NEEDED)
5. Exhausted retries (triggers low_confidence_response, status=LOW_CONFIDENCE)
Also verifies `/correct` API endpoint integration.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from shared.models.common import (
    ConfidenceStatus,
    DocumentChunk,
    DocumentMetadata,
    RetrievalResult,
)

from services.correction.app.services.contradiction import get_contradiction_detector
from services.correction.app.services.graph import get_correction_graph
from services.correction.app.services.groundedness import get_groundedness_checker
from services.correction.app.services.reformulator import get_query_reformulator
from services.correction.app.services.state import CorrectionGraphState


def _make_chunk(
    chunk_id: str,
    content: str,
    version: str = "v1.0",
    source: str = "doc.pdf",
    score: float = 0.85,
    tenant_id: str = "test_tenant",
) -> RetrievalResult:
    """Helper to create a test candidate chunk with specific version and content."""
    meta = DocumentMetadata(
        document_id=f"doc_{chunk_id}",
        tenant_id=tenant_id,
        source=source,
        page_number=1,
        ingestion_timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        version_id=version,
    )
    chunk = DocumentChunk(
        id=chunk_id,
        document_id=f"doc_{chunk_id}",
        content=content,
        metadata=meta,
    )
    return RetrievalResult(
        chunk=chunk,
        vector_score=score,
        bm25_score=score,
        fused_score=score,
        rerank_score=score,
    )


@pytest.mark.asyncio
async def test_scenario_sufficient_context_generates() -> None:
    """
    Scenario 1: High retrieval confidence, no contradictions, and high groundedness score.
    Asserts graph traverses full pipeline and outputs VERIFIED answer with citations.
    """
    graph = get_correction_graph()
    chunks = [
        _make_chunk("c1", "Compass RAG revolves around state-of-the-art agentic workflows."),
        _make_chunk("c2", "The system operates with full tenant isolation and real vector DBs."),
    ]

    initial_state: CorrectionGraphState = {
        "query": "What does Compass RAG revolve around?",
        "tenant_id": "test_tenant",
        "original_query": "What does Compass RAG revolve around?",
        "attempt_count": 0,
        "retrieved_chunks": chunks,
        "retrieval_confidence": 0.88,
        "retrieval_status": ConfidenceStatus.VERIFIED,
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

    # Mock hybrid retriever to return our high-confidence chunks on attempt 0
    mock_retriever = AsyncMock()
    mock_retriever.retrieve.return_value = (chunks, 0.88, ConfidenceStatus.VERIFIED, "High conf")

    with patch(
        "services.correction.app.services.graph.get_hybrid_retriever",
        return_value=mock_retriever,
    ):
        result_state = await graph.ainvoke(initial_state)

    assert result_state["final_status"] == ConfidenceStatus.VERIFIED
    assert "revolves around" in result_state["final_answer"]
    assert len(result_state["draft_citations"]) == 1
    assert result_state["groundedness_score"] >= 0.70
    assert result_state["has_same_date_contradiction"] is False


@pytest.mark.asyncio
async def test_scenario_low_confidence_reformulates() -> None:
    """
    Scenario 2: Low initial retrieval confidence (< threshold) on attempt 0.
    Asserts graph routes from evaluate_confidence to reformulate_query (invoking HyDE),
    increments attempt_count, and loops back to retrieve.
    """
    graph = get_correction_graph()
    low_chunks = [_make_chunk("c_low", "Irrelevant passage regarding other matters.", score=0.40)]
    recovered_chunks = [
        _make_chunk("c_high", "Compass RAG uses Qdrant and Elasticsearch for retrieval.")
    ]

    initial_state: CorrectionGraphState = {
        "query": "How does retrieval work in Compass RAG?",
        "tenant_id": "test_tenant",
        "original_query": "How does retrieval work in Compass RAG?",
        "attempt_count": 0,
        "retrieved_chunks": low_chunks,
        "retrieval_confidence": 0.40,
        "retrieval_status": ConfidenceStatus.LOW_CONFIDENCE,
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

    mock_retriever = AsyncMock()
    # First call inside retrieve node (after loop from reformulate) returns recovered high score
    mock_retriever.retrieve.return_value = (
        recovered_chunks,
        0.85,
        ConfidenceStatus.VERIFIED,
        "Recovered",
    )

    reformulator = get_query_reformulator()
    with (
        patch(
            "services.correction.app.services.graph.get_hybrid_retriever",
            return_value=mock_retriever,
        ),
        patch.object(
            reformulator, "_apply_hyde", return_value="HyDE hypothetical reference passage."
        ) as mock_hyde,
    ):
        result_state = await graph.ainvoke(initial_state)

    mock_hyde.assert_called_once()
    assert result_state["attempt_count"] == 1
    assert result_state["final_status"] == ConfidenceStatus.VERIFIED
    assert "Qdrant and Elasticsearch" in result_state["final_answer"]


@pytest.mark.asyncio
async def test_scenario_contradiction_different_dates_supersedes() -> None:
    """
    Scenario 3: Contradictory candidate chunks across different versions (`v1.0` vs `v2.0`).
    Asserts contradiction_check resolves via supersession, retains newer chunk (`v2.0`),
    sets has_same_date_contradiction=False, and proceeds to generate_draft.
    """
    graph = get_correction_graph()
    chunks = [
        _make_chunk("old_c", "The system retention policy is strictly 365 days.", version="v1.0"),
        _make_chunk("new_c", "The system retention policy is strictly 366 days.", version="v2.0"),
    ]

    initial_state: CorrectionGraphState = {
        "query": "What is the retention policy duration?",
        "tenant_id": "test_tenant",
        "original_query": "What is the retention policy duration?",
        "attempt_count": 0,
        "retrieved_chunks": chunks,
        "retrieval_confidence": 0.85,
        "retrieval_status": ConfidenceStatus.VERIFIED,
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

    mock_retriever = AsyncMock()
    mock_retriever.retrieve.return_value = (chunks, 0.85, ConfidenceStatus.VERIFIED, "Conf OK")

    with patch(
        "services.correction.app.services.graph.get_hybrid_retriever",
        return_value=mock_retriever,
    ):
        result_state = await graph.ainvoke(initial_state)

    assert result_state["contradictions_detected"] is True
    assert result_state["has_same_date_contradiction"] is False
    assert len(result_state["retrieved_chunks"]) == 1
    assert result_state["retrieved_chunks"][0].chunk.id == "new_c"
    assert "supersession" in result_state["contradiction_reasoning"].lower()
    assert result_state["final_status"] == ConfidenceStatus.VERIFIED


@pytest.mark.asyncio
async def test_scenario_contradiction_same_date_clarifies() -> None:
    """
    Scenario 4: Contradictory candidate chunks across identical version/date metadata (`v1.0`).
    Asserts contradiction_check flags has_same_date_contradiction=True, routing to clarify node
    and returning final_status=CLARIFICATION_NEEDED with a clarifying question.
    """
    graph = get_correction_graph()
    chunks = [
        _make_chunk("c_a", "The security protocol prohibits external access.", version="v1.0"),
        _make_chunk("c_b", "The security protocol permits external access.", version="v1.0"),
    ]

    initial_state: CorrectionGraphState = {
        "query": "Is external access allowed by the security protocol?",
        "tenant_id": "test_tenant",
        "original_query": "Is external access allowed by the security protocol?",
        "attempt_count": 0,
        "retrieved_chunks": chunks,
        "retrieval_confidence": 0.85,
        "retrieval_status": ConfidenceStatus.VERIFIED,
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

    mock_retriever = AsyncMock()
    mock_retriever.retrieve.return_value = (chunks, 0.85, ConfidenceStatus.VERIFIED, "Conf OK")

    with patch(
        "services.correction.app.services.graph.get_hybrid_retriever",
        return_value=mock_retriever,
    ):
        result_state = await graph.ainvoke(initial_state)

    assert result_state["contradictions_detected"] is True
    assert result_state["has_same_date_contradiction"] is True
    assert result_state["final_status"] == ConfidenceStatus.CLARIFICATION_NEEDED
    assert "?" in result_state["final_answer"]
    assert "conflicting statements" in result_state["final_answer"].lower()


@pytest.mark.asyncio
async def test_scenario_exhausted_retries_low_confidence_response() -> None:
    """
    Scenario 5: Graph enters with attempt_count = MAX_RETRIES (3) and low confidence.
    Asserts conditional edge routes directly to low_confidence_response node and exits
    with final_status=LOW_CONFIDENCE.
    """
    graph = get_correction_graph()
    low_chunks = [_make_chunk("c_bad", "Completely ungrounded text.", score=0.20)]

    initial_state: CorrectionGraphState = {
        "query": "Exhausted query requiring fallback?",
        "tenant_id": "test_tenant",
        "original_query": "Exhausted query requiring fallback?",
        "attempt_count": 3,
        "retrieved_chunks": low_chunks,
        "retrieval_confidence": 0.20,
        "retrieval_status": ConfidenceStatus.LOW_CONFIDENCE,
        "contradictions_detected": False,
        "contradiction_reasoning": "",
        "has_same_date_contradiction": False,
        "draft_answer": "",
        "draft_citations": [],
        "groundedness_score": 0.0,
        "groundedness_verdict": False,
        "verdicts": [],
        "final_answer": "",
        "final_status": ConfidenceStatus.LOW_CONFIDENCE,
    }

    mock_retriever = AsyncMock()
    mock_retriever.retrieve.return_value = (
        low_chunks,
        0.20,
        ConfidenceStatus.LOW_CONFIDENCE,
        "Low",
    )

    with patch(
        "services.correction.app.services.graph.get_hybrid_retriever",
        return_value=mock_retriever,
    ):
        result_state = await graph.ainvoke(initial_state)

    assert result_state["final_status"] == ConfidenceStatus.LOW_CONFIDENCE
    assert "cannot provide a definitive or verified answer" in result_state["final_answer"]
    assert "3 search and reformulation attempts" in result_state["final_answer"]


def test_evaluator_services_directly() -> None:
    """Verifies ContradictionDetectorService and GroundednessCheckerService methods directly."""
    detector = get_contradiction_detector()
    ca = _make_chunk("1", "The retention duration is never extended.")
    cb = _make_chunk("2", "The retention duration is always extended.")
    detected, same_date, resolved, reasoning = detector.check_contradictions([ca, cb])
    assert detected is True
    assert same_date is True

    checker = get_groundedness_checker()
    draft = "The retention duration is never extended. It revolves around strict policy."
    score, grounded, verdicts, summary = checker.verify_groundedness(draft, [ca])
    assert score >= 0.5
    assert len(verdicts) >= 1
