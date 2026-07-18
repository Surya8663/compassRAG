"""
Comprehensive unit and integration test suite for Phase 5: Hybrid Retrieval Service.
Verifies exact RRF mathematical accuracy, Cross-Encoder reranking accuracy,
Retrieval Confidence Evaluator thresholds, and async parallel search execution.
"""

import math
import time
from datetime import UTC, datetime
from typing import Any

import pytest
from shared.models.common import (
    ConfidenceStatus,
    DocumentChunk,
    DocumentMetadata,
    RetrievalResult,
)

from services.retrieval.app.services.evaluator import get_retrieval_evaluator
from services.retrieval.app.services.hybrid_retriever import get_hybrid_retriever
from services.retrieval.app.services.reranker import get_reranker_service
from services.retrieval.app.services.rrf import compute_rrf_fusion


def _make_dummy_payload(chunk_id: str, content: str = "dummy content") -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "document_id": f"doc_{chunk_id}",
        "chunk_index": 0,
        "content": content,
        "source": f"/docs/{chunk_id}.pdf",
        "page_number": 1,
        "tenant_id": "tenant_test",
        "version_id": "v1.0",
        "ocr_confidence": 0.98,
    }


def _make_retrieval_result(
    chunk_id: str, content: str, score: float
) -> RetrievalResult:
    meta = DocumentMetadata(
        source=f"/docs/{chunk_id}.pdf",
        page_number=1,
        ingestion_timestamp=datetime.now(UTC),
        ocr_confidence=0.99,
        tenant_id="tenant_test",
        version_id="v1.0",
    )
    chunk = DocumentChunk(
        id=chunk_id,
        document_id=f"doc_{chunk_id}",
        content=content,
        chunk_index=0,
        metadata=meta,
        score=score,
    )
    return RetrievalResult(
        chunk=chunk,
        vector_score=score,
        bm25_score=score,
        fused_score=score,
        rerank_score=score,
    )


def test_rrf_mathematical_accuracy() -> None:
    """
    Proves exact formula: score(d) = sum(1 / (k + rank_i(d))) across vector and BM25 lists.
    Given known ranking lists and k=60, verifies exact numerical scores and sorting order.
    """
    k = 60

    vector_results = [
        {
            "chunk_id": "chunk_A",
            "score": 0.95,
            "payload": _make_dummy_payload("chunk_A", "Alpha content"),
        },
        {
            "chunk_id": "chunk_B",
            "score": 0.85,
            "payload": _make_dummy_payload("chunk_B", "Beta content"),
        },
    ]

    bm25_results = [
        {
            "chunk_id": "chunk_B",
            "score": 12.5,
            "payload": _make_dummy_payload("chunk_B", "Beta content"),
        },
        {
            "chunk_id": "chunk_C",
            "score": 8.0,
            "payload": _make_dummy_payload("chunk_C", "Gamma content"),
        },
    ]

    fused = compute_rrf_fusion(
        vector_results=vector_results, bm25_results=bm25_results, k=k
    )

    assert len(fused) == 3

    results_by_id = {r.chunk.id: r for r in fused}

    # Calculate exact expected mathematical RRF scores
    # chunk_A: vector rank 1 -> 1/(60+1)
    expected_A = 1.0 / 61.0
    # chunk_B: vector rank 2 -> 1/(60+2), bm25 rank 1 -> 1/(60+1)
    expected_B = (1.0 / 62.0) + (1.0 / 61.0)
    # chunk_C: bm25 rank 2 -> 1/(60+2)
    expected_C = 1.0 / 62.0

    assert math.isclose(results_by_id["chunk_A"].fused_score, expected_A, rel_tol=1e-9)
    assert math.isclose(results_by_id["chunk_B"].fused_score, expected_B, rel_tol=1e-9)
    assert math.isclose(results_by_id["chunk_C"].fused_score, expected_C, rel_tol=1e-9)

    # Verify exact descending ordering: B > A > C
    assert fused[0].chunk.id == "chunk_B"
    assert fused[1].chunk.id == "chunk_A"
    assert fused[2].chunk.id == "chunk_C"

    # Verify original scores are accurately preserved
    assert results_by_id["chunk_B"].vector_score == 0.85
    assert results_by_id["chunk_B"].bm25_score == 12.5
    assert results_by_id["chunk_A"].bm25_score == 0.0
    assert results_by_id["chunk_C"].vector_score == 0.0


def test_cross_encoder_reranking() -> None:
    """
    Verifies Cross-Encoder re-ranking correctly re-orders candidate results based on semantic
    question-answer relevance using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
    """
    reranker = get_reranker_service()
    query = "How long does it take for Earth to orbit the Sun?"

    item_relevant = _make_retrieval_result(
        chunk_id="rel_1",
        content="The Earth revolves around the Sun in a 365.25-day orbital period.",
        score=0.1,
    )
    item_irrelevant = _make_retrieval_result(
        chunk_id="irrel_1",
        content="Banana bread recipe requires ripe bananas, flour, sugar, and baking soda.",
        score=0.9,
    )

    # Deliberately feed irrelevant item first with higher initial score
    candidates = [item_irrelevant, item_relevant]
    reranked = reranker.rerank(query=query, results=candidates, top_k=2)

    assert len(reranked) == 2
    # Relevant item must be re-ranked to position 0
    assert reranked[0].chunk.id == "rel_1"
    assert reranked[1].chunk.id == "irrel_1"
    assert reranked[0].rerank_score is not None and reranked[1].rerank_score is not None
    assert reranked[0].rerank_score > reranked[1].rerank_score


def test_retrieval_confidence_evaluator() -> None:
    """
    Verifies RetrievalConfidenceEvaluator correctly computes average top-K score
    and transitions between VERIFIED and LOW_CONFIDENCE based on threshold.
    """
    evaluator = get_retrieval_evaluator()
    threshold = 0.70

    # High confidence scenario (average = 0.80)
    high_items = [
        _make_retrieval_result("h1", "content 1", 0.85),
        _make_retrieval_result("h2", "content 2", 0.75),
    ]
    avg, is_conf, status, reason = evaluator.evaluate_confidence(
        results=high_items, threshold=threshold
    )
    assert math.isclose(avg, 0.80, rel_tol=1e-5)
    assert is_conf is True
    assert status == ConfidenceStatus.VERIFIED
    assert "meets or exceeds verified threshold" in reason

    # Low confidence scenario (average = 0.45)
    low_items = [
        _make_retrieval_result("l1", "content 1", 0.40),
        _make_retrieval_result("l2", "content 2", 0.50),
    ]
    avg_l, is_conf_l, status_l, reason_l = evaluator.evaluate_confidence(
        results=low_items, threshold=threshold
    )
    assert math.isclose(avg_l, 0.45, rel_tol=1e-5)
    assert is_conf_l is False
    assert status_l == ConfidenceStatus.LOW_CONFIDENCE
    assert "is below confidence threshold" in reason_l

    # Empty candidate pool
    avg_e, is_conf_e, status_e, reason_e = evaluator.evaluate_confidence(
        results=[], threshold=threshold
    )
    assert avg_e == 0.0
    assert is_conf_e is False
    assert status_e == ConfidenceStatus.LOW_CONFIDENCE
    assert "No candidate chunks retrieved" in reason_e


@pytest.mark.asyncio
async def test_async_parallel_hybrid_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies HybridRetrieverService.retrieve() executes vector and BM25 searches
    concurrently via asyncio.gather and correctly coordinates RRF and reranking.
    """
    retriever = get_hybrid_retriever()

    def mock_vector_sync(
        query: str, tenant_id: str, top_k: int
    ) -> list[dict[str, Any]]:
        time.sleep(0.08)  # Simulate 80ms I/O delay
        return [
            {
                "chunk_id": "parallel_vec_1",
                "score": 0.90,
                "payload": _make_dummy_payload(
                    "parallel_vec_1",
                    "Quantum computing utilizes qubits for superposition.",
                ),
            }
        ]

    def mock_bm25_sync(
        query: str, tenant_id: str, top_k: int
    ) -> list[dict[str, Any]]:
        time.sleep(0.08)  # Simulate 80ms I/O delay
        return [
            {
                "chunk_id": "parallel_bm25_1",
                "score": 15.0,
                "payload": _make_dummy_payload(
                    "parallel_bm25_1",
                    "Superposition allows qubits to exist in multiple states.",
                ),
            }
        ]

    monkeypatch.setattr(retriever, "_vector_search_sync", mock_vector_sync)
    monkeypatch.setattr(retriever, "_bm25_search_sync", mock_bm25_sync)

    start_t = time.perf_counter()
    results, avg_score, status, reasoning = await retriever.retrieve(
        query="What is qubit superposition in quantum computing?",
        tenant_id="tenant_parallel",
        top_k=2,
    )
    elapsed = time.perf_counter() - start_t

    # Since both 80ms tasks run in parallel via asyncio.gather, elapsed time must be
    # close to ~80ms (way under 150ms sequential time)
    assert elapsed < 0.15, f"Expected parallel execution < 150ms, took {elapsed:.3f}s"
    assert len(results) == 2
    assert isinstance(avg_score, float)
    assert status in (ConfidenceStatus.VERIFIED, ConfidenceStatus.LOW_CONFIDENCE)
