"""
Reciprocal Rank Fusion (RRF) Service for combining dense and sparse rankings.
Implements formula: score(d) = sum(1 / (k + rank_i(d))) over all ranking lists.
"""

from datetime import UTC, datetime
from typing import Any

from shared.models.common import DocumentChunk, DocumentMetadata, RetrievalResult


def compute_rrf_fusion(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    k: int = 60,
) -> list[RetrievalResult]:
    """
    Computes exact Reciprocal Rank Fusion across Qdrant and Elasticsearch results.

    Formula:
        score(d) = sum_{r in rankings} (1 / (k + rank_r(d)))
    where rank_r(d) is 1-indexed rank of document d in ranking list r.

    Args:
        vector_results: Ordered dicts from Qdrant (`[{"chunk_id": ..., "score": ...}]`).
        bm25_results: Ordered dicts from Elasticsearch (`[{"chunk_id": ..., "score": ...}]`).
        k: Standard RRF smoothing constant (default 60).

    Returns:
        List of RetrievalResult instances sorted descending by `fused_score`.
    """
    ranks_vector: dict[str, int] = {}
    scores_vector: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}

    for rank_idx, item in enumerate(vector_results, start=1):
        cid = str(item.get("chunk_id", ""))
        if not cid:
            continue
        ranks_vector[cid] = rank_idx
        scores_vector[cid] = float(item.get("score") or 0.0)
        payloads[cid] = item.get("payload") or {}

    ranks_bm25: dict[str, int] = {}
    scores_bm25: dict[str, float] = {}

    for rank_idx, item in enumerate(bm25_results, start=1):
        cid = str(item.get("chunk_id", ""))
        if not cid:
            continue
        ranks_bm25[cid] = rank_idx
        scores_bm25[cid] = float(item.get("score") or 0.0)
        if cid not in payloads:
            payloads[cid] = item.get("payload") or {}

    all_chunk_ids = set(ranks_vector.keys()) | set(ranks_bm25.keys())
    retrieval_results: list[RetrievalResult] = []

    for cid in all_chunk_ids:
        r_vec = ranks_vector.get(cid)
        r_bm25 = ranks_bm25.get(cid)

        fused_score = 0.0
        if r_vec is not None:
            fused_score += 1.0 / (k + r_vec)
        if r_bm25 is not None:
            fused_score += 1.0 / (k + r_bm25)

        vec_score = scores_vector.get(cid, 0.0)
        bm25_score = scores_bm25.get(cid, 0.0)
        payload = payloads.get(cid, {})

        # Extract strictly typed metadata fields
        page_num = payload.get("page_number")
        page_number_clean = int(page_num) if page_num is not None else None

        ocr_conf = payload.get("ocr_confidence")
        ocr_confidence_clean = float(ocr_conf) if ocr_conf is not None else None

        metadata = DocumentMetadata(
            source=str(payload.get("source", "unknown")),
            page_number=page_number_clean,
            ingestion_timestamp=datetime.now(UTC),
            ocr_confidence=ocr_confidence_clean,
            tenant_id=str(payload.get("tenant_id", "unknown")),
            version_id=str(payload.get("version_id", "v1.0")),
        )

        chunk = DocumentChunk(
            id=cid,
            document_id=str(payload.get("document_id", "unknown")),
            content=str(payload.get("content", "")),
            chunk_index=int(payload.get("chunk_index", 0)),
            metadata=metadata,
            score=fused_score,
        )

        retrieval_results.append(
            RetrievalResult(
                chunk=chunk,
                vector_score=vec_score,
                bm25_score=bm25_score,
                fused_score=fused_score,
                rerank_score=None,
            )
        )

    # Sort descending by fused RRF score
    retrieval_results.sort(key=lambda r: r.fused_score, reverse=True)
    return retrieval_results
