"""
Hybrid Retrieval Service.
Orchestrates parallel Qdrant vector and Elasticsearch BM25 searches using asyncio.gather,
fuses candidate lists via Reciprocal Rank Fusion (RRF), re-ranks via CrossEncoder/Cohere,
and evaluates retrieval confidence.
"""

import asyncio
import logging
from functools import lru_cache
from typing import Any

from shared.config import get_settings
from shared.models.common import ConfidenceStatus, RetrievalResult

from services.retrieval.app.services.embedder import get_embedding_service
from services.retrieval.app.services.es_store import get_es_store
from services.retrieval.app.services.evaluator import get_retrieval_evaluator
from services.retrieval.app.services.qdrant_store import get_qdrant_store
from services.retrieval.app.services.reranker import get_reranker_service
from services.retrieval.app.services.rrf import compute_rrf_fusion

logger = logging.getLogger(__name__)


class HybridRetrieverService:
    """
    Production Hybrid Retriever combining dense vector search and sparse BM25 search concurrently.
    Enforces tenant isolation, exact RRF ranking fusion, cross-encoder reranking, and scoring.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedder = get_embedding_service()
        self.qdrant_store = get_qdrant_store()
        self.es_store = get_es_store()
        self.reranker = get_reranker_service()
        self.evaluator = get_retrieval_evaluator()

    def _vector_search_sync(
        self, query: str, tenant_id: str, top_k: int
    ) -> list[dict[str, Any]]:
        """
        Synchronous worker for vector search: embeds query text then queries Qdrant.
        """
        try:
            query_vector = self.embedder.embed_text(query)
            return self.qdrant_store.search(
                query_vector=query_vector, tenant_id=tenant_id, top_k=top_k
            )
        except Exception as exc:
            logger.error("Vector search failed during hybrid retrieval: %s", exc)
            return []

    def _bm25_search_sync(
        self, query: str, tenant_id: str, top_k: int
    ) -> list[dict[str, Any]]:
        """
        Synchronous worker for keyword search: queries Elasticsearch BM25 index.
        """
        try:
            return self.es_store.search_keywords(
                query_text=query, tenant_id=tenant_id, top_k=top_k
            )
        except Exception as exc:
            logger.error("BM25 search failed during hybrid retrieval: %s", exc)
            return []

    async def retrieve(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 10,
        candidate_multiplier: int = 3,
    ) -> tuple[list[RetrievalResult], float, ConfidenceStatus, str]:
        """
        Performs full hybrid retrieval pipeline asynchronously:
        1. Dispatches vector search and BM25 search in parallel threads via `asyncio.gather`.
        2. Merges both ranking lists using exact Reciprocal Rank Fusion (`RRF_K_CONSTANT`).
        3. Re-ranks candidate chunks using CrossEncoder / Cohere model.
        4. Evaluates average top-K score against `RETRIEVAL_CONFIDENCE_THRESHOLD`.

        Args:
            query: User query or self-correction query string.
            tenant_id: Unique tenant identifier enforcing data isolation.
            top_k: Final number of re-ranked candidate chunks to return.
            candidate_multiplier: Multiplier determining initial candidate pool size.

        Returns:
            Tuple of `(reranked_results, average_score, confidence_status, reasoning)`.
        """
        if not tenant_id:
            raise ValueError("tenant_id is required for hybrid retrieval.")

        candidate_count = max(top_k * candidate_multiplier, 30)

        logger.debug(
            "Executing parallel vector+BM25 search for tenant '%s' (candidate pool=%d)...",
            tenant_id,
            candidate_count,
        )

        vector_task = asyncio.to_thread(
            self._vector_search_sync, query, tenant_id, candidate_count
        )
        bm25_task = asyncio.to_thread(
            self._bm25_search_sync, query, tenant_id, candidate_count
        )

        # Execute parallel search concurrently via asyncio.gather
        vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)

        logger.debug(
            "Retrieved %d vector candidates and %d BM25 candidates. Running RRF fusion...",
            len(vector_results),
            len(bm25_results),
        )

        fused_results = compute_rrf_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            k=self.settings.RRF_K_CONSTANT,
        )

        logger.debug(
            "RRF fused %d unique candidate chunks. Executing cross-encoder reranking...",
            len(fused_results),
        )

        reranked_results = self.reranker.rerank(
            query=query,
            results=fused_results,
            top_k=top_k,
        )

        avg_score, is_confident, status, reasoning = self.evaluator.evaluate_confidence(
            results=reranked_results,
        )

        logger.info(
            "Hybrid retrieval done for tenant '%s': %d chunks (status=%s, avg=%.4f)",
            tenant_id,
            len(reranked_results),
            status,
            avg_score,
        )

        return (reranked_results, avg_score, status, reasoning)


@lru_cache
def get_hybrid_retriever() -> HybridRetrieverService:
    """
    Returns cached singleton instance of HybridRetrieverService.
    """
    return HybridRetrieverService()
