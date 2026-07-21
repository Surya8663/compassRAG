"""
Hybrid Retrieval Service.
Orchestrates parallel Qdrant vector and Elasticsearch BM25 searches using asyncio.gather,
fuses candidate lists via Reciprocal Rank Fusion (RRF), re-ranks via CrossEncoder/Cohere,
and evaluates retrieval confidence.

OpenTelemetry spans:
  compass.retrieval.hybrid_retrieve   — parent span wrapping the entire retrieve() call
  compass.retrieval.vector_search     — dense vector search (Qdrant)
  compass.retrieval.bm25_search       — sparse keyword search (Elasticsearch)
  compass.retrieval.rrf_fusion        — Reciprocal Rank Fusion computation
  compass.retrieval.rerank            — CrossEncoder / Cohere reranking
"""

import asyncio
import logging
from functools import lru_cache
from typing import Any

from shared.config import get_settings
from shared.metrics import PIPELINE_STAGE_DURATION
from shared.models.common import ConfidenceStatus, RetrievalResult
from shared.telemetry import get_tracer, traced_span

from services.retrieval.app.services.embedder import get_embedding_service
from services.retrieval.app.services.es_store import get_es_store
from services.retrieval.app.services.evaluator import get_retrieval_evaluator
from services.retrieval.app.services.qdrant_store import get_qdrant_store
from services.retrieval.app.services.reranker import get_reranker_service
from services.retrieval.app.services.rrf import compute_rrf_fusion

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)


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
        Wrapped in a ``compass.retrieval.vector_search`` OTel span.
        """
        with traced_span(
            _tracer,
            "compass.retrieval.vector_search",
            {"query.length": len(query), "tenant_id": tenant_id, "top_k": top_k},
        ) as span:
            try:
                import time
                t0 = time.perf_counter()
                query_vector = self.embedder.embed_text(query)
                results = self.qdrant_store.search(
                    query_vector=query_vector, tenant_id=tenant_id, top_k=top_k
                )
                span.set_attribute("result_count", len(results))
                PIPELINE_STAGE_DURATION.labels(
                    service="compass-rag-retrieval", stage="vector_search"
                ).observe(time.perf_counter() - t0)
                return results
            except Exception as exc:
                logger.error("Vector search failed during hybrid retrieval: %s", exc)
                return []

    def _bm25_search_sync(
        self, query: str, tenant_id: str, top_k: int
    ) -> list[dict[str, Any]]:
        """
        Synchronous worker for keyword search: queries Elasticsearch BM25 index.
        Wrapped in a ``compass.retrieval.bm25_search`` OTel span.
        """
        with traced_span(
            _tracer,
            "compass.retrieval.bm25_search",
            {"query.length": len(query), "tenant_id": tenant_id, "top_k": top_k},
        ) as span:
            try:
                import time
                t0 = time.perf_counter()
                results = self.es_store.search_keywords(
                    query_text=query, tenant_id=tenant_id, top_k=top_k
                )
                span.set_attribute("result_count", len(results))
                PIPELINE_STAGE_DURATION.labels(
                    service="compass-rag-retrieval", stage="bm25_search"
                ).observe(time.perf_counter() - t0)
                return results
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
        1. Dispatches vector search and BM25 search in parallel threads via ``asyncio.gather``.
        2. Merges both ranking lists using exact Reciprocal Rank Fusion (``RRF_K_CONSTANT``).
        3. Re-ranks candidate chunks using CrossEncoder / Cohere model.
        4. Evaluates average top-K score against ``RETRIEVAL_CONFIDENCE_THRESHOLD``.

        Produces a parent OTel span ``compass.retrieval.hybrid_retrieve`` containing all
        sub-stage spans as children for end-to-end waterfall visibility in Jaeger.

        Returns:
            Tuple of ``(reranked_results, average_score, confidence_status, reasoning)``.
        """
        if not tenant_id:
            raise ValueError("tenant_id is required for hybrid retrieval.")

        candidate_count = max(top_k * candidate_multiplier, 30)

        with traced_span(
            _tracer,
            "compass.retrieval.hybrid_retrieve",
            {
                "query.length": len(query),
                "tenant_id": tenant_id,
                "top_k": top_k,
                "candidate_pool": candidate_count,
            },
        ) as parent_span:
            import time
            t_start = time.perf_counter()

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

            # RRF fusion span
            with traced_span(
                _tracer,
                "compass.retrieval.rrf_fusion",
                {
                    "vector_count": len(vector_results),
                    "bm25_count": len(bm25_results),
                    "rrf_k": self.settings.RRF_K_CONSTANT,
                },
            ) as rrf_span:
                t0 = time.perf_counter()
                fused_results = compute_rrf_fusion(
                    vector_results=vector_results,
                    bm25_results=bm25_results,
                    k=self.settings.RRF_K_CONSTANT,
                )
                rrf_span.set_attribute("fused_count", len(fused_results))
                PIPELINE_STAGE_DURATION.labels(
                    service="compass-rag-retrieval", stage="rrf_fusion"
                ).observe(time.perf_counter() - t0)

            logger.debug(
                "RRF fused %d unique candidate chunks. Executing cross-encoder reranking...",
                len(fused_results),
            )

            # Reranking span
            with traced_span(
                _tracer,
                "compass.retrieval.rerank",
                {"candidate_count": len(fused_results), "top_k": top_k},
            ) as rerank_span:
                t0 = time.perf_counter()
                reranked_results = self.reranker.rerank(
                    query=query,
                    results=fused_results,
                    top_k=top_k,
                )
                rerank_span.set_attribute("reranked_count", len(reranked_results))
                PIPELINE_STAGE_DURATION.labels(
                    service="compass-rag-retrieval", stage="rerank"
                ).observe(time.perf_counter() - t0)

            avg_score, is_confident, status, reasoning = self.evaluator.evaluate_confidence(
                results=reranked_results,
            )

            parent_span.set_attribute("result_count", len(reranked_results))
            parent_span.set_attribute("avg_score", avg_score)
            parent_span.set_attribute("confidence_status", str(status))
            PIPELINE_STAGE_DURATION.labels(
                service="compass-rag-retrieval", stage="hybrid_retrieve"
            ).observe(time.perf_counter() - t_start)

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
