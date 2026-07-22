"""
Cross-Encoder Re-ranking Service supporting local models and Cohere API.
Scores candidate document chunks against the query and re-sorts top-K items.
"""

import logging
from functools import lru_cache
from typing import Any

from shared.config import get_settings
from shared.models.common import RetrievalResult

logger = logging.getLogger(__name__)


import math


def _normalize_score(val: float) -> float:
    """Safely converts raw reranker score/logit into a calibrated [0.0, 1.0] confidence float."""
    if val is None or math.isnan(val) or math.isinf(val):
        return 0.0
    # If already in [0, 1] range (e.g. Cohere relevance score)
    if 0.0 <= val <= 1.0:
        return float(val)
    # Apply sigmoid for raw cross-encoder logits
    try:
        sig = 1.0 / (1.0 + math.exp(-val))
        return max(0.0, min(1.0, float(sig)))
    except Exception:
        return 0.0


class RerankerService:
    """
    Reranks candidate RetrievalResult items using Cross-Encoder models.
    Supports offline local CrossEncoder models and online Cohere Rerank v3 API.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = self.settings.RERANKER_PROVIDER.lower()
        if self.provider == "local":
            self.model_name = self.settings.LOCAL_RERANK_MODEL
            logger.info("Initializing local CrossEncoder rerank model: %s", self.model_name)
            from sentence_transformers import CrossEncoder

            self._local_model: Any = CrossEncoder(self.model_name)
            self._cohere_client: Any = None
        elif self.provider == "cohere":
            self.model_name = self.settings.COHERE_RERANK_MODEL
            logger.info("Initializing Cohere Rerank client for model: %s", self.model_name)
            import cohere

            self._cohere_client = cohere.Client(api_key=self.settings.COHERE_API_KEY)
            self._local_model = None
        else:
            raise ValueError(
                f"Unsupported RERANKER_PROVIDER: '{self.provider}'. Must be 'local' or 'cohere'."
            )

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 10
    ) -> list[RetrievalResult]:
        """
        Reranks a list of candidate RetrievalResult objects given the query string.
        Assigns `rerank_score` to each candidate and returns the top_k sorted descending.
        """
        if not results:
            return []

        if not query or not query.strip():
            # If query is empty, preserve existing RRF ordering
            for r in results:
                r.rerank_score = _normalize_score(r.fused_score)
            return results[:top_k]

        if self.provider == "local":
            pairs = [(query, r.chunk.content) for r in results]
            try:
                scores = self._local_model.predict(pairs)
                # Handle scalar vs array return types from sentence_transformers
                if hasattr(scores, "__iter__") and not isinstance(scores, (str, bytes)):
                    score_list = [_normalize_score(float(s)) for s in scores]
                else:
                    score_list = [_normalize_score(float(scores))]

                for idx, r in enumerate(results):
                    r.rerank_score = score_list[idx]
            except Exception as exc:
                logger.error("Local CrossEncoder reranking failed: %s", exc)
                for r in results:
                    r.rerank_score = _normalize_score(r.fused_score)

            results.sort(
                key=lambda r: r.rerank_score if r.rerank_score is not None else 0.0,
                reverse=True,
            )
            return results[:top_k]

        elif self.provider == "cohere":
            docs = [r.chunk.content for r in results]
            try:
                resp = self._cohere_client.rerank(
                    query=query,
                    documents=docs,
                    top_n=min(top_k, len(results)),
                    model=self.model_name,
                )
                reranked_items: list[RetrievalResult] = []
                for hit in resp.results:
                    item = results[hit.index]
                    item.rerank_score = _normalize_score(float(hit.relevance_score))
                    reranked_items.append(item)
                return reranked_items
            except Exception as exc:
                logger.error("Cohere rerank API call failed: %s", exc)
                for r in results:
                    r.rerank_score = _normalize_score(r.fused_score)
                results.sort(
                    key=lambda r: r.rerank_score if r.rerank_score is not None else 0.0,
                    reverse=True,
                )
                return results[:top_k]

        return results[:top_k]


@lru_cache
def get_reranker_service() -> RerankerService:
    """
    Returns cached singleton instance of RerankerService.
    """
    return RerankerService()
