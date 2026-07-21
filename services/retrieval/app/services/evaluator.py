"""
Retrieval Confidence Evaluator Service.
Computes average score of top candidate chunks and evaluates status against threshold.
"""

import logging
from functools import lru_cache

from shared.config import get_settings
from shared.models.common import ConfidenceStatus, RetrievalResult

logger = logging.getLogger(__name__)


class RetrievalConfidenceEvaluator:
    """
    Evaluates confidence of retrieved candidate chunks after RRF fusion and reranking.
    Compares average relevance score across top candidate chunks against configurable thresholds.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def evaluate_confidence(
        self,
        results: list[RetrievalResult],
        threshold: float | None = None,
    ) -> tuple[float, bool, ConfidenceStatus, str]:
        """
        Computes average re-ranked score of top-K candidate results and evaluates status.

        Args:
            results: List of top candidate RetrievalResult objects.
            threshold: Optional override threshold (defaults to configured settings).

        Returns:
            Tuple of `(average_score, is_confident, confidence_status, reasoning)`.
        """
        eval_threshold = (
            threshold
            if threshold is not None
            else self.settings.RETRIEVAL_CONFIDENCE_THRESHOLD
        )

        if not results:
            msg = "No candidate chunks retrieved matching search criteria."
            logger.warning(msg)
            return (0.0, False, ConfidenceStatus.LOW_CONFIDENCE, msg)

        scores = []
        for r in results:
            if hasattr(r, "rerank_score") and r.rerank_score is not None:
                scores.append(float(r.rerank_score))
            elif hasattr(r, "fused_score"):
                scores.append(float(r.fused_score))
            elif isinstance(r, dict):
                scores.append(float(r.get("rerank_score") or r.get("fused_score") or 1.0))
            else:
                scores.append(1.0)
        avg_score = float(sum(scores) / len(scores))

        if avg_score >= eval_threshold:
            is_confident = True
            status = ConfidenceStatus.VERIFIED
            reasoning = (
                f"Retrieval average top-{len(results)} score ({avg_score:.4f}) "
                f"meets or exceeds verified threshold ({eval_threshold:.4f})."
            )
            logger.debug(reasoning)
        else:
            is_confident = False
            status = ConfidenceStatus.LOW_CONFIDENCE
            reasoning = (
                f"Retrieval average top-{len(results)} score ({avg_score:.4f}) "
                f"is below confidence threshold ({eval_threshold:.4f})."
            )
            logger.info(reasoning)

        return (avg_score, is_confident, status, reasoning)


@lru_cache
def get_retrieval_evaluator() -> RetrievalConfidenceEvaluator:
    """
    Returns cached singleton instance of RetrievalConfidenceEvaluator.
    """
    return RetrievalConfidenceEvaluator()
