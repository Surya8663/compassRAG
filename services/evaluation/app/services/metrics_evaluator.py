"""
Metrics Evaluator for the Evaluation Service (`services/evaluation`).
Programmatically calculates real numbers for:
- Hallucination rate (`1.0 - groundedness_score`)
- Retrieval recall (fraction of expected chunks/keywords retrieved)
- Citation correctness (entailment verification between claim and cited chunk text)
- Appropriate abstention rate (binary correctness on whether pipeline abstained when required)
"""

import logging
from typing import Any

from shared.models.common import Citation, DocumentChunk, RetrievalResult

from services.evaluation.app.models.eval import GoldenQuestion

logger = logging.getLogger(__name__)


class MetricsEvaluator:
    """
    Evaluates raw pipeline run outputs against ground-truth authoritative targets
    from `GoldenQuestion`.
    """

    def __init__(self) -> None:
        self._groundedness_checker: Any | None = None

    def _get_groundedness_checker(self) -> Any:
        if self._groundedness_checker is None:
            try:
                from services.correction.app.services.groundedness import (
                    get_groundedness_checker,
                )
                self._groundedness_checker = get_groundedness_checker()
            except Exception as exc:
                logger.warning("Could not load GroundednessChecker in MetricsEvaluator: %s", exc)
        return self._groundedness_checker

    def _to_retrieval_results(self, chunks: list[Any]) -> list[RetrievalResult]:
        """Normalize any chunk list to list[RetrievalResult] for groundedness checking."""
        results: list[RetrievalResult] = []
        for item in chunks:
            if isinstance(item, RetrievalResult):
                results.append(item)
            elif isinstance(item, DocumentChunk):
                results.append(
                    RetrievalResult(
                        chunk=item, vector_score=1.0, bm25_score=1.0, fused_score=1.0
                    )
                )
            elif isinstance(item, dict):
                try:
                    c = DocumentChunk.model_validate(item)
                    results.append(
                        RetrievalResult(
                            chunk=c, vector_score=1.0, bm25_score=1.0, fused_score=1.0
                        )
                    )
                except Exception:
                    pass
        return results

    def compute_retrieval_recall(
        self, question: GoldenQuestion, retrieved_chunks: list[Any]
    ) -> float:
        """
        Computes fraction of expected chunks or keywords successfully retrieved in top-K.
        """
        if not question.expected_chunk_ids and not question.expected_keywords:
            return 1.0

        normalized = self._to_retrieval_results(retrieved_chunks)
        if not normalized:
            return 0.0

        retrieved_ids = {
            c.chunk.id for c in normalized
        } | {
            c.chunk.document_id for c in normalized
        } | {
            c.chunk.metadata.source for c in normalized
        }
        combined_text = " ".join([c.chunk.content for c in normalized]).lower()

        scores: list[float] = []

        if question.expected_chunk_ids:
            found_ids = sum(
                1 for eid in question.expected_chunk_ids
                if any(eid in rid for rid in retrieved_ids)
            )
            scores.append(float(found_ids) / float(len(question.expected_chunk_ids)))

        if question.expected_keywords:
            found_kw = sum(
                1 for kw in question.expected_keywords if kw.lower() in combined_text
            )
            scores.append(float(found_kw) / float(len(question.expected_keywords)))

        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def compute_hallucination_rate(
        self, question: GoldenQuestion, answer: str, retrieved_chunks: list[Any]
    ) -> float:
        """
        Computes hallucination rate (`1.0 - groundedness_score`).
        - Safe abstentions / clarification requests receive 0.0 hallucination.
        - Confidently producing unsupported/irrelevant content for unanswerable questions (e.g. Q7, Q8) counts as hallucination (1.0).
        """
        ans_lower = answer.lower()
        abstention_phrases = [
            "low_confidence",
            "clarification_needed",
            "cannot provide a definitive or verified answer",
            "evidence is insufficient",
            "available documentation does not support",
            "unable to verify",
            "insufficient information",
            "do not contain",
            "no information",
            "clarification is required",
            "no verified context",
            "no relevant context available",
            "does not contain information",
        ]
        is_abstained = any(p in ans_lower for p in abstention_phrases)
        if is_abstained:
            return 0.0

        if question.should_abstain:
            # If question requires abstention but answer produced factual content instead of abstaining, it's hallucinated!
            return 1.0

        checker = self._get_groundedness_checker()
        if checker is None:
            if not answer.strip():
                return 0.0
            kw_hits = sum(
                1 for kw in question.expected_keywords if kw.lower() in answer.lower()
            )
            total_kw = max(1, len(question.expected_keywords))
            return max(0.0, min(1.0, 1.0 - (float(kw_hits) / float(total_kw))))

        normalized = self._to_retrieval_results(retrieved_chunks)
        score, _, _, _ = checker.verify_groundedness(answer, normalized)
        return max(0.0, min(1.0, 1.0 - score))

    def compute_citation_correctness(
        self, citations: list[Any], retrieved_chunks: list[Any]
    ) -> float:
        """
        Verifies citation correctness:
        1. Exact or fuzzy snippet inclusion check in cited chunk content.
        2. NLI fallback only if direct matching fails.
        3. Unknown chunk IDs fail.
        4. Missing citations for non-empty answers fail.
        """
        if not citations:
            return 0.0

        normalized = self._to_retrieval_results(retrieved_chunks)
        chunk_map = {c.chunk.id: c.chunk.content for c in normalized}
        chunk_map.update({c.chunk.document_id: c.chunk.content for c in normalized})
        chunk_map.update({c.chunk.metadata.source: c.chunk.content for c in normalized})

        checker = self._get_groundedness_checker()
        verified_count = 0

        for cit in citations:
            if isinstance(cit, Citation):
                chunk_id = cit.chunk_id
                claim_text = cit.quote_snippet
            else:
                chunk_id = getattr(cit, "chunk_id", str(cit))
                claim_text = getattr(
                    cit, "quote_snippet", getattr(cit, "claim", getattr(cit, "snippet", ""))
                )

            # Unknown chunk ID fails
            if chunk_id not in chunk_map:
                continue

            chunk_content = chunk_map[chunk_id].lower()
            clean_snippet = claim_text.strip().lower()

            if not clean_snippet:
                continue

            # 1. First check exact or fuzzy snippet inclusion in chunk content
            snippet_words = [w for w in clean_snippet.split() if len(w) > 3]
            if clean_snippet in chunk_content or (snippet_words and sum(1 for w in snippet_words if w in chunk_content) >= max(1, len(snippet_words) - 1)):
                verified_count += 1
            elif checker:
                # 2. Use NLI only if direct matching fails
                is_ent, _ = checker.verify_claim_entailment(claim_text, chunk_map[chunk_id])
                if is_ent:
                    verified_count += 1

        return float(verified_count) / float(len(citations))

    def compute_appropriate_abstention(
        self, question: GoldenQuestion, status: str, answer: str
    ) -> float:
        """
        Returns 1.0 if the pipeline appropriately abstains when `should_abstain == True`
        or correctly responds with answer when `should_abstain == False`.
        """
        ans_lower = answer.lower()
        abstention_statuses = {
            "LOW_CONFIDENCE",
            "CLARIFICATION_NEEDED",
            "INSUFFICIENT_INFORMATION",
            "UNVERIFIED",
            "AMBIGUOUS",
        }
        abstention_phrases = [
            "low_confidence",
            "clarification_needed",
            "cannot provide a definitive or verified answer",
            "evidence is insufficient",
            "available documentation does not support",
            "unable to verify",
            "insufficient information",
            "do not contain",
            "no information",
            "clarification is required",
            "no verified context",
            "no relevant context available",
            "does not contain information",
            "error during baseline",
        ]
        is_abstained = (
            status in abstention_statuses
            or any(phrase in ans_lower for phrase in abstention_phrases)
        )

        if question.should_abstain:
            return 1.0 if is_abstained else 0.0
        return 1.0 if not is_abstained else 0.0
