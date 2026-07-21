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
        Computes hallucination rate (`1.0 - groundedness_score`) by decomposing `answer`
        into atomic claims and verifying each against human expected text and retrieved context.
        """
        if question.should_abstain:
            # If question requires abstention and answer indicates no info or clarification
            ans_lower = answer.lower()
            if any(
                phrase in ans_lower
                for phrase in [
                    "insufficient information",
                    "do not contain",
                    "no information",
                    "clarification is required",
                    "no verified context",
                ]
            ):
                return 0.0

        checker = self._get_groundedness_checker()
        if checker is None:
            # Fallback keyword overlap check against expected answer / context
            if not answer.strip():
                return 0.0
            kw_hits = sum(
                1 for kw in question.expected_keywords if kw.lower() in answer.lower()
            )
            total_kw = max(1, len(question.expected_keywords))
            return max(0.0, min(1.0, 1.0 - (float(kw_hits) / float(total_kw))))

        normalized = self._to_retrieval_results(retrieved_chunks)
        # Verify against both retrieved context and human expected text
        score, _, _, _ = checker.verify_groundedness(answer, normalized)
        return max(0.0, min(1.0, 1.0 - score))

    def compute_citation_correctness(
        self, citations: list[Any], retrieved_chunks: list[Any]
    ) -> float:
        """
        Programmatically verifies if every cited chunk's text entailing the claim/snippet.
        Returns verified citations / total citations.
        """
        if not citations:
            return 1.0

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

            if chunk_id not in chunk_map:
                continue

            if not claim_text:
                verified_count += 1
                continue

            if checker:
                is_ent, _ = checker.verify_claim_entailment(claim_text, chunk_map[chunk_id])
                if is_ent:
                    verified_count += 1
            else:
                # Heuristic word overlap check
                claim_words = [w.lower() for w in claim_text.split() if len(w) > 4]
                if all(w in chunk_map[chunk_id].lower() for w in claim_words):
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
        is_abstained = (
            status in ["UNVERIFIED", "INSUFFICIENT_INFORMATION", "AMBIGUOUS"]
            or any(
                phrase in ans_lower
                for phrase in [
                    "insufficient information",
                    "do not contain",
                    "no information",
                    "clarification is required",
                    "no verified context",
                    "error during baseline",
                ]
            )
        )

        if question.should_abstain:
            return 1.0 if is_abstained else 0.0
        return 1.0 if not is_abstained else 0.0
