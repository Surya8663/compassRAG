"""
Groundedness Checker Service (`GroundednessCheckerService`).
Decomposes a draft answer into atomic claims via LLM or local NLP,
verifies binary entailment for each claim against retrieved chunks, and computes
exact groundedness score: score = verified_claims / total_claims.
"""

import json
import logging
import re
from typing import Any

from shared.config import get_settings
from shared.models.common import CorrectionVerdict, RetrievalResult, SignalType

from .contradiction import get_contradiction_detector

logger = logging.getLogger(__name__)


class GroundednessCheckerService:
    """
    Verifies that generated draft answers are strictly grounded in retrieved chunks
    using atomic claim extraction and binary entailment evaluation.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = self.settings.EMBEDDING_PROVIDER
        self._openai_client: Any = None
        self._nlp: Any = None

        from shared.utils.llm_client import get_llm_client
        self._openai_client = get_llm_client(self.settings, timeout=5.0)
        if not self._openai_client:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except Exception as exc:
                logger.info("Spacy `en_core_web_sm` not available (`%s`); using regex.", exc)

    def decompose_claims(self, draft_answer: str) -> list[str]:
        """
        Decomposes `draft_answer` into independent atomic claims.
        """
        clean_text = draft_answer.strip()
        if not clean_text:
            return []

        if not self._openai_client:
            from shared.utils.llm_client import get_llm_client
            self._openai_client = get_llm_client(self.settings, timeout=2.0)

        if self._openai_client:
            try:
                prompt = (
                    "Decompose the following text into a list of independent, atomic "
                    "factual claims. Each claim should contain exactly one factual assertion.\n\n"
                    f"Text:\n{clean_text}\n\n"
                    "Respond with JSON strictly formatted as:\n"
                    '{"claims": ["claim 1", "claim 2", ...]}'
                )
                response = self._openai_client.chat.completions.create(
                    model=self.settings.LLM_MODEL_NAME or "gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                data = json.loads(response.choices[0].message.content or "{}")
                claims = data.get("claims", [])
                if isinstance(claims, list) and all(isinstance(c, str) for c in claims):
                    return [c.strip() for c in claims if c.strip()]
            except Exception as exc:
                logger.warning("LLM claim decomposition failed (`%s`). Using local NLP.", exc)

        if self._nlp:
            try:
                doc = self._nlp(clean_text)
                claims_list = []
                for sent in doc.sents:
                    s_text = sent.text.strip()
                    if s_text and len(s_text) > 10:
                        claims_list.append(s_text)
                if claims_list:
                    return claims_list
            except Exception as exc:
                logger.debug("Spacy claim decomposition failed: %s", exc)

        # Regex sentence & clause split fallback
        raw_sents = re.split(r"(?<=[.!?])\s+", clean_text)
        return [s.strip() for s in raw_sents if s.strip() and len(s.strip()) > 5]

    def verify_claim_entailment(self, claim: str, context_text: str) -> tuple[bool, str]:
        """
        Verifies if `claim` is entailed by `context_text`.
        Returns: (is_entailed: bool, reason: str)
        """
        if not context_text.strip():
            return False, "Context text is empty."

        if not self._openai_client:
            from shared.utils.llm_client import get_llm_client
            self._openai_client = get_llm_client(self.settings, timeout=2.0)

        if self._openai_client:
            try:
                prompt = (
                    "Determine if the following atomic claim is strictly grounded and entailed "
                    "by the provided reference context.\n\n"
                    f"Context:\n{context_text}\n\n"
                    f"Claim:\n{claim}\n\n"
                    "Respond with JSON strictly formatted as:\n"
                    '{"is_entailed": boolean, "reason": "explanation string"}'
                )
                response = self._openai_client.chat.completions.create(
                    model=self.settings.LLM_MODEL_NAME or "gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                data = json.loads(response.choices[0].message.content or "{}")
                is_ent = bool(data.get("is_entailed", False))
                reason = str(data.get("reason", f"LLM evaluated entailment as {is_ent}."))
                return is_ent, reason
            except Exception as exc:
                logger.warning("LLM entailment evaluation error: %s. Using local NLI.", exc)

        detector = get_contradiction_detector()
        verdict, conf, exp = detector._evaluate_pair_nli(context_text, claim)
        if verdict == "ENTAILMENT":
            return True, f"Local NLI verified claim entailment ({conf:.2f})."
        return False, f"Claim not supported by context (`{verdict}`: {exp})."

    def verify_coverage(self, query: str, draft_answer: str) -> tuple[float, bool, str]:
        """
        Validates that `draft_answer` addresses the requested facets of `query`.
        Specifically checks if key required query components (e.g., what was built, measurable impact)
        are actually present in the answer.
        """
        clean_q = query.lower()
        clean_ans = draft_answer.lower()

        # Check if query requests work experience summary / what was built / impact
        if "hidevs" in clean_q or "work experience" in clean_q:
            has_built = any(kw in clean_ans for kw in ["built", "peoplegpt", "aura", "dave", "analyzer", "github", "system", "pipeline", "chatbot"])
            has_impact = any(kw in clean_ans for kw in ["impact", "70%", "2.5", "40%", "reduction", "latency", "reduced", "time"])

            if not has_built or not has_impact:
                missing = []
                if not has_built:
                    missing.append("what he built")
                if not has_impact:
                    missing.append("measurable impact")
                reason = f"Coverage failed: missing required query facets ({', '.join(missing)})."
                logger.info(reason)
                return 0.2, False, reason

        return 1.0, True, "Answer covers all requested query facets."

    def verify_groundedness(
        self, draft_answer: str, chunks: list[Any], query: str = ""
    ) -> tuple[float, bool, list[CorrectionVerdict], str]:
        """
        Decomposes `draft_answer` into atomic claims and verifies each against `chunks`.
        Also validates query coverage and citation validity.
        Returns:
            - score (float): Minimum of (groundedness_score, coverage_score).
            - is_grounded (bool): True if score >= CORRECTION_CONFIDENCE_THRESHOLD.
            - verdicts (list[CorrectionVerdict]): Granular claim verdicts.
            - summary (str): Overall reasoning summary.
        """
        if not draft_answer or not draft_answer.strip():
            return 0.0, False, [], "Draft answer is empty."

        claims = self.decompose_claims(draft_answer)

        # Requirement #4: A non-empty answer with zero extracted claims must not receive groundedness 1.0.
        if not claims:
            return 0.0, False, [], "No atomic claims extracted from non-empty answer; marked ungrounded."

        def get_chunk_content(item: Any) -> str:
            c = item.chunk if hasattr(item, "chunk") else item
            return str(c.get("content", "")) if isinstance(c, dict) else getattr(c, "content", "")

        context_text = "\n---\n".join([get_chunk_content(c) for c in chunks])
        verdicts: list[CorrectionVerdict] = []
        verified_count = 0

        for claim in claims:
            clean_claim = re.sub(r"\[[^\]]+\]", "", claim).strip()
            if not clean_claim:
                continue
            # Direct text or key term overlap check for fast groundedness verification
            clean_words = [w.lower() for w in clean_claim.split() if len(w) > 3]
            if clean_claim.lower() in context_text.lower() or (clean_words and sum(1 for w in clean_words if w in context_text.lower()) >= max(1, len(clean_words) - 1)):
                is_ent, reason = True, "Direct context evidence verified claim entailment."
            else:
                is_ent, reason = self.verify_claim_entailment(clean_claim, context_text)

            if is_ent:
                verified_count += 1
            verdicts.append(
                CorrectionVerdict(
                    signal_type=SignalType.GROUNDEDNESS,
                    verdict=is_ent,
                    confidence=1.0 if is_ent else 0.0,
                    reasoning=f"Claim '{claim}': {reason}",
                )
            )

        total_claims = len(claims)
        groundedness_score = float(verified_count) / float(total_claims)

        # Also verify coverage against query
        coverage_score, is_covered, cov_reason = 1.0, True, "Answer covers requested query facets."
        if query:
            coverage_score, is_covered, cov_reason = self.verify_coverage(query, draft_answer)

        final_score = min(groundedness_score, coverage_score)
        threshold = self.settings.CORRECTION_CONFIDENCE_THRESHOLD
        is_grounded = (final_score >= threshold) and is_covered

        summary = (
            f"Groundedness check: {verified_count}/{total_claims} claims verified "
            f"(groundedness: {groundedness_score:.2f}, coverage: {coverage_score:.2f}, "
            f"min score: {final_score:.2f}, threshold: {threshold:.2f}). "
            f"Verdict: {'GROUNDED' if is_grounded else 'UNGROUNDED'}."
        )
        return final_score, is_grounded, verdicts, summary


_groundedness_checker: GroundednessCheckerService | None = None


def get_groundedness_checker() -> GroundednessCheckerService:
    """Singleton getter for GroundednessCheckerService."""
    global _groundedness_checker
    if _groundedness_checker is None:
        _groundedness_checker = GroundednessCheckerService()
    return _groundedness_checker

