"""
Clarification & Fallback Service (`ClarificationAndFallbackService`).
Generates clarifying questions for same-date unresolvable contradictions,
and transparent low-confidence fallback responses when retries are exhausted.
"""

import logging
from typing import Any

from shared.config import get_settings
from shared.models.common import RetrievalResult

logger = logging.getLogger(__name__)


class ClarificationAndFallbackService:
    """
    Generates user-facing clarifying questions when contradictory context cannot be reconciled
    automatically, and honest low-confidence responses when search attempts are exhausted.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = self.settings.EMBEDDING_PROVIDER
        self._openai_client: Any = None

        if self.provider == "openai" and self.settings.OPENAI_API_KEY:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self.settings.OPENAI_API_KEY)
            except Exception as exc:
                logger.warning("Failed to initialize OpenAI client for Clarification: %s", exc)

    def generate_clarification(
        self, query: str, contradictory_chunks: list[RetrievalResult]
    ) -> str:
        """
        Generates a clarifying question highlighting conflicting claims across same-date sources.
        """
        conflict_summary = []
        for i, res in enumerate(contradictory_chunks[:3], 1):
            meta = res.chunk.metadata
            snippet = res.chunk.content.strip()[:150]
            conflict_summary.append(
                f"Source {i} (`{res.chunk.id}`, ver `{meta.version_id}`): \"{snippet}...\""
            )
        conflicts_text = "\n".join(conflict_summary)

        if self._openai_client:
            try:
                prompt = (
                    "You are a helpful and precise AI coding and reference assistant. "
                    "The user asked a question, but retrieved documents from the same date "
                    "or version contain directly contradictory claims. Write a concise "
                    "clarifying question asking the user how to proceed or which condition "
                    "or source they want to follow.\n\n"
                    f"User Query: {query}\n\n"
                    f"Conflicting Document Excerpts:\n{conflicts_text}\n\n"
                    "Respond with only the clarifying question to present to the user."
                )
                response = self._openai_client.chat.completions.create(
                    model=self.settings.LLM_MODEL_NAME or "gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                question = (response.choices[0].message.content or "").strip()
                if question and len(question) > 15:
                    return question
            except Exception as exc:
                logger.warning("OpenAI clarification generation error: %s. Using fallback.", exc)

        # Offline / local clarification generation
        sources_str = ", ".join([f"`{c.chunk.id}`" for c in contradictory_chunks[:2]])
        return (
            f"We found conflicting statements regarding '{query}' across documents with identical "
            f"version metadata ({sources_str}). For example, one source states different "
            f"requirements than the other. Could you please clarify which specific document or "
            f"scenario you would like us to follow?"
        )

    def generate_low_confidence_response(
        self, query: str, attempt_count: int, reasoning: str
    ) -> str:
        """
        Generates an honest, transparent response when retrieval retries are exhausted.
        """
        if self._openai_client:
            try:
                prompt = (
                    "You are a transparent, highly truthful AI assistant. "
                    "We attempted to answer the user's query and performed multiple reformulation "
                    "attempts, but could not retrieve sufficiently confident or grounded "
                    "reference material. Write a polite response stating that we cannot "
                    "definitively answer the question with available documents, summarizing the "
                    "search efforts and what remains unconfirmed.\n\n"
                    f"User Query: {query}\n"
                    f"Attempts Made: {attempt_count}\n"
                    f"Last Audit Reasoning: {reasoning}\n\n"
                    "Respond with only the low-confidence response to present to the user."
                )
                response = self._openai_client.chat.completions.create(
                    model=self.settings.LLM_MODEL_NAME or "gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                resp = (response.choices[0].message.content or "").strip()
                if resp and len(resp) > 20:
                    return resp
            except Exception as exc:
                logger.warning("OpenAI low-confidence generation error: %s. Using fallback.", exc)

        # Offline / local transparent low-confidence response
        return (
            f"I cannot provide a definitive or verified answer to your query regarding '{query}'. "
            f"We conducted {attempt_count} search and reformulation attempts (including query "
            f"expansion and keyword broadening), but the retrieved documentation did not meet our "
            f"strict confidence and groundedness thresholds (`{reasoning}`). Please verify the "
            f"documentation or rephrase with additional specifics."
        )


_clarification_service: ClarificationAndFallbackService | None = None


def get_clarification_service() -> ClarificationAndFallbackService:
    """Singleton getter for ClarificationAndFallbackService."""
    global _clarification_service
    if _clarification_service is None:
        _clarification_service = ClarificationAndFallbackService()
    return _clarification_service
