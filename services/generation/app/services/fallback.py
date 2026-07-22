"""
Fallback generation synthesizer (`FallbackSynthesizerService`).
Executes when the primary circuit breaker is OPEN (`GPT-4o-mini`), synthesizing grounded
answers while appending an explicit `reduced accuracy` disclaimer to the output.
"""

import json
import logging
from typing import Any

from shared.config import Settings, get_settings
from shared.models.common import Citation, DocumentChunk

logger = logging.getLogger(__name__)

FALLBACK_DISCLAIMER = (
    "\n\n[Disclaimer: This response was generated using fallback mode with "
    "reduced accuracy due to primary service degradation.]"
)


class FallbackSynthesizerService:
    """
    Synthesizes answers using a lighter, high-availability fallback model (`GPT-4o-mini`)
    when primary generation (`FlagshipSynthesizerService`) is degraded or circuit-broken.
    Appends an explicit reduced accuracy disclaimer to ensure transparency.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_name = self.settings.GENERATION_FALLBACK_MODEL or "gemini-3.5-flash"
        self._openai_client: Any = None

        from shared.utils.llm_client import get_llm_client
        self._openai_client = get_llm_client(self.settings, timeout=5.0)

    def synthesize(self, query: str, chunks: list[DocumentChunk]) -> tuple[str, list[Citation]]:
        """
        Synthesizes a fallback answer for `query` grounded on `chunks` with explicit disclaimer.
        Returns:
            - answer (str): Synthesized answer text plus fallback disclaimer.
            - citations (list[Citation]): Grounded citations mapped to `chunk_id`.
        """
        if not chunks:
            return f"No verified context available to answer the query.{FALLBACK_DISCLAIMER}", []

        chunk_map: dict[str, DocumentChunk] = {c.id: c for c in chunks}

        context_blocks = []
        for c in chunks:
            block = (
                f"Chunk ID: {c.id}\n"
                f"Document ID: {c.document_id}\n"
                f"Source: {c.metadata.source} (Page {c.metadata.page_number})\n"
                f"Content:\n{c.content}\n"
            )
            context_blocks.append(block)
        context_str = "\n---\n".join(context_blocks)

        system_prompt = (
            "You are a fallback RAG generation assistant. Answer the query using "
            "the reference chunks provided. Note: Reference chunks may contain PII redaction placeholders "
            "like <PERSON>, <EMAIL_ADDRESS>, etc. Treat them naturally or work around them without exposing "
            "raw redaction tags where avoidable. Output structured JSON matching:\n"
            "{\n"
            '  "answer": "Synthesized fallback answer...",\n'
            '  "claims": [\n'
            "    {\n"
            '      "claim_text": "Assertion",\n'
            '      "chunk_id": "Supporting Chunk ID",\n'
            '      "quote_snippet": "Quote excerpt"\n'
            "    }\n"
            "  ]\n"
            "}"
        )
        user_prompt = f"Reference Context:\n{context_str}\n\nUser Query: {query}"

        if not self._openai_client:
            from shared.utils.llm_client import get_llm_client
            self._openai_client = get_llm_client(self.settings, timeout=10.0)

        if not self._openai_client:
            raise RuntimeError("LLM API client is not configured (`_openai_client` is None) in FallbackSynthesizerService.")

        response = None
        for attempt in range(3):
            try:
                logger.info(
                    "Invoking fallback LLM API (model: '%s') for query: '%s' (attempt %d)",
                    self.model_name,
                    query,
                    attempt + 1,
                )
                response = self._openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                break
            except Exception as exc:
                if attempt < 2 and ("429" in str(exc) or "RateLimit" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)):
                    sleep_time = (attempt + 1) * 3.0
                    logger.warning("Fallback LLM rate limit hit (%s). Retrying in %.1fs...", exc, sleep_time)
                    import time
                    time.sleep(sleep_time)
                else:
                    logger.error("Fallback LLM synthesis failed (`%s`).", exc)
                    raise RuntimeError(f"Fallback LLM synthesis failed: {exc}") from exc

        if not response or not response.choices:
            raise RuntimeError("Fallback LLM API returned an empty response.")

        content = response.choices[0].message.content or "{}"
        logger.info("Fallback LLM response received successfully. Raw content: %s", content)

        data = json.loads(content)
        base_answer = str(data.get("answer", "")).strip()
        raw_claims = data.get("claims", [])

        citations: list[Citation] = []
        for item in raw_claims:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("chunk_id", "")).strip()
            snippet = str(item.get("quote_snippet", "")).strip()
            target_chunk = chunk_map.get(cid) or chunks[0]
            citations.append(
                Citation(
                    chunk_id=target_chunk.id,
                    document_id=target_chunk.document_id,
                    source=target_chunk.metadata.source,
                    page_number=target_chunk.metadata.page_number,
                    quote_snippet=snippet or target_chunk.content[:100],
                )
            )

        if not citations and chunks:
            top_c = chunks[0]
            citations.append(
                Citation(
                    chunk_id=top_c.id,
                    document_id=top_c.document_id,
                    source=top_c.metadata.source,
                    page_number=top_c.metadata.page_number,
                    quote_snippet=top_c.content[:100],
                )
            )

        if not base_answer:
            raise RuntimeError("Fallback LLM synthesized answer is empty.")

        full_answer = f"{base_answer}{FALLBACK_DISCLAIMER}"
        return full_answer, citations
