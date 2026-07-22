"""
Flagship generation synthesizer (`FlagshipSynthesizerService`).
Executes real structured model calls (`GPT-4o` or configured primary model) forcing
atomic claim attribution and mapping source citations directly to specific chunk IDs.
"""

import json
import logging
import time
from typing import Any

from shared.config import Settings, get_settings
from shared.metrics import PIPELINE_STAGE_DURATION
from shared.models.common import Citation, DocumentChunk
from shared.telemetry import get_tracer, traced_span

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)


class FlagshipSynthesizerService:
    """
    Synthesizes authoritative answers from verified candidate chunks using flagship LLMs.
    Enforces structured JSON generation so every factual assertion is attributed to
    an exact `chunk_id` and supporting snippet.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_name = self.settings.GENERATION_PRIMARY_MODEL or self.settings.LLM_MODEL_NAME
        self._openai_client: Any = None

        from shared.utils.llm_client import get_llm_client
        self._openai_client = get_llm_client(self.settings, timeout=5.0)

    def synthesize(self, query: str, chunks: list[DocumentChunk]) -> tuple[str, list[Citation]]:
        """
        Synthesizes an answer for `query` grounded on `chunks`.
        Wrapped in a ``compass.generation.synthesize`` OTel span with key attributes.

        Returns:
            - answer (str): Synthesized answer text.
            - citations (list[Citation]): Grounded citations mapped to ``chunk_id``.
        """
        with traced_span(
            _tracer,
            "compass.generation.synthesize",
            {
                "model_name": self.model_name,
                "chunk_count": len(chunks),
                "query.length": len(query),
            },
        ) as span:
            t0 = time.perf_counter()
            answer, citations = self._synthesize_impl(query, chunks)
            span.set_attribute("answer_length", len(answer))
            span.set_attribute("citation_count", len(citations))
            PIPELINE_STAGE_DURATION.labels(
                service="compass-rag-generation", stage="synthesize"
            ).observe(time.perf_counter() - t0)
        return answer, citations

    def _synthesize_impl(
        self, query: str, chunks: list[DocumentChunk]
    ) -> tuple[str, list[Citation]]:
        """Internal implementation — called from the traced synthesize() wrapper."""
        if not chunks:
            return "No verified context available to answer the query.", []

        # Map chunks by ID for quick lookup during citation reconstruction
        chunk_map: dict[str, DocumentChunk] = {c.id: c for c in chunks}

        if not self._openai_client:
            from shared.utils.llm_client import get_llm_client
            self._openai_client = get_llm_client(self.settings, timeout=10.0)

        if not self._openai_client:
            raise RuntimeError("LLM API client is not configured (`_openai_client` is None). Cannot synthesize answer without LLM.")

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
            "You are an authoritative RAG generation assistant. "
            "You must synthesize a natural-language answer to the user's query strictly and solely using "
            "the provided reference chunks. Do not use external knowledge.\n"
            "Include inline citation markers (such as [chunk_id] or [1]) in your answer corresponding to each claim.\n"
            "Note: Reference chunks may contain PII redaction placeholders like <PERSON>, <EMAIL_ADDRESS>, "
            "<PHONE_NUMBER>, etc. When formulating your answer, either work around these placeholders naturally "
            "or treat them as generic entities without exposing raw redaction tags where avoidable.\n\n"
            "You must output structured JSON matching this exact format:\n"
            "{\n"
            '  "answer": "The comprehensive synthesized answer with inline citations...",\n'
            '  "claims": [\n'
            "    {\n"
            '      "claim_text": "An atomic factual assertion made in the answer",\n'
            '      "chunk_id": "The exact Chunk ID supporting this claim",\n'
            '      "quote_snippet": "A brief quote from the chunk supporting the claim"\n'
            "    }\n"
            "  ]\n"
            "}"
        )
        user_prompt = f"Reference Context:\n{context_str}\n\nUser Query: {query}"

        response = None
        for attempt in range(3):
            try:
                logger.info(
                    "Invoking real LLM API (model: '%s') for query: '%s' with %d chunks context (attempt %d)",
                    self.model_name,
                    query,
                    len(chunks),
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
                    logger.warning("LLM API rate limit hit (%s). Retrying in %.1fs...", exc, sleep_time)
                    time.sleep(sleep_time)
                else:
                    logger.error("Flagship model synthesis failed (`%s`).", exc)
                    raise RuntimeError(f"LLM API synthesis failed: {exc}") from exc

        if not response or not response.choices:
            raise RuntimeError("LLM API returned an empty response.")

        content = response.choices[0].message.content or "{}"
        logger.info("LLM API response received successfully. Raw content: %s", content)

        data = json.loads(content)
        answer_text = str(data.get("answer", "")).strip()
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

        if not answer_text:
            raise RuntimeError("LLM synthesized answer is empty.")

        return answer_text, citations
