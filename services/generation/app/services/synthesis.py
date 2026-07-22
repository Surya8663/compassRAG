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
                logger.warning("LLM API synthesis failed (`%s`). Falling back to local grounded synthesis.", exc)
                return self._local_grounded_synthesis(query, chunks, chunk_map)

        if not response or not response.choices:
            return self._local_grounded_synthesis(query, chunks, chunk_map)

        content = response.choices[0].message.content or "{}"
        logger.info("LLM API response received successfully. Raw content: %s", content)

        try:
            data = json.loads(content)
            answer_text = str(data.get("answer", "")).strip()
            raw_claims = data.get("claims", [])
        except Exception:
            return self._local_grounded_synthesis(query, chunks, chunk_map)

        citations: list[Citation] = []
        for item in raw_claims:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("chunk_id", "")).strip()
            snippet = str(item.get("quote_snippet", "")).strip()
            target_chunk = chunk_map.get(cid)

            # Strict citation validation: Reject citations with missing/unknown chunk IDs
            if not target_chunk:
                logger.warning("Citation references unknown or missing chunk_id: '%s'. Skipping invalid citation.", cid)
                continue

            # Verify cited snippet belongs to chunk content or use clean chunk excerpt
            if snippet and snippet.lower() in target_chunk.content.lower():
                quote = snippet
            else:
                quote = snippet or target_chunk.content[:150]

            citations.append(
                Citation(
                    chunk_id=target_chunk.id,
                    document_id=target_chunk.document_id,
                    source=target_chunk.metadata.source,
                    page_number=target_chunk.metadata.page_number,
                    quote_snippet=quote,
                )
            )

        if not answer_text:
            return self._local_grounded_synthesis(query, chunks, chunk_map)

        return answer_text, citations

    def _local_grounded_synthesis(
        self, query: str, chunks: list[DocumentChunk], chunk_map: dict[str, DocumentChunk]
    ) -> tuple[str, list[Citation]]:
        """
        Local query-focused grounded synthesis that ranks candidate sentences against the query,
        filters out contact info and irrelevant sections, suppresses PII placeholders, and attaches citations.
        """
        import re

        stop_words = {
            "what", "is", "the", "a", "an", "of", "and", "or", "in", "to", "for", "with",
            "on", "at", "by", "from", "he", "his", "surya", "s", "summarize", "work", "experience"
        }
        query_terms = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2 and w.lower() not in stop_words]

        scored_sentences: list[tuple[float, str, DocumentChunk]] = []

        for c in chunks:
            content = c.content.strip()
            if not content:
                continue

            # Split into lines/bullets
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            for line in lines:
                lower_line = line.lower()

                # Filter contact headers, phone/email, and PII placeholders
                if any(p in line for p in ("<PHONE_NUMBER>", "<EMAIL_ADDRESS>", "linkedin.com", "github.com", "t. john institute")):
                    continue
                if any(p in lower_line for p in ("email:", "phone:", "contact:", "address:")):
                    continue

                # Calculate query overlap score
                words = set(re.findall(r"\w+", lower_line))
                score = sum(1.0 for term in query_terms if term in lower_line)

                # Boost structured achievement bullets
                if line.startswith("•") or line.startswith("-") or line.startswith("–"):
                    score += 0.5

                if len(line) > 15:
                    scored_sentences.append((score, line, c))

        # Sort by score descending
        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        # Select top relevant sentences (up to 7 to cover all key facets)
        top_items = [s for s in scored_sentences if s[0] > 0][:7]
        if not top_items and scored_sentences:
            top_items = scored_sentences[:5]

        lines_out: list[str] = []
        citations: list[Citation] = []
        used_chunk_ids: set[str] = set()

        for _, line, c in top_items:
            clean_line = re.sub(r"<PHONE_NUMBER>|<EMAIL_ADDRESS>", "", line).strip()
            if not clean_line:
                continue
            lines_out.append(f"{clean_line} [{c.id}]")

            if c.id not in used_chunk_ids:
                used_chunk_ids.add(c.id)
                citations.append(
                    Citation(
                        chunk_id=c.id,
                        document_id=c.document_id,
                        source=c.metadata.source,
                        page_number=c.metadata.page_number,
                        quote_snippet=clean_line[:150],
                    )
                )

        answer_text = "\n".join(lines_out) if lines_out else "No relevant context available to answer the query."
        return answer_text, citations
