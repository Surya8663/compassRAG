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
        self.model_name = self.settings.GENERATION_FALLBACK_MODEL or "gpt-4o-mini"
        self._openai_client: Any = None

        if self.settings.OPENAI_API_KEY:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self.settings.OPENAI_API_KEY)
            except Exception as exc:
                logger.warning("Failed to initialize OpenAI client in FallbackSynthesizer: %s", exc)

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

        if self._openai_client:
            try:
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
                    "the reference chunks provided. Output structured JSON matching:\n"
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

                response = self._openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )

                content = response.choices[0].message.content or "{}"
                data = json.loads(content)
                base_answer = str(data.get("answer", "")).strip() or chunks[0].content
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

                full_answer = f"{base_answer}{FALLBACK_DISCLAIMER}"
                return full_answer, citations

            except Exception as exc:
                logger.warning(
                    "OpenAI fallback generation failed (`%s`). Using local fallback.", exc
                )

        logger.info("Using local fallback synthesis.")
        top_chunk = chunks[0]
        base_answer = f"Synthesized from {top_chunk.metadata.source}: {top_chunk.content}"
        citations = [
            Citation(
                chunk_id=top_chunk.id,
                document_id=top_chunk.document_id,
                source=top_chunk.metadata.source,
                page_number=top_chunk.metadata.page_number,
                quote_snippet=top_chunk.content[:100],
            )
        ]
        full_answer = f"{base_answer}{FALLBACK_DISCLAIMER}"
        return full_answer, citations
