"""
Baseline RAG Pipeline for the Evaluation Service (`services/evaluation`).
Implements a genuinely simpler retrieve-then-generate path with zero self-correction logic,
skipping contradiction detection, groundedness loops, query reformulation, and clarifying workflows.
"""

import asyncio
import logging
import time
from typing import Any

from shared.models.common import DocumentChunk

from services.generation.app.services.synthesis import FlagshipSynthesizerService
from services.retrieval.app.services.hybrid_retriever import get_hybrid_retriever

logger = logging.getLogger(__name__)


class BaselineRAGPipeline:
    """
    Direct retrieve-then-generate baseline pipeline without self-correction logic.
    Used to calculate honest benchmark comparison metrics against self-correcting RAG.
    """

    def __init__(self) -> None:
        self.retriever = get_hybrid_retriever()
        self.synthesizer = FlagshipSynthesizerService()

    async def run_async(
        self,
        query: str,
        tenant_id: str = "eval_tenant",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Asynchronously executes baseline retrieval and synthesis.
        """
        start_time = time.perf_counter()

        try:
            raw_results, _, status, _ = await self.retriever.retrieve(
                query=query, tenant_id=tenant_id, top_k=top_k
            )
            chunks = []
            for item in raw_results:
                if isinstance(item, DocumentChunk):
                    chunks.append(item)
                elif hasattr(item, "chunk") and isinstance(item.chunk, DocumentChunk):
                    chunks.append(item.chunk)
                elif isinstance(item, dict):
                    chunks.append(DocumentChunk.model_validate(item))
        except Exception as exc:
            logger.error("Retriever search failed in baseline run for query '%s': %s", query, exc)
            raise RuntimeError(f"Baseline retrieval infrastructure failed for query '{query}': {exc}") from exc

        try:
            answer, citations = self.synthesizer.synthesize(query=query, chunks=chunks)
        except Exception as exc:
            logger.error("Synthesizer failed in baseline run: %s", exc)
            raise RuntimeError(f"Baseline synthesizer failed for query '{query}': {exc}") from exc

        # Fix baseline status for abstentions & clarifications
        ans_lower = answer.lower()
        status_str = str(status.value if hasattr(status, "value") else status)
        if any(p in ans_lower for p in ["insufficient information", "no relevant context", "cannot provide", "no information"]):
            status_str = "LOW_CONFIDENCE"
        elif any(p in ans_lower for p in ["please clarify", "clarification", "depends on manager"]):
            status_str = "CLARIFICATION_NEEDED"

        latency = time.perf_counter() - start_time

        return {
            "answer": answer,
            "confidence_status": status_str,
            "retrieved_chunks": chunks,
            "citations": citations,
            "latency_seconds": latency,
        }

    def run(
        self,
        query: str,
        tenant_id: str = "eval_tenant",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Executes baseline retrieval and synthesis synchronously.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self.run_async(query, tenant_id, top_k))
        return asyncio.run(self.run_async(query, tenant_id, top_k))
