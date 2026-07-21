"""
Baseline RAG Pipeline for the Evaluation Service (`services/evaluation`).
Implements a genuinely simpler retrieve-then-generate path with zero self-correction logic,
skipping contradiction detection, groundedness loops, query reformulation, and clarifying workflows.
"""

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

    def run(
        self,
        query: str,
        tenant_id: str = "eval_tenant",
        top_k: int = 5,
        pre_retrieved_chunks: list[DocumentChunk] | None = None,
    ) -> dict[str, Any]:
        """
        Executes baseline retrieval and synthesis.
        Returns dictionary containing:
            - answer (str)
            - confidence_status (str)
            - retrieved_chunks (list[DocumentChunk])
            - citations (list[Citation])
            - latency_seconds (float)
        """
        start_time = time.perf_counter()

        if pre_retrieved_chunks is not None:
            chunks = pre_retrieved_chunks
            status = "VERIFIED"
        else:
            try:
                raw_results, _, status, _ = self.retriever.retrieve(
                    query=query, tenant_id=tenant_id, top_k=top_k
                )
                chunks = []
                for item in raw_results:
                    if isinstance(item, DocumentChunk):
                        chunks.append(item)
                    elif isinstance(item, dict):
                        chunks.append(DocumentChunk.model_validate(item))
            except Exception as exc:
                logger.warning("Retriever search failed in baseline run: %s", exc)
                chunks = []
                status = "UNVERIFIED"

        try:
            answer, citations = self.synthesizer.synthesize(query=query, chunks=chunks)
        except Exception as exc:
            logger.warning("Synthesizer failed in baseline run: %s", exc)
            answer = "Error during baseline generation."
            citations = []

        latency = time.perf_counter() - start_time

        return {
            "answer": answer,
            "confidence_status": str(status.value if hasattr(status, "value") else status),
            "retrieved_chunks": chunks,
            "citations": citations,
            "latency_seconds": latency,
        }
