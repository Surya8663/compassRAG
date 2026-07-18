"""
Semantic Chunking and Tagging Service.
Uses LangChain's RecursiveCharacterTextSplitter with natural sentence boundaries to split
redacted text into semantic chunks, attaching domain DocumentMetadata to each chunk.
"""

import logging
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from shared.models.common import DocumentChunk, DocumentMetadata

logger = logging.getLogger(__name__)


class SemanticChunkerService:
    """
    Sentence-boundary-aware chunking service that splits redacted document pages into discrete
    semantic segments and attaches strict domain metadata.
    """

    def __init__(
        self, chunk_size: int = 500, chunk_overlap: int = 50
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            keep_separator=True,
        )

    def chunk_and_tag_page(
        self,
        redacted_text: str,
        document_id: str,
        source: str,
        page_number: int,
        ingestion_timestamp: datetime,
        tenant_id: str,
        version_id: str,
        ocr_confidence: float | None = None,
        base_index: int = 0,
    ) -> list[DocumentChunk]:
        """
        Splits `redacted_text` into semantic sentence-boundary chunks and attaches
        strict `DocumentMetadata` to each segment.
        """
        if not redacted_text or not redacted_text.strip():
            logger.warning(
                "Received empty redacted text for document_id=%s, page_number=%d",
                document_id,
                page_number,
            )
            return []

        segments = self.splitter.split_text(redacted_text)
        chunks: list[DocumentChunk] = []

        for idx, segment_text in enumerate(segments):
            clean_text = segment_text.strip()
            if not clean_text:
                continue

            metadata = DocumentMetadata(
                source=source,
                page_number=page_number,
                ingestion_timestamp=ingestion_timestamp,
                ocr_confidence=ocr_confidence,
                tenant_id=tenant_id,
                version_id=version_id,
            )

            chunk_id = f"{document_id}_p{page_number}_c{base_index + idx}"
            chunk_obj = DocumentChunk(
                id=chunk_id,
                document_id=document_id,
                content=clean_text,
                chunk_index=base_index + idx,
                metadata=metadata,
            )
            chunks.append(chunk_obj)

        logger.debug(
            "Created %d semantic chunks for document_id=%s page_number=%d",
            len(chunks),
            document_id,
            page_number,
        )
        return chunks


# Singleton instance helper
_chunker_service_instance: SemanticChunkerService | None = None


def get_chunker_service() -> SemanticChunkerService:
    """
    Returns a cached singleton instance of SemanticChunkerService.
    """
    global _chunker_service_instance
    if _chunker_service_instance is None:
        _chunker_service_instance = SemanticChunkerService()
    return _chunker_service_instance
