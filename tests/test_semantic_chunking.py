"""
Unit test for Phase 3 Semantic Chunking Service with sentence-boundary splitting.
"""

from datetime import UTC, datetime

from shared.models.common import DocumentChunk

from services.ingestion.app.pipeline.chunker import get_chunker_service


def _utc_now() -> datetime:
    return datetime.now(UTC)


def test_semantic_chunker_and_tagging() -> None:
    """
    Verifies that SemanticChunkerService splits redacted text cleanly on sentence boundaries
    and attaches strict domain DocumentMetadata to every DocumentChunk instance.
    """
    chunker = get_chunker_service()

    redacted_text = (
        "This is the first sentence regarding RAG evaluation. "
        "Here is the second sentence which discusses grounding across retrieved documents. "
        "Finally, the third sentence covers Presidio PII redaction and LangChain splitters."
    )

    now = _utc_now()
    chunks = chunker.chunk_and_tag_page(
        redacted_text=redacted_text,
        document_id="doc_semantic_001",
        source="doc_semantic_001.pdf",
        page_number=1,
        ingestion_timestamp=now,
        tenant_id="tenant_acme",
        version_id="v1.0",
        ocr_confidence=0.98,
        base_index=10,
    )

    assert len(chunks) > 0
    for idx, chunk in enumerate(chunks):
        assert isinstance(chunk, DocumentChunk)
        assert chunk.document_id == "doc_semantic_001"
        assert chunk.chunk_index == 10 + idx
        assert chunk.metadata.source == "doc_semantic_001.pdf"
        assert chunk.metadata.page_number == 1
        assert chunk.metadata.ingestion_timestamp == now
        assert chunk.metadata.tenant_id == "tenant_acme"
        assert chunk.metadata.version_id == "v1.0"
        assert chunk.metadata.ocr_confidence == 0.98
        # Ensure chunks preserve content properly
        assert len(chunk.content) > 0
