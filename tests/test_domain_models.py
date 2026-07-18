from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from shared.models.common import (
    Citation,
    ConfidenceStatus,
    CorrectionVerdict,
    Document,
    DocumentChunk,
    DocumentMetadata,
    QueryRequest,
    QueryResponse,
    RetrievalResult,
    SignalType,
    TenantContext,
)


@pytest.fixture
def valid_metadata() -> DocumentMetadata:
    return DocumentMetadata(
        source="s3://bucket/tenant_abc/report_v1.pdf",
        page_number=5,
        ingestion_timestamp=datetime.now(UTC),
        ocr_confidence=0.98,
        tenant_id="tenant_abc",
        version_id="v1.0.0",
    )


@pytest.fixture
def valid_tenant_context() -> TenantContext:
    return TenantContext(
        tenant_id="tenant_abc",
        user_id="user_123",
        roles=["admin", "analyst"],
        permissions=["read:documents", "execute:queries"],
    )


@pytest.fixture
def valid_chunk(valid_metadata: DocumentMetadata) -> DocumentChunk:
    return DocumentChunk(
        id="chunk_001",
        document_id="doc_100",
        content="The self-correcting RAG architecture uses multi-stage evaluation.",
        chunk_index=0,
        metadata=valid_metadata,
        score=0.89,
    )


# ==============================================================================
# TENANT CONTEXT TESTS
# ==============================================================================

def test_tenant_context_valid(valid_tenant_context: TenantContext) -> None:
    assert valid_tenant_context.tenant_id == "tenant_abc"
    assert "admin" in valid_tenant_context.roles


def test_tenant_context_missing_required_fields() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TenantContext(user_id="user_123")
    assert "tenant_id" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        TenantContext(tenant_id="tenant_abc")
    assert "user_id" in str(exc_info.value)


# ==============================================================================
# DOCUMENT METADATA TESTS
# ==============================================================================

def test_document_metadata_valid(valid_metadata: DocumentMetadata) -> None:
    assert valid_metadata.ocr_confidence == 0.98
    assert valid_metadata.page_number == 5


def test_document_metadata_ocr_confidence_bounds() -> None:
    now = datetime.now(UTC)
    # Test ocr_confidence > 1.0
    with pytest.raises(ValidationError) as exc_info:
        DocumentMetadata(
            source="test.pdf",
            page_number=1,
            ingestion_timestamp=now,
            ocr_confidence=1.05,
            tenant_id="t1",
            version_id="v1",
        )
    assert "ocr_confidence" in str(exc_info.value)

    # Test ocr_confidence < 0.0
    with pytest.raises(ValidationError) as exc_info:
        DocumentMetadata(
            source="test.pdf",
            page_number=1,
            ingestion_timestamp=now,
            ocr_confidence=-0.1,
            tenant_id="t1",
            version_id="v1",
        )
    assert "ocr_confidence" in str(exc_info.value)


def test_document_metadata_invalid_page_number() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError) as exc_info:
        DocumentMetadata(
            source="test.pdf",
            page_number=0,  # Must be >= 1
            ingestion_timestamp=now,
            ocr_confidence=0.9,
            tenant_id="t1",
            version_id="v1",
        )
    assert "page_number" in str(exc_info.value)


# ==============================================================================
# DOCUMENT & CHUNK TESTS
# ==============================================================================

def test_document_valid(valid_metadata: DocumentMetadata) -> None:
    doc = Document(
        id="doc_100",
        content="Full raw content of report v1.",
        metadata=valid_metadata,
    )
    assert doc.id == "doc_100"
    assert doc.metadata.tenant_id == "tenant_abc"


def test_document_chunk_missing_fields(valid_metadata: DocumentMetadata) -> None:
    with pytest.raises(ValidationError) as exc_info:
        DocumentChunk(
            id="chunk_001",
            # missing document_id
            content="Snippet text.",
            metadata=valid_metadata,
        )
    assert "document_id" in str(exc_info.value)


def test_document_chunk_negative_index(valid_metadata: DocumentMetadata) -> None:
    with pytest.raises(ValidationError) as exc_info:
        DocumentChunk(
            id="chunk_001",
            document_id="doc_100",
            content="Snippet text.",
            chunk_index=-1,
            metadata=valid_metadata,
        )
    assert "chunk_index" in str(exc_info.value)


# ==============================================================================
# RETRIEVAL RESULT TESTS
# ==============================================================================

def test_retrieval_result_valid(valid_chunk: DocumentChunk) -> None:
    res = RetrievalResult(
        chunk=valid_chunk,
        vector_score=0.88,
        bm25_score=12.5,
        fused_score=0.91,
        rerank_score=0.95,
    )
    assert res.chunk.id == "chunk_001"
    assert res.rerank_score == 0.95


def test_retrieval_result_missing_scores(valid_chunk: DocumentChunk) -> None:
    with pytest.raises(ValidationError) as exc_info:
        RetrievalResult(
            chunk=valid_chunk,
            vector_score=0.88,
            bm25_score=12.5,
            # missing fused_score
        )
    assert "fused_score" in str(exc_info.value)


# ==============================================================================
# CORRECTION VERDICT & ENUM TESTS
# ==============================================================================

def test_correction_verdict_valid() -> None:
    verdict = CorrectionVerdict(
        signal_type=SignalType.GROUNDEDNESS,
        verdict=True,
        confidence=0.92,
        reasoning="All factual claims are explicitly supported by chunk_001.",
    )
    assert verdict.signal_type == SignalType.GROUNDEDNESS
    assert verdict.signal_type.value == "GROUNDEDNESS"


def test_correction_verdict_invalid_enum() -> None:
    with pytest.raises(ValidationError) as exc_info:
        CorrectionVerdict(
            signal_type="INVALID_SIGNAL",
            verdict=False,
            confidence=0.5,
            reasoning="Unsupported claims detected.",
        )
    assert "GROUNDEDNESS" in str(exc_info.value)


def test_correction_verdict_confidence_bounds() -> None:
    with pytest.raises(ValidationError) as exc_info:
        CorrectionVerdict(
            signal_type=SignalType.CONTRADICTION,
            verdict=False,
            confidence=1.2,
            reasoning="Out of bounds confidence test.",
        )
    assert "confidence" in str(exc_info.value)


# ==============================================================================
# QUERY REQUEST & RESPONSE TESTS
# ==============================================================================

def test_query_request_valid(valid_tenant_context: TenantContext) -> None:
    req = QueryRequest(
        query="Explain the self-correction mechanism in Compass RAG.",
        tenant_context=valid_tenant_context,
        top_k=5,
        metadata_filter={"source": "report_v1.pdf", "page_number": 5},
    )
    assert req.top_k == 5
    assert req.metadata_filter["source"] == "report_v1.pdf"


def test_query_response_valid() -> None:
    cit = Citation(
        chunk_id="chunk_001",
        document_id="doc_100",
        source="report_v1.pdf",
        page_number=5,
        quote_snippet="multi-stage evaluation",
    )
    resp = QueryResponse(
        answer="Compass RAG uses multi-stage evaluation for self-correction.",
        confidence_status=ConfidenceStatus.VERIFIED,
        confidence_score=0.94,
        citations=[cit],
        verdicts=[],
    )
    assert resp.confidence_status == ConfidenceStatus.VERIFIED
    assert resp.citations[0].chunk_id == "chunk_001"


def test_query_response_invalid_confidence_status() -> None:
    with pytest.raises(ValidationError) as exc_info:
        QueryResponse(
            answer="Test answer.",
            confidence_status="SOME_UNKNOWN_STATUS",
            confidence_score=0.8,
        )
    assert "VERIFIED" in str(exc_info.value)
