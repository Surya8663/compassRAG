"""
Integration test verifying mandatory PII redaction enforcement in worker task before DB/chunking.
"""

from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from services.ingestion.app.db.models import Base, DocumentBatch, DocumentPage
from services.ingestion.app.workers.tasks import process_document_page
from tests.fixtures.make_test_pdfs import create_native_pdf


@pytest.fixture
def test_db_session_pii(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Generator[Session, None, None]:
    """
    Sets up an isolated SQLite database session for verifying PII pipeline dependency enforcement.
    """
    db_path = tmp_path / "test_pii_pipeline.db"
    dsn = f"sqlite:///{db_path}"

    import app.db.session as db_sess

    engine = create_engine(dsn)
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    monkeypatch.setattr(db_sess, "_engine", engine)
    monkeypatch.setattr(db_sess, "_sync_session_factory", factory)
    monkeypatch.setattr(db_sess, "get_engine", lambda: engine)
    monkeypatch.setattr(db_sess, "get_session_factory", lambda: factory)

    session = factory()
    try:
        yield session
    finally:
        session.close()


def test_pipeline_dependency_enforcement_before_chunking(
    test_db_session_pii: Session, tmp_path: Path
) -> None:
    """
    Verifies critical security constraint: when process_document_page runs on a document
    containing sensitive PII (`999-01-1234` and `john.doe@example.com`), redaction happens
    BEFORE any downstream call, ensuring:
    1. Stored `extracted_text` in Postgres (`DocumentPage`) contains ONLY redacted text.
    2. Generated chunks (`chunks` inside task result) contain ONLY redacted text.
    3. Zero unredacted PII can be accessed from the DB or chunking output.
    """
    sensitive_content = (
        "CONFIDENTIAL MEMO. Employee John Doe (email: john.doe@example.com) has Social "
        "Security Number 999-01-1234. Please ensure this record is indexed safely."
    )
    pdf_path = tmp_path / "sensitive_memo.pdf"
    create_native_pdf(pdf_path, text=sensitive_content)

    batch_id = "batch_pii_test_001"
    doc_id = "doc_pii_100"

    batch = DocumentBatch(
        id=batch_id,
        document_id=doc_id,
        filename="sensitive_memo.pdf",
        expected_pages=1,
        received_pages=0,
        status="PROCESSING",
    )
    page = DocumentPage(
        id="page_pii_001",
        batch_id=batch_id,
        page_number=1,
        status="PROCESSING",
    )
    test_db_session_pii.add(batch)
    test_db_session_pii.add(page)
    test_db_session_pii.commit()

    # Execute worker task synchronously
    result = process_document_page(batch_id, 1, pdf_path, doc_id)

    assert result["status"] == "EXTRACTED"
    assert result["chunks_indexed"] > 0

    # Verify what was stored inside Postgres
    test_db_session_pii.expire_all()
    db_page = (
        test_db_session_pii.query(DocumentPage)
        .filter(DocumentPage.id == "page_pii_001")
        .first()
    )
    assert db_page is not None
    assert db_page.extracted_text is not None

    # Assert zero unredacted PII reached Postgres extracted_text column
    assert "John Doe" not in db_page.extracted_text
    assert "john.doe@example.com" not in db_page.extracted_text
    assert "999-01-1234" not in db_page.extracted_text

    # Assert Presidio tags are in Postgres
    assert "<PERSON>" in db_page.extracted_text
    assert "<EMAIL_ADDRESS>" in db_page.extracted_text
    assert "<US_SSN>" in db_page.extracted_text

    # Assert zero unredacted PII reached any downstream DocumentChunk
    chunks_data = result["chunks"]
    assert len(chunks_data) > 0
    for chunk_dict in chunks_data:
        content = chunk_dict["content"]
        assert "John Doe" not in content
        assert "john.doe@example.com" not in content
        assert "999-01-1234" not in content
        assert "<PERSON>" in content or "<EMAIL_ADDRESS>" in content or "<US_SSN>" in content
