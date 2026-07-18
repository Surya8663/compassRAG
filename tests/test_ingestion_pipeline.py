import os
from collections.abc import Generator
from pathlib import Path

import fitz
import pytest
from shared.config import get_settings
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from services.ingestion.app.db.models import (
    Base,
    DocumentBatch,
    DocumentPage,
    ManualReviewQueue,
)
from services.ingestion.app.pipeline.classifier import classify_and_extract_page
from services.ingestion.app.pipeline.ocr import process_page_ocr
from services.ingestion.app.workers.tasks import process_document_page
from tests.fixtures.make_test_pdfs import create_native_pdf, create_scanned_pdf


@pytest.fixture
def test_db_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Generator[Session, None, None]:
    """
    Sets up an isolated SQLite database session for Batch State Manager verification.
    """
    db_path = tmp_path / "test_ingestion.db"
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


def test_document_classifier_native_vs_scanned(tmp_path: Path) -> None:
    """
    Verifies that PyMuPDF classifier detects extractable text vs image-only pages.
    """
    native_path = create_native_pdf(tmp_path / "native.pdf")
    scanned_path = create_scanned_pdf(tmp_path / "scanned.pdf")

    with fitz.open(native_path) as native_doc:
        page_native = native_doc.load_page(0)
        classification, text = classify_and_extract_page(page_native)
        assert classification == "NATIVE_TEXT"
        assert "Compass RAG Native Text Page 1" in text

    with fitz.open(scanned_path) as scanned_doc:
        page_scanned = scanned_doc.load_page(0)
        classification, text = classify_and_extract_page(page_scanned)
        assert classification == "SCANNED_IMAGE"
        assert text == ""


def test_real_tesseract_ocr_confidence_scoring(tmp_path: Path) -> None:
    """
    Verifies that real Tesseract binary is invoked on a scanned PDF pixmap,
    returning non-hardcoded float confidence scores and real extracted text.
    """
    settings = get_settings()
    if not os.path.exists(settings.TESSERACT_CMD):
        pytest.skip("Real Tesseract binary not found at configured TESSERACT_CMD path")

    scanned_path = create_scanned_pdf(tmp_path / "scanned_ocr.pdf")

    with fitz.open(scanned_path) as doc:
        page = doc.load_page(0)
        text, confidence = process_page_ocr(page)

        # Assert confidence is computed as a real float between 0 and 1
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        # Assert text extracted from the bitmap contains the baked-in string
        assert "COMPASS" in text.upper() or "RAG" in text.upper() or "OCR" in text.upper()


def test_batch_state_manager_and_manual_review_routing(
    test_db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Verifies full Batch State Manager tracking (`received_pages` vs `expected_pages`)
    and quality threshold routing to ManualReviewQueue when OCR confidence is low.
    """
    settings = get_settings()
    if not os.path.exists(settings.TESSERACT_CMD):
        pytest.skip("Real Tesseract binary not found at configured TESSERACT_CMD path")

    # Force threshold higher than typical OCR confidence to test routing to review queue
    monkeypatch.setattr(settings, "OCR_CONFIDENCE_THRESHOLD", 0.999)

    scanned_path = create_scanned_pdf(tmp_path / "batch_scanned.pdf")
    batch_id = "batch_test_001"
    doc_id = "doc_test_100"

    # Seed initial Batch and Page states
    batch = DocumentBatch(
        id=batch_id,
        document_id=doc_id,
        filename="batch_scanned.pdf",
        expected_pages=1,
        received_pages=0,
        status="PROCESSING",
    )
    page = DocumentPage(
        id="page_test_001",
        batch_id=batch_id,
        page_number=1,
        status="PROCESSING",
    )
    test_db_session.add(batch)
    test_db_session.add(page)
    test_db_session.commit()

    # Run worker task synchronously
    result = process_document_page(batch_id, 1, scanned_path, doc_id)

    # Verify task result
    assert result["batch_id"] == batch_id
    assert result["classification"] == "SCANNED_IMAGE"
    assert result["status"] == "MANUAL_REVIEW"

    # Verify Postgres updates
    test_db_session.expire_all()
    updated_batch = (
        test_db_session.query(DocumentBatch).filter(DocumentBatch.id == batch_id).first()
    )
    assert updated_batch is not None
    assert updated_batch.received_pages == 1
    assert updated_batch.status == "REQUIRES_REVIEW"

    updated_page = (
        test_db_session.query(DocumentPage)
        .filter(DocumentPage.id == "page_test_001")
        .first()
    )
    assert updated_page is not None
    assert updated_page.status == "MANUAL_REVIEW"
    assert updated_page.ocr_confidence is not None
    assert updated_page.ocr_confidence < 0.999

    review_item = (
        test_db_session.query(ManualReviewQueue)
        .filter(ManualReviewQueue.batch_id == batch_id)
        .first()
    )
    assert review_item is not None
    assert review_item.page_number == 1
    assert review_item.status == "PENDING_REVIEW"
    assert "below threshold" in review_item.reason
