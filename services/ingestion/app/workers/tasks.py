from datetime import UTC, datetime
from typing import Any

import fitz
from shared.config import get_settings

from app.db.models import DocumentBatch, DocumentPage, ManualReviewQueue
from app.db.session import get_sync_session
from app.pipeline.classifier import classify_and_extract_page
from app.pipeline.ocr import process_page_ocr
from app.workers.celery_app import celery_app


def _utc_now() -> datetime:
    return datetime.now(UTC)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=5)  # type: ignore[untyped-decorator]
def process_document_page(
    self: Any, batch_id: str, page_number: int, file_path: str, document_id: str
) -> dict[str, Any]:
    """
    Celery task to process a single PDF page asynchronously:
    1. Classifies as NATIVE_TEXT vs SCANNED_IMAGE.
    2. Runs PyMuPDF text extraction or real Tesseract OCR.
    3. Checks OCR confidence against threshold and routes to ManualReviewQueue if low.
    4. Updates DocumentPage and increments DocumentBatch.received_pages in Postgres.
    """
    settings = get_settings()
    session = get_sync_session()

    try:
        batch = session.query(DocumentBatch).filter(DocumentBatch.id == batch_id).first()
        page = (
            session.query(DocumentPage)
            .filter(DocumentPage.batch_id == batch_id, DocumentPage.page_number == page_number)
            .first()
        )

        if not batch or not page:
            return {"status": "error", "message": "Batch or page not found in Postgres"}

        if page.status in ("EXTRACTED", "MANUAL_REVIEW"):
            return {"status": "skipped", "message": "Page already processed"}

        try:
            with fitz.open(file_path) as doc:
                if page_number < 1 or page_number > len(doc):
                    raise ValueError(f"Page number {page_number} out of range (1..{len(doc)})")
                page_obj = doc.load_page(page_number - 1)
                classification, extracted_text = classify_and_extract_page(page_obj)

                if classification == "SCANNED_IMAGE":
                    extracted_text, ocr_confidence = process_page_ocr(page_obj)
                    page.classification = classification
                    page.ocr_confidence = ocr_confidence
                    page.extracted_text = extracted_text
                    page.processed_at = _utc_now()

                    if ocr_confidence < settings.OCR_CONFIDENCE_THRESHOLD:
                        page.status = "MANUAL_REVIEW"
                        review_entry = ManualReviewQueue(
                            document_id=document_id,
                            batch_id=batch_id,
                            page_number=page_number,
                            ocr_confidence=ocr_confidence,
                            status="PENDING_REVIEW",
                            reason=(
                                f"OCR confidence {ocr_confidence:.2f} below threshold "
                                f"{settings.OCR_CONFIDENCE_THRESHOLD}"
                            ),
                        )
                        session.add(review_entry)
                    else:
                        page.status = "EXTRACTED"
                else:
                    page.classification = classification
                    page.ocr_confidence = 1.0
                    page.extracted_text = extracted_text
                    page.status = "EXTRACTED"
                    page.processed_at = _utc_now()

        except Exception as exc:
            page.status = "FAILED"
            page.error_message = str(exc)
            page.processed_at = _utc_now()

        # Flush pending ORM changes before querying aggregate page counts
        session.flush()

        # Atomically increment received_pages and update batch status if complete
        batch.received_pages += 1
        if batch.received_pages >= batch.expected_pages:
            failed_count = (
                session.query(DocumentPage)
                .filter(DocumentPage.batch_id == batch_id, DocumentPage.status == "FAILED")
                .count()
            )
            review_count = (
                session.query(DocumentPage)
                .filter(DocumentPage.batch_id == batch_id, DocumentPage.status == "MANUAL_REVIEW")
                .count()
            )

            if failed_count > 0:
                batch.status = "FAILED"
            elif review_count > 0:
                batch.status = "REQUIRES_REVIEW"
            else:
                batch.status = "COMPLETED"

        session.commit()
        return {
            "status": page.status,
            "batch_id": batch_id,
            "page_number": page_number,
            "classification": page.classification,
            "ocr_confidence": page.ocr_confidence,
        }

    except Exception as e:
        session.rollback()
        raise self.retry(exc=e) from e
    finally:
        session.close()
