import uuid
from datetime import UTC, datetime

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """
    Base class for SQLAlchemy declarative ORM models.
    """
    pass


def _utc_now() -> datetime:
    return datetime.now(UTC)


class DocumentBatch(Base):
    """
    Tracks multi-page document ingestion state across asynchronous Celery worker tasks.
    Enables detection and resumption of partial uploads.
    """
    __tablename__ = "document_batches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    expected_pages: Mapped[int] = mapped_column(Integer, nullable=False)
    received_pages: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="PROCESSING", nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=_utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        default=_utc_now, onupdate=_utc_now, nullable=False
    )

    pages: Mapped[list["DocumentPage"]] = relationship(
        "DocumentPage", back_populates="batch", cascade="all, delete-orphan"
    )
    manual_reviews: Mapped[list["ManualReviewQueue"]] = relationship(
        "ManualReviewQueue", back_populates="batch", cascade="all, delete-orphan"
    )


class DocumentPage(Base):
    """
    Tracks individual page status, classification, OCR confidence, and extracted text.
    """
    __tablename__ = "document_pages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    batch_id: Mapped[str] = mapped_column(
        ForeignKey("document_batches.id", ondelete="CASCADE"), index=True, nullable=False
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="PROCESSING", nullable=False)
    classification: Mapped[str | None] = mapped_column(String(50), nullable=True)
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    processed_at: Mapped[datetime | None] = mapped_column(nullable=True)

    batch: Mapped["DocumentBatch"] = relationship("DocumentBatch", back_populates="pages")


class ManualReviewQueue(Base):
    """
    Stores pages where OCR confidence fell below the configurable quality threshold (e.g., < 0.85).
    Status defaults to PENDING_REVIEW for manual intervention.
    """
    __tablename__ = "manual_review_queue"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    batch_id: Mapped[str] = mapped_column(
        ForeignKey("document_batches.id", ondelete="CASCADE"), index=True, nullable=False
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    ocr_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="PENDING_REVIEW", nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=_utc_now, nullable=False)

    batch: Mapped["DocumentBatch"] = relationship("DocumentBatch", back_populates="manual_reviews")
