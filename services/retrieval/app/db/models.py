"""
Relational database models for Retrieval Service.
Stores DocumentChunk records and associated embedding metadata.
"""

from datetime import UTC, datetime

from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """
    Base class for Retrieval Service SQLAlchemy declarative ORM models.
    """
    pass


def _utc_now() -> datetime:
    return datetime.now(UTC)


class DocumentChunkRecord(Base):
    """
    Stores chunk text content and metadata along with the provider/model version
    used to generate its dense vector representation.
    """
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(255), nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    version_id: Mapped[str] = mapped_column(String(50), nullable=False)
    embedding_provider: Mapped[str] = mapped_column(String(50), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=_utc_now, nullable=False)
