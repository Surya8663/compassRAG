"""
SQLAlchemy ORM models for the API Gateway (`services/api-gateway`).
Stores role-based access control (`DocumentAccessPermission`) records and queries job status.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for API Gateway declarative ORM models."""
    pass


def _utc_now() -> datetime:
    return datetime.now(UTC)


class DocumentAccessPermission(Base):
    """
    Document-level RBAC permissions table.
    Enforces role (`required_role`) and optional user (`allowed_user_id`) restrictions
    per document within a tenant namespace (`tenant_id`).
    """
    __tablename__ = "document_permissions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    document_id: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    required_role: Mapped[str | None] = mapped_column(String(100), nullable=True)
    allowed_user_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utc_now, nullable=False)


class DocumentBatchRecord(Base):
    """
    Reflection of `document_batches` table for Gateway job status polling.
    """
    __tablename__ = "document_batches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    expected_pages: Mapped[int] = mapped_column(Integer, nullable=False)
    received_pages: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    updated_at: Mapped[datetime] = mapped_column(nullable=False)
