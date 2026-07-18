"""
Postgres Chunk Metadata Store for Retrieval Service.
Manages transactional storage and lookup of DocumentChunk records and embedding version lineage.
"""

import logging
from functools import lru_cache

from shared.models.common import DocumentChunk
from sqlalchemy import select
from sqlalchemy.orm import Session

from services.retrieval.app.db.models import DocumentChunkRecord
from services.retrieval.app.db.session import get_sync_session

logger = logging.getLogger(__name__)


class PostgresChunkStore:
    """
    Helper service for storing and querying chunk metadata records in PostgreSQL.
    """

    def save_chunk_record(
        self,
        chunk: DocumentChunk,
        embedding_provider: str,
        embedding_model: str,
        session: Session | None = None,
    ) -> bool:
        """
        Persists a DocumentChunk metadata record along with embedding lineage into Postgres.
        """
        close_session = False
        if session is None:
            session = get_sync_session()
            close_session = True

        try:
            record = DocumentChunkRecord(
                id=chunk.id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                source=chunk.metadata.source,
                page_number=chunk.metadata.page_number,
                tenant_id=chunk.metadata.tenant_id,
                version_id=chunk.metadata.version_id,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                created_at=chunk.metadata.ingestion_timestamp,
            )
            session.merge(record)
            session.commit()
            logger.debug("Saved chunk metadata record %s to Postgres", chunk.id)
            return True
        except Exception as exc:
            session.rollback()
            logger.error("Error saving chunk record %s: %s", chunk.id, exc)
            return False
        finally:
            if close_session:
                session.close()

    def get_chunk_record(
        self, chunk_id: str, session: Session | None = None
    ) -> DocumentChunkRecord | None:
        """
        Retrieves a DocumentChunkRecord by its ID.
        """
        close_session = False
        if session is None:
            session = get_sync_session()
            close_session = True

        try:
            stmt = select(DocumentChunkRecord).where(
                DocumentChunkRecord.id == chunk_id
            )
            return session.execute(stmt).scalar_one_or_none()
        finally:
            if close_session:
                session.close()

    def list_chunks_by_document(
        self,
        document_id: str,
        tenant_id: str,
        session: Session | None = None,
    ) -> list[DocumentChunkRecord]:
        """
        Returns all chunk records for a given document and tenant, ordered by index.
        """
        close_session = False
        if session is None:
            session = get_sync_session()
            close_session = True

        try:
            stmt = (
                select(DocumentChunkRecord)
                .where(
                    DocumentChunkRecord.document_id == document_id,
                    DocumentChunkRecord.tenant_id == tenant_id,
                )
                .order_by(DocumentChunkRecord.chunk_index)
            )
            return list(session.execute(stmt).scalars().all())
        finally:
            if close_session:
                session.close()


@lru_cache
def get_postgres_store() -> PostgresChunkStore:
    """
    Returns cached singleton instance of PostgresChunkStore.
    """
    return PostgresChunkStore()
