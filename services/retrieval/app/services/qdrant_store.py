"""
Qdrant Vector Storage Layer for Retrieval Service.
Provides dense vector indexing and retrieval with strict payload-based tenant isolation.
"""

import logging
import uuid
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from shared.config import get_settings
from shared.models.common import DocumentChunk

logger = logging.getLogger(__name__)


class QdrantStoreService:
    """
    Service for managing Qdrant vector collections and performing searches.
    Enforces mandatory payload filtering by `tenant_id` to guarantee isolation.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = QdrantClient(url=self.settings.QDRANT_URL)

    def ensure_collection(
        self, collection_name: str = "compass_rag_chunks", dimension: int = 384
    ) -> None:
        """
        Creates vector collection and payload keyword index for `tenant_id` if needed.
        """
        try:
            if not self.client.collection_exists(collection_name=collection_name):
                logger.info(
                    "Creating Qdrant collection '%s' with dimension %d...",
                    collection_name,
                    dimension,
                )
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=dimension, distance=models.Distance.COSINE
                    ),
                )
            # Ensure index on tenant_id for high performance filtering
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="tenant_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            logger.warning(
                "Notice during Qdrant ensure_collection '%s': %s", collection_name, exc
            )

    def _to_point_id(self, chunk_id: str) -> str:
        """
        Converts any arbitrary chunk ID string into a deterministic UUID string for Qdrant.
        If chunk_id is already a valid UUID string, returns it directly.
        """
        try:
            return str(uuid.UUID(chunk_id))
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def upsert_chunk(
        self,
        chunk: DocumentChunk,
        vector: list[float],
        collection_name: str = "compass_rag_chunks",
    ) -> bool:
        """
        Upserts a DocumentChunk and its dense vector representation into Qdrant.
        """
        self.ensure_collection(collection_name=collection_name, dimension=len(vector))
        point_id = self._to_point_id(chunk.id)
        payload: dict[str, Any] = {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "source": chunk.metadata.source,
            "page_number": chunk.metadata.page_number,
            "tenant_id": chunk.metadata.tenant_id,
            "version_id": chunk.metadata.version_id,
            "ocr_confidence": chunk.metadata.ocr_confidence,
        }

        try:
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
            logger.debug("Successfully upserted chunk %s into Qdrant", chunk.id)
            return True
        except Exception as exc:
            logger.error("Failed to upsert chunk %s into Qdrant: %s", chunk.id, exc)
            return False

    def search(
        self,
        query_vector: list[float],
        tenant_id: str,
        top_k: int = 10,
        collection_name: str = "compass_rag_chunks",
    ) -> list[dict[str, Any]]:
        """
        Performs vector similarity search restricted exclusively to points matching `tenant_id`.
        """
        if not tenant_id:
            raise ValueError("tenant_id is mandatory for Qdrant vector search.")

        self.ensure_collection(collection_name=collection_name, dimension=len(query_vector))

        tenant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id", match=models.MatchValue(value=tenant_id)
                )
            ]
        )

        try:
            points = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=tenant_filter,
                limit=top_k,
            )
            results: list[dict[str, Any]] = []
            for point in points:
                payload = point.payload or {}
                results.append(
                    {
                        "chunk_id": payload.get("chunk_id"),
                        "score": point.score,
                        "payload": payload,
                    }
                )
            return results
        except Exception as exc:
            logger.error("Qdrant search error for tenant '%s': %s", tenant_id, exc)
            raise RuntimeError(f"Qdrant vector search failed: {exc}") from exc


@lru_cache
def get_qdrant_store() -> QdrantStoreService:
    """
    Returns cached singleton instance of QdrantStoreService.
    """
    return QdrantStoreService()
