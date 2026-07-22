"""
Elasticsearch BM25 Keyword Storage Layer for Retrieval Service.
Provides full-text BM25 search over chunk content with mandatory tenant_id filtering.
"""

import logging
from functools import lru_cache
from typing import Any

from elasticsearch import Elasticsearch
import elasticsearch._sync.client._base as _es_base
_es_base._COMPAT_MIMETYPE_TEMPLATE = "application/vnd.elasticsearch+%s; compatible-with=8"
_es_base._COMPAT_MIMETYPE_SUB = _es_base._COMPAT_MIMETYPE_TEMPLATE % (r"\g<1>",)

from shared.config import get_settings
from shared.models.common import DocumentChunk

logger = logging.getLogger(__name__)


class ElasticsearchStoreService:
    """
    Service for managing Elasticsearch indexes and performing BM25 searches.
    Enforces mandatory `term` filtering on `tenant_id` to guarantee isolation.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = Elasticsearch(self.settings.ELASTICSEARCH_URL)

    def ensure_index(self, index_name: str = "compass_rag_chunks") -> None:
        """
        Creates index with explicit mappings for BM25 search and keyword filtering.
        """
        try:
            if not self.client.indices.exists(index=index_name):
                logger.info("Creating Elasticsearch index '%s'...", index_name)
                mappings: dict[str, Any] = {
                    "properties": {
                        "chunk_id": {"type": "keyword"},
                        "document_id": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                        "content": {"type": "text", "analyzer": "standard"},
                        "source": {"type": "keyword"},
                        "page_number": {"type": "integer"},
                        "tenant_id": {"type": "keyword"},
                        "version_id": {"type": "keyword"},
                        "ocr_confidence": {"type": "float"},
                    }
                }
                self.client.indices.create(index=index_name, mappings=mappings)
        except Exception as exc:
            logger.warning(
                "Notice during Elasticsearch ensure_index '%s': %s", index_name, exc
            )

    def index_chunk(
        self,
        chunk: DocumentChunk,
        index_name: str = "compass_rag_chunks",
        refresh: bool = True,
    ) -> bool:
        """
        Indexes a DocumentChunk into Elasticsearch for keyword search.
        If refresh is True, refreshes the index so the document is searchable without delay.
        """
        self.ensure_index(index_name)
        doc: dict[str, Any] = {
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
            self.client.index(
                index=index_name, id=chunk.id, document=doc, refresh=refresh
            )
            logger.debug("Successfully indexed chunk %s in Elasticsearch", chunk.id)
            return True
        except Exception as exc:
            logger.error("Failed to index chunk %s in Elasticsearch: %s", chunk.id, exc)
            return False

    def search_keywords(
        self,
        query_text: str,
        tenant_id: str,
        top_k: int = 10,
        index_name: str = "compass_rag_chunks",
    ) -> list[dict[str, Any]]:
        """
        Performs BM25 keyword search strictly filtered by `tenant_id`.
        """
        es_query: dict[str, Any] = {
            "bool": {
                "must": [{"match": {"content": query_text}}],
                "filter": [{"term": {"tenant_id": tenant_id}}],
            }
        }
        try:
            self.ensure_index(index_name)
            resp = self.client.search(index=index_name, query=es_query, size=top_k)
            hits = resp.get("hits", {}).get("hits", [])
            results: list[dict[str, Any]] = []
            for hit in hits:
                source = hit.get("_source", {})
                results.append(
                    {
                        "chunk_id": hit.get("_id") or source.get("chunk_id"),
                        "score": float(hit.get("_score") or 0.0),
                        "payload": source,
                    }
                )
            return results
        except Exception as exc:
            logger.error("Elasticsearch search error for tenant '%s': %s", tenant_id, exc)
            return []


@lru_cache
def get_es_store() -> ElasticsearchStoreService:
    """
    Returns cached singleton instance of ElasticsearchStoreService.
    """
    return ElasticsearchStoreService()
