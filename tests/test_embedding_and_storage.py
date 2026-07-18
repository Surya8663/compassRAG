"""
Integration and security verification tests for Phase 4: Embedding Service & Dual Storage Layer.
Verifies real model embeddings, Redis cache-aside behavior, Qdrant/Elasticsearch dual indexing,
and strict multi-tenant data isolation (`tenant_id`) across vector and keyword retrieval.
"""

import uuid
from datetime import UTC, datetime

import pytest
from shared.models.common import DocumentChunk, DocumentMetadata

from services.retrieval.app.db.models import Base
from services.retrieval.app.db.session import get_engine, init_db
from services.retrieval.app.services.embedder import get_embedding_service
from services.retrieval.app.services.es_store import get_es_store
from services.retrieval.app.services.postgres_store import get_postgres_store
from services.retrieval.app.services.qdrant_store import get_qdrant_store


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _make_test_chunk(
    chunk_id: str,
    doc_id: str,
    content: str,
    tenant_id: str,
    idx: int = 0,
) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id=doc_id,
        chunk_index=idx,
        content=content,
        metadata=DocumentMetadata(
            source=f"{doc_id}.pdf",
            page_number=1,
            ingestion_timestamp=_utc_now(),
            tenant_id=tenant_id,
            version_id="v1.0",
            ocr_confidence=0.99,
        ),
    )


@pytest.fixture(scope="module", autouse=True)
def setup_retrieval_db() -> None:
    """
    Initializes database schema for retrieval service chunk metadata testing.
    """
    try:
        init_db()
    except Exception:
        # If Postgres is unreachable or sqlite fall-back used
        engine = get_engine()
        Base.metadata.create_all(bind=engine)


def test_redis_embedding_cache_aside() -> None:
    """
    Verifies that when EmbeddingService embeds text, it stores the vector in Redis.
    Subsequent calls for identical text hit Redis and return the vector without re-computing.
    """
    embedder = get_embedding_service()
    unique_phrase = f"Compass RAG Redis verification phrase {uuid.uuid4()}"

    # First embed -> cache miss, computes and writes to Redis
    vec1 = embedder.embed_text(unique_phrase)
    assert isinstance(vec1, list)
    assert len(vec1) == embedder.dimension

    # Check Redis directly
    cache_key = embedder._compute_cache_key(unique_phrase)
    cached_raw = embedder.redis_client.get(cache_key)
    assert cached_raw is not None

    # Second embed -> cache hit from Redis
    vec2 = embedder.embed_text(unique_phrase)
    assert vec1 == vec2


def test_qdrant_tenant_isolation_enforcement() -> None:
    """
    Proves critical security requirement: Tenant A CANNOT retrieve Tenant B vectors,
    even when Tenant A queries with the exact vector representation of Tenant B chunk.
    """
    embedder = get_embedding_service()
    qdrant = get_qdrant_store()

    test_collection = "test_qdrant_tenant_isolation"
    qdrant.ensure_collection(
        collection_name=test_collection, dimension=embedder.dimension
    )

    chunk_alpha = _make_test_chunk(
        chunk_id=f"chunk_alpha_{uuid.uuid4()}",
        doc_id="doc_alpha_1",
        content="Alpha tenant secret document about internal quantum algorithms.",
        tenant_id="tenant_alpha",
    )
    chunk_beta = _make_test_chunk(
        chunk_id=f"chunk_beta_{uuid.uuid4()}",
        doc_id="doc_beta_1",
        content="Beta tenant confidential report on merger negotiations.",
        tenant_id="tenant_beta",
    )

    vec_alpha = embedder.embed_text(chunk_alpha.content)
    vec_beta = embedder.embed_text(chunk_beta.content)

    assert qdrant.upsert_chunk(chunk_alpha, vec_alpha, collection_name=test_collection)
    assert qdrant.upsert_chunk(chunk_beta, vec_beta, collection_name=test_collection)

    # Tenant Alpha queries with vec_beta (exact vector of Beta's confidential report)
    results_alpha = qdrant.search(
        query_vector=vec_beta,
        tenant_id="tenant_alpha",
        top_k=10,
        collection_name=test_collection,
    )

    retrieved_ids = [item["chunk_id"] for item in results_alpha]
    assert (
        chunk_beta.id not in retrieved_ids
    ), "Security Violation: Tenant Alpha retrieved Tenant Beta vector!"
    for res in results_alpha:
        assert res["payload"]["tenant_id"] == "tenant_alpha"


def test_elasticsearch_bm25_tenant_isolation() -> None:
    """
    Proves critical security requirement for keyword search: BM25 queries with Tenant A
    cannot return documents indexed by Tenant B, even if keyword terms match exactly.
    """
    es = get_es_store()
    test_index = "test_es_tenant_isolation"
    es.ensure_index(index_name=test_index)

    chunk_A = _make_test_chunk(
        chunk_id=f"es_alpha_{uuid.uuid4()}",
        doc_id="doc_es_1",
        content="Secret quarterly financial report containing sensitive revenue metrics.",
        tenant_id="tenant_alpha",
    )
    chunk_B = _make_test_chunk(
        chunk_id=f"es_beta_{uuid.uuid4()}",
        doc_id="doc_es_2",
        content="Secret quarterly financial report containing competitor analysis data.",
        tenant_id="tenant_beta",
    )

    assert es.index_chunk(chunk_A, index_name=test_index, refresh=True)
    assert es.index_chunk(chunk_B, index_name=test_index, refresh=True)

    # Search exact keyword phrase "quarterly financial report" as tenant_alpha
    results = es.search_keywords(
        query_text="quarterly financial report",
        tenant_id="tenant_alpha",
        top_k=10,
        index_name=test_index,
    )

    retrieved_ids = [item["chunk_id"] for item in results]
    assert (
        chunk_A.id in retrieved_ids
    ), "Tenant Alpha failed to retrieve its own keyword match"
    assert (
        chunk_B.id not in retrieved_ids
    ), "Security Violation: Tenant Alpha retrieved Tenant Beta document!"
    for res in results:
        assert res["payload"]["tenant_id"] == "tenant_alpha"


def test_full_dual_indexing_and_retrieval_flow() -> None:
    """
    End-to-end integration test verifying chunk ingestion across all three storage layers:
    1. Postgres relational metadata (`document_chunks` table).
    2. Qdrant dense vector store (`test_dual_qdrant`).
    3. Elasticsearch BM25 keyword store (`test_dual_es`).
    """
    embedder = get_embedding_service()
    qdrant = get_qdrant_store()
    es = get_es_store()
    pg_store = get_postgres_store()

    coll_name = "test_dual_qdrant"
    idx_name = "test_dual_es"
    qdrant.ensure_collection(collection_name=coll_name, dimension=embedder.dimension)
    es.ensure_index(index_name=idx_name)

    chunk_id = f"dual_chunk_{uuid.uuid4()}"
    doc_id = "dual_doc_100"
    tenant_id = "tenant_dual"
    content = (
        "Compass RAG dual retrieval test verifying dense vector and sparse keyword fusion."
    )

    chunk = _make_test_chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        tenant_id=tenant_id,
        idx=5,
    )

    # 1. Generate real vector (utilizing cache-aside)
    vec = embedder.embed_text(chunk.content)

    # 2. Store in Postgres metadata store
    assert pg_store.save_chunk_record(
        chunk=chunk,
        embedding_provider=embedder.provider,
        embedding_model=embedder.model_name,
    )

    # 3. Upsert into Qdrant
    assert qdrant.upsert_chunk(chunk, vec, collection_name=coll_name)

    # 4. Index into Elasticsearch
    assert es.index_chunk(chunk, index_name=idx_name, refresh=True)

    # Verify retrieval via Qdrant dense search
    dense_results = qdrant.search(
        query_vector=vec,
        tenant_id=tenant_id,
        top_k=5,
        collection_name=coll_name,
    )
    assert any(
        item["chunk_id"] == chunk_id for item in dense_results
    ), "Dense vector retrieval failed to locate ingested chunk"

    # Verify retrieval via Elasticsearch BM25 keyword search
    sparse_results = es.search_keywords(
        query_text="sparse keyword fusion capabilities",
        tenant_id=tenant_id,
        top_k=5,
        index_name=idx_name,
    )
    assert any(
        item["chunk_id"] == chunk_id for item in sparse_results
    ), "BM25 keyword retrieval failed to locate ingested chunk"

    # Verify Postgres record inspection
    record = pg_store.get_chunk_record(chunk_id)
    assert record is not None
    assert record.id == chunk_id
    assert record.content == content
    assert record.tenant_id == tenant_id
    assert record.embedding_model == embedder.model_name
