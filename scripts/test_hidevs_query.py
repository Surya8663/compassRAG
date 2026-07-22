import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "services" / "api-gateway"))

from app.services.orchestrator import gateway_orchestrator
from services.evaluation.app.data.corpus_generator import CORPUS_DIR, generate_golden_corpus
from services.ingestion.app.pipeline.chunker import get_chunker_service
from services.ingestion.app.pipeline.pii_redaction import get_pii_service
from services.retrieval.app.services.embedder import get_embedding_service
from services.retrieval.app.services.es_store import get_es_store
from services.retrieval.app.services.qdrant_store import get_qdrant_store
from shared.models.common import QueryRequest, TenantContext

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_hidevs_query")


async def run():
    print("==============================================================")
    print("1. INGESTING ACTUAL G_Surya_Resume.pdf INTO QDRANT & ELASTICSEARCH...")
    print("==============================================================")
    generate_golden_corpus()
    resume_path = CORPUS_DIR / "G_Surya_Resume.pdf"
    assert resume_path.exists(), f"Resume PDF not found at {resume_path}"

    import fitz
    doc = fitz.open(resume_path)
    page_text = doc[0].get_text("text") or ""
    doc.close()
    assert len(page_text.strip()) > 50, "Resume text must be non-empty"

    tenant_id = "tenant_enterprise"
    document_id = "G_Surya_Resume.pdf"

    pii = get_pii_service()
    redacted = pii.redact_text(page_text)
    chunker = get_chunker_service()
    chunks = chunker.chunk_and_tag_page(
        redacted_text=redacted,
        document_id=document_id,
        source=document_id,
        page_number=1,
        ingestion_timestamp=datetime.now(UTC),
        tenant_id=tenant_id,
        version_id="v1",
    )
    assert len(chunks) > 0, "Chunking must generate at least 1 chunk for resume"

    q_store = get_qdrant_store()
    es_store = get_es_store()
    embedder = get_embedding_service()

    # Clean slate: delete existing collection/index to avoid stale chunks from previous runs
    try:
        q_store.client.delete_collection(collection_name="compass_rag_chunks")
    except Exception:
        pass
    try:
        es_store.client.indices.delete(index="compass_rag_chunks", ignore=[400, 404])
    except Exception:
        pass

    q_store.ensure_collection(collection_name="compass_rag_chunks", dimension=384)
    es_store.ensure_index(index_name="compass_rag_chunks")

    indexed_chunk_map = {}
    for c in chunks:
        vec = embedder.embed_text(c.content)
        q_ok = q_store.upsert_chunk(c, vec)
        es_ok = es_store.index_chunk(c)
        assert q_ok, f"Failed to index chunk {c.id} into Qdrant"
        assert es_ok, f"Failed to index chunk {c.id} into Elasticsearch"
        indexed_chunk_map[c.id] = c.content

    print(f"Indexing Verified: {len(chunks)} chunks indexed into Qdrant & Elasticsearch under '{tenant_id}'.")

    query_str = "Summarize Surya's work experience at HiDevs, including what he built and the measurable impact."
    tenant_context = TenantContext(tenant_id=tenant_id, user_id="test_user", roles=["admin"])
    query_req = QueryRequest(query=query_str, tenant_id=tenant_id)

    response = await gateway_orchestrator.process_query(query_req, tenant_context)

    print("\n==============================================================")
    print("FINAL HIDEVS QUERY RESPONSE")
    print("==============================================================")
    print(f"Status: {response.confidence_status}")
    print(f"Confidence Score: {response.confidence_score}")
    print(f"Final Answer:\n{response.answer}")
    print("\nCitations:")
    for cit in response.citations:
        print(f"- [{cit.chunk_id}] (Source: {cit.source}, Page: {cit.page_number}): {cit.quote_snippet[:100]}")
    print("==============================================================")

    # STRICT ASSERTIONS - MUST EXIT NON-ZERO ON FAILURE
    status_str = str(response.confidence_status.value if hasattr(response.confidence_status, "value") else response.confidence_status)
    assert status_str == "VERIFIED", f"Expected status VERIFIED, got {status_str}"
    assert response.confidence_score > 0.0, f"Expected retrieval/confidence score > 0, got {response.confidence_score}"

    required_facts = ["PeopleGPT", "13,000", "70%", "Dave", "2.5", "GitHub", "40%"]
    missing = [fact for fact in required_facts if fact.lower() not in response.answer.lower()]
    assert not missing, f"Missing required HiDevs facts in final answer: {missing}"

    assert len(response.citations) > 0, "Final answer must have at least 1 valid citation"
    for cit in response.citations:
        assert cit.chunk_id in indexed_chunk_map, f"Citation references unknown chunk ID '{cit.chunk_id}'"
        assert cit.quote_snippet.strip().lower() in indexed_chunk_map[cit.chunk_id].lower(), f"Citation snippet '{cit.quote_snippet}' not supported by chunk content"

    print("\n[SUCCESS] ALL HIDEVS END-TO-END STRICT ASSERTIONS PASSED PERFECTLY!")


if __name__ == "__main__":
    asyncio.run(run())
