#!/usr/bin/env python3
"""
Diagnostic CLI tool: Print Qdrant and Elasticsearch chunk counts for tenant_id.
"""

import sys
from shared.tenant import resolve_tenant_id
from services.retrieval.app.services.qdrant_store import get_qdrant_store
from services.retrieval.app.services.es_store import get_es_store


def main():
    tenant_id = sys.argv[1] if len(sys.argv) > 1 else "tenant_enterprise"
    print(f"==============================================================")
    print(f"TENANT CHUNK DIAGNOSTIC: tenant_id='{tenant_id}'")
    print(f"==============================================================")

    # 1. Qdrant check
    qdrant_count = 0
    collection_name = "compass_rag_chunks"
    try:
        q_store = get_qdrant_store()
        res, _ = q_store.client.scroll(
            collection_name=collection_name,
            scroll_filter={
                "must": [
                    {"key": "tenant_id", "match": {"value": tenant_id}}
                ]
            },
            limit=100,
            with_payload=True,
        )
        qdrant_count = len(res)
        print(f"Qdrant collection '{collection_name}': {qdrant_count} chunks found for tenant '{tenant_id}'")
        for point in res[:3]:
            payload = point.payload or {}
            print(f"  - Chunk ID: {point.id} | Doc ID: {payload.get('document_id')} | Source: {payload.get('source')}")
    except Exception as exc:
        print(f"Qdrant error or collection missing: {exc}")

    # 2. Elasticsearch check
    es_count = 0
    index_name = "compass_rag_chunks"
    try:
        es_store = get_es_store()
        es_hits = es_store.search_keywords(query_text="*", tenant_id=tenant_id, top_k=100)
        es_count = len(es_hits)
        print(f"Elasticsearch index '{index_name}': {es_count} chunks found for tenant '{tenant_id}'")
        for hit in es_hits[:3]:
            payload = hit.get("payload", {})
            print(f"  - Chunk ID: {hit.get('chunk_id')} | Doc ID: {payload.get('document_id')} | Source: {payload.get('source')}")
    except Exception as exc:
        print(f"Elasticsearch error or index missing: {exc}")

    print(f"==============================================================")
    print(f"TOTAL INDEXED CHUNKS: Qdrant={qdrant_count}, Elasticsearch={es_count}")
    print(f"==============================================================")


if __name__ == "__main__":
    main()
