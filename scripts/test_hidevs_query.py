"""
Test script to run the exact HiDevs query end-to-end against the updated Self-Correcting RAG pipeline.
"""

import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "services" / "ingestion"))

from app.pipeline.pii_redaction import get_pii_service
from app.pipeline.chunker import get_chunker_service
from services.correction.app.services.graph import get_correction_graph
from shared.models.common import ConfidenceStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_hidevs_query")

resume_text = """G. Surya
Bangalore, India | +91 86672 34480 | iamsurya195@gmail.com | linkedin.com/in/g-surya-63a01b290 | github.com/Surya8663

Summary
Generative AI Engineer with production experience building RAG systems, agentic workflows, and voice AI applications using Python, FastAPI, LangGraph, and Azure OpenAI. Built retrieval systems over 13,000+ profiles and reduced chatbot latency from 10–15 seconds to 2.5 seconds.

Experience
Generative AI Developer Intern Nov 2025 – May 2026
HiDevs Bangalore, India (Hybrid)
– Built PeopleGPT / Aura AI Agent – RAG pipeline over 13,000+ profiles using Qdrant hybrid search, LangChain, FastAPI (SSE); cut manual search time by 70%.
– Optimized Dave Chatbot latency from 10–15s to ~2.5s via code review; added streaming and MongoDB session persistence.
– Built AI GitHub Repository Analyzer (AST checks, commit analysis, Gemini summaries); reduced codebase review time by 40%.

AI Intern – Azure AI Services Mar 2024 – Jun 2024
In-Biot Private Limited Bangalore, India
– Developed five Azure AI prototypes across computer vision, NLP, and Generative AI using Azure Cognitive Services; earned the Microsoft AI-900 certification.

Research Experience
Research Collaborator – Zero-Shot Traffic Accident Anticipation Mar 2026 – Apr 2026
Dr. Sudaroli Dhananjeyan, TJIT Bangalore, India
– Co-developed a zero-shot multi-modal ensemble (CLIP ViT-L/14, optical flow, MiniLM); paper submitted to an Elsevier journal.

Education
T. John Institute of Technology (VTU) Aug 2023 – May 2027 (Expected)
B.E. Computer Science and Engineering – CGPA: 8.7/10.0 Bangalore, India

Skills
AI/ML: LLMs, GenAI, Agentic AI, RAG, Azure OpenAI, AI Search, LangChain, LangGraph, Multi-Agent Systems
"""

async def run():
    print("==============================================================")
    print("1. INGESTING RESUME & INDEXING INTO QDRANT AND ELASTICSEARCH...")
    print("==============================================================")
    tenant_id = "tenant_enterprise"
    document_id = "G_Surya_Resume.pdf"
    pii = get_pii_service()
    redacted = pii.redact_text(resume_text)
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
    print(f"Generated {len(chunks)} chunks.")

    # Index into Qdrant and Elasticsearch under tenant_enterprise
    from services.retrieval.app.services.qdrant_store import get_qdrant_store
    from services.retrieval.app.services.es_store import get_es_store
    from services.retrieval.app.services.embedder import get_embedding_service

    q_store = get_qdrant_store()
    es_store = get_es_store()
    embedder = get_embedding_service()

    q_store.ensure_collection(collection_name="compass_rag_chunks", dimension=384)
    es_store.ensure_index(index_name="compass_rag_chunks")

    q_indexed = 0
    es_indexed = 0
    for c in chunks:
        vec = embedder.embed_text(c.content)
        if q_store.upsert_chunk(c, vec):
            q_indexed += 1
        if es_store.index_chunk(c):
            es_indexed += 1

    print(f"Indexing Complete: Qdrant={q_indexed}/{len(chunks)}, Elasticsearch={es_indexed}/{len(chunks)} for tenant '{tenant_id}'.")

    print("\n==============================================================")
    print("2. VERIFYING INDEXED CHUNKS IN QDRANT & ELASTICSEARCH...")
    print("==============================================================")
    q_hits = q_store.search(query_vector=embedder.embed_text("HiDevs"), tenant_id=tenant_id, top_k=5)
    es_hits = es_store.search_keywords(query_text="HiDevs", tenant_id=tenant_id, top_k=5)
    print(f"Qdrant Search 'HiDevs': {len(q_hits)} candidate chunks found.")
    print(f"Elasticsearch Search 'HiDevs': {len(es_hits)} candidate chunks found.")

    print("\n==============================================================")
    print("3. EXECUTING CORRECTION GRAPH WITH GENUINE RETRIEVAL (NO FAKE SCORES)...")
    print("==============================================================")
    query = "Summarize Surya's work experience at HiDevs, including what he built and the measurable impact."
    graph = get_correction_graph()

    # Pass empty retrieved_chunks to force genuine retrieval from Qdrant/ES
    initial_state = {
        "query": query,
        "original_query": query,
        "tenant_id": tenant_id,
        "attempt_count": 0,
        "retrieved_chunks": [],
        "retrieval_confidence": 0.0,
        "retrieval_status": ConfidenceStatus.VERIFIED,
        "contradictions_detected": False,
        "contradiction_reasoning": "",
        "has_same_date_contradiction": False,
        "draft_answer": "",
        "draft_citations": [],
        "groundedness_score": 0.0,
        "groundedness_verdict": True,
        "verdicts": [],
        "final_answer": "",
        "final_status": ConfidenceStatus.VERIFIED,
    }

    res = await graph.ainvoke(initial_state)

    print("\n==============================================================")
    print("FINAL HIDEVS QUERY RESPONSE")
    print("==============================================================")
    print(f"Status: {res.get('final_status')}")
    print(f"Confidence Score: {res.get('groundedness_score', 0.0)}")
    final_answer = res.get("final_answer", "")
    print(f"Final Answer:\n{final_answer}")
    print("\nCitations:")
    for cit in res.get("draft_citations", []):
        print(f"- [{cit.chunk_id}] (Source: {cit.source}, Page: {cit.page_number}): {cit.quote_snippet[:100]}")
    print("==============================================================")

    # Verification of required key facts
    required_facts = ["PeopleGPT", "13,000", "70%", "Dave", "2.5", "GitHub", "40%"]
    missing = [fact for fact in required_facts if fact.lower() not in final_answer.lower()]
    if missing:
        print(f"\n[WARNING] Missing key facts in final answer: {missing}")
    else:
        print("\n[SUCCESS] All required HiDevs experience facts are present in the final answer!")

if __name__ == "__main__":
    asyncio.run(run())
