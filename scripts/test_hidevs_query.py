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
    print("1. RUNNING RESUME INGESTION & CHUNKING...")
    print("==============================================================")
    pii = get_pii_service()
    redacted = pii.redact_text(resume_text)
    chunker = get_chunker_service()
    chunks = chunker.chunk_and_tag_page(
        redacted_text=redacted,
        document_id="G_Surya_Resume.pdf",
        source="G_Surya_Resume.pdf",
        page_number=1,
        ingestion_timestamp=datetime.now(UTC),
        tenant_id="eval_tenant",
        version_id="v1",
    )
    print(f"Generated {len(chunks)} chunks.")
    for c in chunks:
        if "HiDevs" in c.content:
            print("\n--- RETRIEVED HIDEVS CHUNK CONTENT ---")
            print(c.content)

    print("\n==============================================================")
    print("2. EXECUTING CORRECTION GRAPH ON HIDEVS QUERY...")
    print("==============================================================")
    query = "Summarize Surya's work experience at HiDevs, including what he built and the measurable impact."
    graph = get_correction_graph()

    from shared.models.common import RetrievalResult
    retrieved_results = [
        RetrievalResult(
            chunk=c, vector_score=0.95, bm25_score=0.95, fused_score=0.95, rerank_score=0.95
        )
        for c in chunks
    ]

    initial_state = {
        "query": query,
        "original_query": query,
        "tenant_id": "eval_tenant",
        "attempt_count": 0,
        "retrieved_chunks": retrieved_results,
        "retrieval_confidence": 0.95,
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
    print(f"Final Answer:\n{res.get('final_answer')}")
    print("\nCitations:")
    for cit in res.get("draft_citations", []):
        print(f"- [{cit.chunk_id}] (Source: {cit.source}, Page: {cit.page_number}): {cit.quote_snippet[:100]}")
    print("==============================================================")

if __name__ == "__main__":
    asyncio.run(run())
