import sys
import time
from datetime import datetime, UTC
from pathlib import Path

# Add project root and services/ingestion to path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "services" / "ingestion"))

from app.pipeline.pii_redaction import get_pii_service
from app.pipeline.chunker import get_chunker_service

resume_text = """G. Surya
Bangalore, India | +91 86672 34480 | iamsurya195@gmail.com | linkedin.com/in/g-surya-63a01b290 | github.com/Surya8663

Summary
Generative AI Engineer with production experience building RAG systems, agentic workflows, and voice AI applications using Python, FastAPI, LangGraph, and Azure OpenAI. Built retrieval systems over 13,000+ profiles and reduced chatbot latency from 10–15 seconds to 2.5 seconds.

Experience
Generative AI Developer Intern Nov 2025 – May 2026
HiDevs Bangalore, India (Hybrid)
– Built PeopleGPT / Aura AI Agent – RAG pipeline over 13,000+ profiles using Qdrant hybrid search, LangChain, FastAPI (SSE); cut manual search time by 70%.
– Optimized Dave Chatbot latency from 10–15s to ∼2.5s via code review; added streaming and MongoDB session persistence.
– Built AI GitHub Repository Analyzer (AST checks, commit analysis, Gemini summaries); reduced codebase review time by 40%.

AI Intern – Azure AI Services Mar 2024 – Jun 2024
In-Biot Private Limited Bangalore, India
– Developed five Azure AI prototypes across computer vision, NLP, and Generative AI using Azure Cognitive Services; earned the Microsoft AI-900 certification.

Research Experience
Research Collaborator – Zero-Shot Traffic Accident Anticipation Mar 2026 – Apr 2026
Dr. Sudaroli Dhananjeyan, TJIT Bangalore, India
– Co-developed a zero-shot multi-modal ensemble (CLIP ViT-L/14, optical flow, MiniLM); paper submitted to an Elsevier journal.

Research Assistant – Autonomous Robotic Invigilation System Mar 2026 – Present
Dr. Sudaroli Dhananjeyan, TJIT Bangalore, India
– Contributing to a funded robotics project combining SLAM navigation and multi-modal sensor fusion (thermal, RF, EMF, UV/IR) with an Edge AI perception stack (YOLOv8, LSTM, CRAFT-OCR).

Education
T. John Institute of Technology (VTU) Aug 2023 – May 2027 (Expected)
B.E. Computer Science and Engineering – CGPA: 8.7/10.0 Bangalore, India
– Coursework: DSA, Software Engineering, Computer Networks, OS, AI/ML | Top-four finishes across 4 hackathons (300+ teams) | IIIC Coordinator

Skills
AI/ML: LLMs, GenAI, Agentic AI, RAG, Azure OpenAI, AI Search, LangChain, LangGraph, Multi-Agent Systems, Prompt Engineering, Embeddings, NLP, Whisper (STT)
Frameworks: Python, Django, Flask, FastAPI, REST API, SQLAlchemy, React, JavaScript, Java
Databases: SQL, PostgreSQL, MySQL, Supabase, Qdrant, MongoDB
Cloud/DevOps: Microsoft Azure, Docker, Git, CI/CD, Agile/Scrum, Twilio API, Groq API

Projects
DocuMind – Multi-Tenant Enterprise RAG Platform Jun 2026 – Jul 2026
Django, Azure OpenAI, Azure AI Search, PostgreSQL
– Architected multi-tenant knowledge-agent platform with tenant-isolated RAG (Azure OpenAI + Azure AI Search hybrid) and a LangGraph confidence-check agent.

HireAI – AI Voice Interview Screening Platform Jun 2026
FastAPI, React, Twilio, Groq, Whisper, Supabase
– Automated resume screening + voice interviews via Twilio/Whisper STT; cut first-round screening time by 75% (20 min to 5 min).

Kira – Multi-Agent Conversational AI Oct 2025 – Nov 2025
Python, LangGraph, LangChain, Qdrant, Streamlit
– Evaluated on a curated test set, achieving a 0.84 answer-relevance score and reducing unsupported responses by 35% against the single-agent baseline.

Certifications
Microsoft Azure AI Fundamentals AI-900 (Jun 2024) · Oracle Cloud Infrastructure AI Foundations Associate (Aug 2024) · AWS Foundations of Prompt Engineering (Aug 2024) · Google Cloud: Introduction to Generative AI (Jan 2025) · Anthropic: Claude 101 and Al Fluency Framework and Foundations (July 2026)"""

def main():
    print("==============================================================")
    print("STARTING RESUME INGESTION TEST (PII Redaction + LangChain Chunking)")
    print("==============================================================")
    
    t0 = time.perf_counter()
    print("[1/2] Loading spaCy en_core_web_sm model & Presidio PII Redaction Engine...")
    pii_service = get_pii_service()
    t_pii_init = time.perf_counter() - t0
    print(f"       -> Engine initialized in {t_pii_init:.2f} seconds.")

    t1 = time.perf_counter()
    print("[2/2] Running PII Redaction across resume text...")
    redacted_text = pii_service.redact_text(resume_text)
    t_redact = time.perf_counter() - t1
    print(f"       -> Redaction completed in {t_redact:.2f} seconds.")
    print("\n--- REDACTED RESUME PREVIEW (Sensitive Data Masked) ---")
    print(redacted_text[:600] + "\n[... truncated preview ...]\n")

    t2 = time.perf_counter()
    print("[3/3] Running LangChain Semantic Sentence-Boundary Chunking...")
    chunker = get_chunker_service()
    chunks = chunker.chunk_and_tag_page(
        redacted_text=redacted_text,
        document_id="doc_surya_resume_2026",
        source="G_Surya_Resume.pdf",
        page_number=1,
        ingestion_timestamp=datetime.now(UTC),
        tenant_id="tenant_enterprise",
        version_id="v1",
        ocr_confidence=1.0,
    )
    t_chunk = time.perf_counter() - t2
    print(f"       -> Created {len(chunks)} semantic chunks in {t_chunk:.4f} seconds.")
    
    print("\n--- GENERATED CHUNK METADATA & SNIPPETS ---")
    for i, c in enumerate(chunks, 1):
        print(f"\n[Chunk #{i}] ID: {c.id} (Length: {len(c.content)} chars)")
        print(f"Metadata: Tenant={c.metadata.tenant_id}, Source={c.metadata.source}, Page={c.metadata.page_number}")
        print(f"Content Preview: {c.content[:150]}...")

    total_time = time.perf_counter() - t0
    print("\n==============================================================")
    print(f"TOTAL DIRECT INGESTION TIME: {total_time:.2f} seconds")
    print("==============================================================")

if __name__ == "__main__":
    main()
