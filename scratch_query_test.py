import asyncio
import sys
import time
from pathlib import Path

# Ensure project root and services are in PYTHONPATH
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "services" / "correction"))
sys.path.insert(0, str(root_dir / "services" / "retrieval"))
sys.path.insert(0, str(root_dir / "services" / "generation"))

from shared.models.common import ConfidenceStatus
from services.correction.app.services.graph import get_correction_graph

async def main():
    query_str = "How are annual bonus multipliers calculated when overtime hours are involved?"
    
    print("=========================================================================")
    print("STARTING LANGGRAPH SELF-CORRECTING RAG QUERY TEST")
    print(f"Query: \"{query_str}\"")
    print("=========================================================================\n")
    
    t0 = time.perf_counter()
    graph = get_correction_graph()
    
    initial_state = {
        "query": query_str,
        "original_query": query_str,
        "tenant_id": "tenant_enterprise",
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
    
    print("[1/2] Invoking LangGraph Correction Router (Retrieval -> Cross-Encoder -> Generation)...")
    final_state = await graph.ainvoke(initial_state)
    elapsed = (time.perf_counter() - t0) * 1000
    
    print(f"[2/2] Graph execution finished in {elapsed:.1f}ms!\n")
    print("======================== FINAL PIPELINE RESPONSE ========================")
    print(f"Status Badge:       {final_state.get('final_status')}")
    print(f"Confidence Score:   {final_state.get('groundedness_score', final_state.get('retrieval_confidence', 0.0)) * 100:.1f}%")
    print(f"Contradiction Note: {final_state.get('contradiction_reasoning', 'None detected')}")
    print(f"Attempt Count:      {final_state.get('attempt_count')}")
    print("\n----------------------------- ANSWER TEXT -----------------------------")
    print(final_state.get("final_answer", "No answer generated."))
    print("\n------------------------------ CITATIONS ------------------------------")
    citations = final_state.get("draft_citations", [])
    if not citations:
        print("No citations generated.")
    for i, c in enumerate(citations, 1):
        print(f"[{i}] Chunk ID: {c.chunk_id} | Source: {c.source} (Page {c.page_number})")
        print(f"    Snippet: {c.quote_snippet}")
    print("=========================================================================")

if __name__ == "__main__":
    asyncio.run(main())
