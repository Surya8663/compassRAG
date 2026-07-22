import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
json_path = root / "evaluation_results.json"
md_path = root / "evaluation_report.md"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=== REPOSITORY-ROOT EVALUATION REPORT METRICS ===")
print(f"Root JSON Path: {json_path}")
print(f"Root MD Path: {md_path}")
print(f"Run ID: {data['run_id']}")
print(f"Total Questions: {data['total_questions']}")
print(f"Total Results in JSON: {len(data['question_results'])}")
print(f"Baseline Hallucination Rate: {data['baseline_avg_hallucination_rate']:.2%}")
print(f"Corrected Hallucination Rate: {data['corrected_avg_hallucination_rate']:.2%}")
print(f"Baseline Retrieval Recall: {data['baseline_avg_retrieval_recall']:.2%}")
print(f"Corrected Retrieval Recall: {data['corrected_avg_retrieval_recall']:.2%}")
print(f"Baseline Citation Correctness: {data['baseline_avg_citation_correctness']:.2%}")
print(f"Corrected Citation Correctness: {data['corrected_avg_citation_correctness']:.2%}")
print(f"Baseline Abstention Rate: {data['baseline_appropriate_abstention_rate']:.2%}")
print(f"Corrected Abstention Rate: {data['corrected_appropriate_abstention_rate']:.2%}")
print(f"Baseline Avg Latency: {data['baseline_avg_latency_seconds']:.3f}s")
print(f"Corrected Avg Latency: {data['corrected_avg_latency_seconds']:.3f}s")

print("\n=== PER-QUESTION DETAILED STATUS & ANSWERS ===")
q_base = {r["question_id"]: r for r in data["question_results"] if r["pipeline_type"] == "baseline"}
q_corr = {r["question_id"]: r for r in data["question_results"] if r["pipeline_type"] == "corrected"}

for qid in sorted(q_base.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
    b = q_base[qid]
    c = q_corr[qid]
    print(f"\n--- {qid} ---")
    print(f"Baseline Status: {b['confidence_status']}")
    print(f"Corrected Status: {c['confidence_status']}")
    print(f"Baseline Answer: {b['answer']}")
    print(f"Corrected Answer: {c['answer']}")
    print(f"Corrected Citations: {c['citations_cited']}")
