"""
Run full 12-question benchmark evaluation harness across Baseline RAG vs Self-Correcting RAG.
Generates evaluation_results.json and evaluation_report.md.
"""

import logging
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from services.evaluation.app.services.report_generator import (
    EvaluationRunnerService,
    load_golden_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_evaluation")

if __name__ == "__main__":
    try:
        logger.info("Initializing Evaluation Runner Service...")
        runner = EvaluationRunnerService()
        dataset = load_golden_dataset()
        logger.info("Loaded benchmark golden dataset with %d questions.", len(dataset))
        
        logger.info("Executing evaluation benchmark harness across 12 test questions...")
        report = runner.run_evaluation(dataset=dataset, tenant_id="eval_tenant")
        logger.info("Successfully executed benchmark evaluation. Run ID: %s", report.run_id)
        print("\n==============================================================")
        print("BENCHMARK EVALUATION COMPLETED SUCCESSFULLY")
        print("==============================================================")
        print(f"Hallucination Rate: Baseline={report.baseline_avg_hallucination_rate:.2%} -> Corrected={report.corrected_avg_hallucination_rate:.2%} ({report.hallucination_improvement_pct:.1f}% reduction)")
        print(f"Retrieval Recall: Baseline={report.baseline_avg_retrieval_recall:.2%} -> Corrected={report.corrected_avg_retrieval_recall:.2%}")
        print(f"Citation Correctness: Baseline={report.baseline_avg_citation_correctness:.2%} -> Corrected={report.corrected_avg_citation_correctness:.2%}")
        print(f"Appropriate Abstention: Baseline={report.baseline_appropriate_abstention_rate:.2%} -> Corrected={report.corrected_appropriate_abstention_rate:.2%}")
        print(f"Average Latency: Baseline={report.baseline_avg_latency_seconds:.3f}s -> Corrected={report.corrected_avg_latency_seconds:.3f}s")
        print("Output files generated: evaluation_results.json and evaluation_report.md")
        print("==============================================================\n")
    except Exception as exc:
        logger.error("Error during evaluation benchmark execution: %s", exc, exc_info=True)
        sys.exit(1)
