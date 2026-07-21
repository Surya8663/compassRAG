"""
Runs the 15-question evaluation benchmark and prints the side-by-side comparison summary.
"""

import logging
import sys

from services.evaluation.app.services.report_generator import EvaluationRunnerService

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("benchmark_runner")

if __name__ == "__main__":
    logger.info("Starting 15-question benchmark evaluation run...")
    try:
        runner = EvaluationRunnerService()
        report = runner.run_evaluation()
        print("\n" + "=" * 60)
        print("          COMPASS RAG EVALUATION BENCHMARK REPORT")
        print("=" * 60)
        print(f"Run ID: {report.run_id}")
        print(f"Timestamp: {report.timestamp}")
        print(f"Total Questions Evaluated: {report.total_questions}\n")
        print("SIDE-BY-SIDE METRICS:")
        print(
            f"  - Hallucination Rate:     "
            f"Baseline: {report.baseline_avg_hallucination_rate:.2%} | "
            f"Corrected: {report.corrected_avg_hallucination_rate:.2%} "
            f"({report.hallucination_improvement_pct:+.1f}%)"
        )
        print(
            f"  - Retrieval Recall:       "
            f"Baseline: {report.baseline_avg_retrieval_recall:.2%} | "
            f"Corrected: {report.corrected_avg_retrieval_recall:.2%} "
            f"({report.retrieval_recall_improvement_pct:+.1f}%)"
        )
        print(
            f"  - Citation Correctness:   "
            f"Baseline: {report.baseline_avg_citation_correctness:.2%} | "
            f"Corrected: {report.corrected_avg_citation_correctness:.2%} "
            f"({report.citation_correctness_improvement_pct:+.1f}%)"
        )
        print(
            f"  - Appropriate Abstention: "
            f"Baseline: {report.baseline_appropriate_abstention_rate:.2%} | "
            f"Corrected: {report.corrected_appropriate_abstention_rate:.2%} "
            f"({report.abstention_improvement_pct:+.1f}%)"
        )
        print(
            f"  - Average Latency:        "
            f"Baseline: {report.baseline_avg_latency_seconds:.3f}s | "
            f"Corrected: {report.corrected_avg_latency_seconds:.3f}s"
        )
        print("=" * 60 + "\n")
    except Exception as exc:
        logger.error("Benchmark run failed: %s", exc, exc_info=True)
        sys.exit(1)
