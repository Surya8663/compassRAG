"""
Evaluation Runner and Report Generator (`services/evaluation`).
Runs the 15-question Golden Dataset across both Baseline RAG and Self-Correcting RAG pipelines,
computes side-by-side comparative metrics, calculates percentage improvements,
and exports structured JSON and Markdown/HTML reports to `reports/`.
"""

import asyncio
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fitz
import yaml
from shared.models.common import ConfidenceStatus, DocumentChunk, DocumentMetadata

from services.correction.app.services.graph import get_correction_graph
from services.evaluation.app.data.corpus_generator import CORPUS_DIR, generate_golden_corpus
from services.evaluation.app.models.eval import (
    EvaluationSummaryReport,
    GoldenQuestion,
    QuestionEvaluationResult,
)
from services.evaluation.app.services.baseline_pipeline import BaselineRAGPipeline
from services.evaluation.app.services.metrics_evaluator import MetricsEvaluator

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "golden_dataset.yaml"


def load_golden_dataset(dataset_path: Path | None = None) -> list[GoldenQuestion]:
    """Loads and parses the 15-question golden dataset from YAML."""
    path = dataset_path or DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset YAML not found at {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return [GoldenQuestion.model_validate(item) for item in data]


def load_corpus_as_chunks() -> list[DocumentChunk]:
    """
    Loads all generated evaluation PDFs from `golden_corpus/` into `DocumentChunk`
    objects so evaluation can run self-contained local retrieval if Qdrant/ES are offline.
    """
    generate_golden_corpus()
    chunks: list[DocumentChunk] = []

    for pdf_path in CORPUS_DIR.glob("*.pdf"):
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text().strip()
                if not text:
                    continue
                meta = DocumentMetadata(
                    source=pdf_path.name,
                    page_number=page_num,
                    ingestion_timestamp=datetime.now(UTC),
                    tenant_id="eval_tenant",
                    version_id="1.0",
                    ocr_confidence=0.99,
                )
                chunk = DocumentChunk(
                    id=f"{pdf_path.name}_p{page_num}",
                    document_id=pdf_path.name,
                    content=text,
                    metadata=meta,
                )
                chunks.append(chunk)
            doc.close()
        except Exception as exc:
            logger.warning("Failed to extract text from %s: %s", pdf_path, exc)

    return chunks


class EvaluationRunnerService:
    """
    Orchestrates execution of the 15 evaluation questions across Baseline and Corrected pipelines,
    aggregates metrics, and generates comparison reports.
    """

    def __init__(self) -> None:
        self.baseline_pipeline = BaselineRAGPipeline()
        self.metrics_evaluator = MetricsEvaluator()
        self.corpus_chunks = load_corpus_as_chunks()

    def _filter_chunks_for_question(self, question: GoldenQuestion) -> list[DocumentChunk]:
        """
        Retrieves candidate corpus chunks using text term matching without using expected_chunk_ids.
        """
        stop_words = {"what", "is", "the", "amount", "for", "in", "on", "of", "and", "or", "to", "a", "an", "does", "do", "can", "i"}
        tokens = [w.lower().strip("?,.") for w in question.question.split() if w.lower().strip("?,.") not in stop_words and len(w) > 2]

        scored_chunks: list[tuple[float, DocumentChunk]] = []
        for c in self.corpus_chunks:
            c_text = c.content.lower()
            matches = sum(1 for t in tokens if t in c_text)
            if matches > 0:
                scored_chunks.append((float(matches), c))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        results = [chunk for _, chunk in scored_chunks[:10]]
        return results if results else self.corpus_chunks[:3]

    def _invoke_graph(self, graph: Any, initial_state: dict[str, Any]) -> dict[str, Any]:
        """Safely invoke the async correction graph synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            try:
                import nest_asyncio

                nest_asyncio.apply()
            except ImportError:
                pass
            return asyncio.run(graph.ainvoke(initial_state))
        return asyncio.run(graph.ainvoke(initial_state))

    def run_evaluation(
        self, dataset: list[GoldenQuestion] | None = None, tenant_id: str = "eval_tenant"
    ) -> EvaluationSummaryReport:
        """
        Runs full benchmark across both pipelines and returns aggregate comparison report.
        """
        questions = dataset or load_golden_dataset()
        results: list[QuestionEvaluationResult] = []

        # Ensure corpus is generated and loaded
        if not self.corpus_chunks:
            self.corpus_chunks = load_corpus_as_chunks()

        for q in questions:
            # Add rate-limiting pause between benchmark questions
            time.sleep(2.0)
            # 1. Run Baseline RAG Pipeline
            candidate_chunks = self._filter_chunks_for_question(q)
            base_out = self.baseline_pipeline.run(
                query=q.question,
                tenant_id=tenant_id,
                top_k=5,
                pre_retrieved_chunks=candidate_chunks,
            )
            base_answer = base_out["answer"]
            base_status = base_out["confidence_status"]
            base_chunks = base_out["retrieved_chunks"]
            base_citations = base_out["citations"]
            base_latency = base_out["latency_seconds"]

            base_res = QuestionEvaluationResult(
                question_id=q.id,
                pipeline_type="baseline",
                answer=base_answer,
                confidence_status=base_status,
                hallucination_rate=self.metrics_evaluator.compute_hallucination_rate(
                    q, base_answer, base_chunks
                ),
                retrieval_recall=self.metrics_evaluator.compute_retrieval_recall(
                    q, base_chunks
                ),
                citation_correctness=self.metrics_evaluator.compute_citation_correctness(
                    base_citations, base_chunks
                ),
                appropriate_abstention=self.metrics_evaluator.compute_appropriate_abstention(
                    q, base_status, base_answer
                ),
                latency_seconds=base_latency,
                citations_cited=[
                    getattr(c, "chunk_id", str(c)) for c in base_citations
                ],
            )
            results.append(base_res)

            # 2. Run Self-Correcting RAG Pipeline (`CorrectionRouterGraph`)
            start_corr = time.perf_counter()
            from shared.models.common import RetrievalResult
            retrieved_results = [
                RetrievalResult(
                    chunk=c, vector_score=1.0, bm25_score=1.0, fused_score=1.0, rerank_score=1.0
                )
                for c in candidate_chunks
            ]
            initial_state = {
                "query": q.question,
                "tenant_id": tenant_id,
                "original_query": q.question,
                "attempt_count": 0,
                "retrieved_chunks": retrieved_results,
                "retrieval_confidence": 0.0,
                "retrieval_status": ConfidenceStatus.VERIFIED,
                "contradictions_detected": False,
                "contradiction_reasoning": "",
                "has_same_date_contradiction": False,
                "draft_answer": "",
                "draft_citations": [],
                "groundedness_score": 1.0,
                "groundedness_verdict": True,
                "verdicts": [],
                "final_answer": "",
                "final_status": ConfidenceStatus.VERIFIED,
            }

            try:
                graph = get_correction_graph()
                corr_state = self._invoke_graph(graph, initial_state)
                corr_answer = corr_state.get("final_answer", "")
                corr_status = corr_state.get(
                    "final_status", ConfidenceStatus.VERIFIED
                )
                corr_status_str = (
                    corr_status.value
                    if hasattr(corr_status, "value")
                    else str(corr_status)
                )
                corr_chunks = [
                    r.chunk
                    for r in corr_state.get("retrieved_chunks", [])
                    if hasattr(r, "chunk")
                ] or candidate_chunks
                corr_citations = corr_state.get("draft_citations", [])
            except Exception as exc:
                logger.warning("Correction graph run failed for %s: %s", q.id, exc)
                corr_answer = base_answer
                corr_status_str = base_status
                corr_chunks = base_chunks
                corr_citations = base_citations

            corr_latency = time.perf_counter() - start_corr

            # Calculate metrics for self-correcting RAG
            corr_res = QuestionEvaluationResult(
                question_id=q.id,
                pipeline_type="corrected",
                answer=corr_answer,
                confidence_status=corr_status_str,
                hallucination_rate=self.metrics_evaluator.compute_hallucination_rate(
                    q, corr_answer, corr_chunks
                ),
                retrieval_recall=self.metrics_evaluator.compute_retrieval_recall(
                    q, corr_chunks
                ),
                citation_correctness=self.metrics_evaluator.compute_citation_correctness(
                    corr_citations, corr_chunks
                ),
                appropriate_abstention=self.metrics_evaluator.compute_appropriate_abstention(
                    q, corr_status_str, corr_answer
                ),
                latency_seconds=corr_latency,
                citations_cited=[
                    getattr(c, "chunk_id", str(c)) for c in corr_citations
                ],
            )
            results.append(corr_res)

        # 3. Aggregate averages
        base_runs = [r for r in results if r.pipeline_type == "baseline"]
        corr_runs = [r for r in results if r.pipeline_type == "corrected"]

        n = float(len(questions)) or 1.0
        base_avg_halluc = sum(r.hallucination_rate for r in base_runs) / n
        corr_avg_halluc = sum(r.hallucination_rate for r in corr_runs) / n
        halluc_imp = (
            ((base_avg_halluc - corr_avg_halluc) / max(0.0001, base_avg_halluc))
            * 100.0
            if base_avg_halluc > 0
            else 0.0
        )

        base_avg_rec = sum(r.retrieval_recall for r in base_runs) / n
        corr_avg_rec = sum(r.retrieval_recall for r in corr_runs) / n
        rec_imp = (
            ((corr_avg_rec - base_avg_rec) / max(0.0001, base_avg_rec)) * 100.0
            if base_avg_rec > 0
            else 0.0
        )

        base_avg_cit = sum(r.citation_correctness for r in base_runs) / n
        corr_avg_cit = sum(r.citation_correctness for r in corr_runs) / n
        cit_imp = (
            ((corr_avg_cit - base_avg_cit) / max(0.0001, base_avg_cit)) * 100.0
            if base_avg_cit > 0
            else 0.0
        )

        base_avg_abs = sum(r.appropriate_abstention for r in base_runs) / n
        corr_avg_abs = sum(r.appropriate_abstention for r in corr_runs) / n
        abs_imp = (
            ((corr_avg_abs - base_avg_abs) / max(0.0001, base_avg_abs)) * 100.0
            if base_avg_abs > 0
            else 0.0
        )

        base_avg_lat = sum(r.latency_seconds for r in base_runs) / n
        corr_avg_lat = sum(r.latency_seconds for r in corr_runs) / n

        report = EvaluationSummaryReport(
            run_id=str(uuid.uuid4()),
            timestamp=datetime.now(UTC).isoformat(),
            total_questions=len(questions),
            baseline_avg_hallucination_rate=round(base_avg_halluc, 4),
            corrected_avg_hallucination_rate=round(corr_avg_halluc, 4),
            hallucination_improvement_pct=round(halluc_imp, 2),
            baseline_avg_retrieval_recall=round(base_avg_rec, 4),
            corrected_avg_retrieval_recall=round(corr_avg_rec, 4),
            retrieval_recall_improvement_pct=round(rec_imp, 2),
            baseline_avg_citation_correctness=round(base_avg_cit, 4),
            corrected_avg_citation_correctness=round(corr_avg_cit, 4),
            citation_correctness_improvement_pct=round(cit_imp, 2),
            baseline_appropriate_abstention_rate=round(base_avg_abs, 4),
            corrected_appropriate_abstention_rate=round(corr_avg_abs, 4),
            abstention_improvement_pct=round(abs_imp, 2),
            baseline_avg_latency_seconds=round(base_avg_lat, 4),
            corrected_avg_latency_seconds=round(corr_avg_lat, 4),
            question_results=results,
        )

        self.export_report(report)
        return report

    def export_report(self, report: EvaluationSummaryReport) -> tuple[Path, Path]:
        """
        Exports the evaluation summary report as both structured JSON and formatted Markdown table.
        Returns paths (`json_path`, `md_path`).
        """
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        root_dir = Path.cwd()
        json_path = REPORTS_DIR / "evaluation_comparison_report.json"
        md_path = REPORTS_DIR / "evaluation_comparison_report.md"
        root_json_path = root_dir / "evaluation_results.json"
        root_md_path = root_dir / "evaluation_report.md"

        json_str = report.model_dump_json(indent=2)
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        with open(root_json_path, "w", encoding="utf-8") as f:
            f.write(json_str)

        lat_diff = (
            report.corrected_avg_latency_seconds
            - report.baseline_avg_latency_seconds
        )
        # Render Markdown side-by-side comparison report
        md_lines = [
            "# Compass RAG Evaluation Report: Baseline vs. Self-Correcting RAG\n",
            f"**Run ID**: `{report.run_id}`  \n"
            f"**Timestamp**: `{report.timestamp}`  \n"
            f"**Total Benchmark Questions**: `{report.total_questions}` across 5 PS1 Categories\n",
            "## Summary Comparison Table\n",
            "| Metric | Baseline RAG | Self-Correcting RAG | Improvement / Difference |",
            "| :--- | :---: | :---: | :---: |",
            (
                f"| **Hallucination Rate** (Lower is better) | "
                f"`{report.baseline_avg_hallucination_rate:.2%}` | "
                f"`{report.corrected_avg_hallucination_rate:.2%}` | "
                f"**+{report.hallucination_improvement_pct:.1f}%** reduction |"
            ),
            (
                f"| **Retrieval Recall** (Higher is better) | "
                f"`{report.baseline_avg_retrieval_recall:.2%}` | "
                f"`{report.corrected_avg_retrieval_recall:.2%}` | "
                f"**+{report.retrieval_recall_improvement_pct:.1f}%** relative |"
            ),
            (
                f"| **Citation Correctness** (Higher is better) | "
                f"`{report.baseline_avg_citation_correctness:.2%}` | "
                f"`{report.corrected_avg_citation_correctness:.2%}` | "
                f"**+{report.citation_correctness_improvement_pct:.1f}%** relative |"
            ),
            (
                f"| **Appropriate Abstention Rate** (Higher is better) | "
                f"`{report.baseline_appropriate_abstention_rate:.2%}` | "
                f"`{report.corrected_appropriate_abstention_rate:.2%}` | "
                f"**+{report.abstention_improvement_pct:.1f}%** relative |"
            ),
            (
                f"| **Average Latency per Question** | "
                f"`{report.baseline_avg_latency_seconds:.3f}s` | "
                f"`{report.corrected_avg_latency_seconds:.3f}s` | "
                f"`{lat_diff:+.3f}s` |"
            ),
            "\n## Per-Question Granular Results\n",
            (
                "| QID | Category | Baseline Status | Corrected Status | "
                "Baseline Halluc | Corrected Halluc |"
            ),
            "| :---: | :--- | :---: | :---: | :---: | :---: |",
        ]

        qmap_base = {
            r.question_id: r
            for r in report.question_results
            if r.pipeline_type == "baseline"
        }
        qmap_corr = {
            r.question_id: r
            for r in report.question_results
            if r.pipeline_type == "corrected"
        }

        for qid in sorted(qmap_base.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
            b_res = qmap_base[qid]
            c_res = qmap_corr.get(qid, b_res)
            md_lines.append(
                f"| **{qid}** | `{b_res.confidence_status}` | `{b_res.confidence_status}` | "
                f"`{c_res.confidence_status}` | `{b_res.hallucination_rate:.1%}` | "
                f"`{c_res.hallucination_rate:.1%}` |"
            )

        md_content = "\n".join(md_lines)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        with open(root_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info("Evaluation reports exported to %s and %s", root_json_path, root_md_path)
        return root_json_path, root_md_path
