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


def load_corpus_as_chunks(tenant_id: str = "eval_tenant") -> tuple[list[DocumentChunk], dict[str, Any]]:
    """
    Ingests all 6 generated evaluation PDFs from `golden_corpus/` into Qdrant and Elasticsearch under `tenant_id`.
    Sends scanned image receipts through document classification and Tesseract OCR.
    Enforces strict OCR assertions for reimbursement_receipt_scanned.pdf.
    """
    from services.ingestion.app.pipeline.classifier import classify_and_extract_page
    from services.ingestion.app.pipeline.chunker import get_chunker_service
    from services.ingestion.app.pipeline.ocr import process_page_ocr
    from services.ingestion.app.pipeline.pii_redaction import get_pii_service
    from services.retrieval.app.services.embedder import get_embedding_service
    from services.retrieval.app.services.es_store import get_es_store
    from services.retrieval.app.services.qdrant_store import get_qdrant_store

    generate_golden_corpus()

    pii = get_pii_service()
    chunker = get_chunker_service()
    embedder = get_embedding_service()
    q_store = get_qdrant_store()
    es_store = get_es_store()

    q_store.ensure_collection(collection_name="compass_rag_chunks", dimension=384)
    es_store.ensure_index(index_name="compass_rag_chunks")

    all_chunks: list[DocumentChunk] = []
    ocr_meta: dict[str, Any] = {}

    for pdf_path in CORPUS_DIR.glob("*.pdf"):
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc, start=1):
                native_text = page.get_text("text") or ""
                page_class, text_extract = classify_and_extract_page(page, min_text_length=20)

                ocr_invoked = False
                page_text = text_extract
                ocr_conf = 0.99

                if page_class == "SCANNED_IMAGE" or not text_extract.strip():
                    ocr_invoked = True
                    ocr_text, ocr_conf = process_page_ocr(page)
                    page_text = ocr_text

                # Perform strict assertions on reimbursement_receipt_scanned.pdf
                if pdf_path.name == "reimbursement_receipt_scanned.pdf":
                    ocr_meta["native_text_len"] = len(native_text.strip())
                    ocr_meta["page_class"] = page_class
                    ocr_meta["ocr_invoked"] = ocr_invoked
                    ocr_meta["ocr_text"] = page_text

                    logger.info("OCR Verification for %s: native_len=%d, class=%s, ocr_invoked=%s", pdf_path.name, len(native_text.strip()), page_class, ocr_invoked)

                    assert len(native_text.strip()) <= 20, f"Native text should be empty/nearly empty for scanned receipt, got len={len(native_text.strip())}"
                    assert page_class == "SCANNED_IMAGE", f"Expected SCANNED_IMAGE classification, got {page_class}"
                    assert ocr_invoked, "Tesseract OCR should have been invoked for scanned receipt"
                    assert "Apex Catering" in page_text, f"OCR failed to extract 'Apex Catering' from receipt. Text: {page_text}"
                    assert "485.50" in page_text, f"OCR failed to extract '485.50' from receipt. Text: {page_text}"
                    assert "99-8877665" in page_text, f"OCR failed to extract '99-8877665' from receipt. Text: {page_text}"
                    logger.info("All OCR assertions passed for reimbursement_receipt_scanned.pdf!")

                if not page_text.strip():
                    continue

                redacted = pii.redact_text(page_text)
                p_chunks = chunker.chunk_and_tag_page(
                    redacted_text=redacted,
                    document_id=pdf_path.name,
                    source=pdf_path.name,
                    page_number=page_num,
                    ingestion_timestamp=datetime.now(UTC),
                    tenant_id=tenant_id,
                    version_id="1.0",
                )

                for c in p_chunks:
                    all_chunks.append(c)
                    vec = embedder.embed_text(c.content)
                    q_store.upsert_chunk(c, vec)
                    es_store.index_chunk(c)

            doc.close()
        except Exception as exc:
            logger.error("Failed to process benchmark PDF %s: %s", pdf_path, exc, exc_info=True)
            raise

    logger.info("Successfully ingested %d chunks into Qdrant & ES for tenant '%s'", len(all_chunks), tenant_id)
    return all_chunks, ocr_meta


class EvaluationRunnerService:
    """
    Orchestrates execution of the 12 evaluation questions across Baseline and Corrected pipelines,
    aggregates metrics, and generates comparison reports.
    """

    def __init__(self) -> None:
        self.baseline_pipeline = BaselineRAGPipeline()
        self.metrics_evaluator = MetricsEvaluator()

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

        # 1. Ingest corpus into Qdrant & ES with strict OCR assertions
        logger.info("Starting benchmark corpus ingestion for tenant '%s'...", tenant_id)
        corpus_chunks, ocr_meta = load_corpus_as_chunks(tenant_id=tenant_id)
        assert len(corpus_chunks) > 0, "Ingested corpus chunks must not be empty"

        for q in questions:
            time.sleep(1.0)
            logger.info("Evaluating Question %s (%s): %s", q.id, q.category, q.question)

            # 1. Run Baseline RAG Pipeline through genuine retrieval
            base_out = self.baseline_pipeline.run(
                query=q.question,
                tenant_id=tenant_id,
                top_k=5,
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

            # 2. Run Self-Correcting RAG Pipeline (`CorrectionRouterGraph`) through genuine retrieval
            start_corr = time.perf_counter()
            initial_state = {
                "query": q.question,
                "tenant_id": tenant_id,
                "original_query": q.question,
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
                "groundedness_verdict": False,
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
                    r.chunk if hasattr(r, "chunk") else r
                    for r in corr_state.get("retrieved_chunks", [])
                ]
                corr_citations = corr_state.get("draft_citations", [])
            except Exception as exc:
                logger.warning("Correction graph run failed for %s: %s", q.id, exc)
                corr_answer = "Error during correction graph execution."
                corr_status_str = "LOW_CONFIDENCE"
                corr_chunks = []
                corr_citations = []

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
            # CRITICAL FIX: Append corr_res immediately inside question loop
            results.append(corr_res)

        # STRICT ASSERTIONS ON RESULT COUNTS AND INFRASTRUCTURE
        base_runs = [r for r in results if r.pipeline_type == "baseline"]
        corr_runs = [r for r in results if r.pipeline_type == "corrected"]

        assert len(base_runs) == 12, f"Expected exactly 12 baseline results, got {len(base_runs)}"
        assert len(corr_runs) == 12, f"Expected exactly 12 corrected results, got {len(corr_runs)}"
        assert len(results) == 24, f"Expected exactly 24 total QuestionEvaluationResult objects, got {len(results)}"
        logger.info("All 24 evaluation results successfully collected and verified!")

        # MAP RESULTS FOR GRANULAR ASSERTIONS
        qmap_base = {r.question_id: r for r in base_runs}
        qmap_corr = {r.question_id: r for r in corr_runs}

        # 1. BASELINE RETRIEVAL ASSERTIONS
        base_avg_recall = sum(r.retrieval_recall for r in base_runs) / len(base_runs)
        assert base_avg_recall > 0.0, f"Baseline average retrieval recall must be > 0 (got {base_avg_recall})"

        base_avg_latency = sum(r.latency_seconds for r in base_runs) / len(base_runs)
        assert base_avg_latency > 0.01, f"Baseline average latency must not be near zero (got {base_avg_latency}s)"

        answerable_qids = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q9", "Q10", "Q12"]
        for qid in answerable_qids:
            b_res = qmap_base.get(qid)
            assert b_res is not None, f"Baseline result missing for {qid}"
            assert b_res.retrieval_recall > 0.0, f"Baseline {qid} must retrieve at least one expected chunk"
            assert "No verified context available" not in b_res.answer, f"Baseline {qid} must not return 'No verified context available'"

        # 2. CORRECTED OCR RETRIEVAL (Q5 & Q6)
        for qid in ["Q5", "Q6"]:
            q_res = qmap_corr.get(qid)
            assert q_res is not None, f"Result missing for {qid}"
            assert q_res.retrieval_recall > 0.0, f"{qid} must successfully retrieve OCR receipt chunks"

        # 3. CORRECTED ABSTENTION (Q7 & Q8 must abstain with non-VERIFIED status)
        for qid in ["Q7", "Q8"]:
            q_res = qmap_corr.get(qid)
            assert q_res is not None, f"Result missing for {qid}"
            assert q_res.confidence_status in ("LOW_CONFIDENCE", "INSUFFICIENT_INFORMATION"), (
                f"{qid} must abstain with non-VERIFIED status (got {q_res.confidence_status})"
            )
            assert q_res.appropriate_abstention == 1.0, f"{qid} must appropriately abstain on unanswerable queries"

        # 4. VERSION 2.0 POLICY (Q9 & Q10)
        q9_res = qmap_corr.get("Q9")
        assert q9_res is not None, "Result missing for Q9"
        assert "250" in q9_res.answer, f"Q9 must use $250 policy (got '{q9_res.answer}')"
        assert all("v1" not in c.lower() and "2024" not in c.lower() for c in q9_res.citations_cited), (
            "Q9 must cite only Version 2.0 policy"
        )

        q10_res = qmap_corr.get("Q10")
        assert q10_res is not None, "Result missing for Q10"
        assert "95" in q10_res.answer, f"Q10 must use $95 policy (got '{q10_res.answer}')"
        assert all("v1" not in c.lower() and "2024" not in c.lower() for c in q10_res.citations_cited), (
            "Q10 must cite only Version 2.0 policy"
        )

        # 5. AMBIGUOUS QUESTION (Q11)
        q11_res = qmap_corr.get("Q11")
        assert q11_res is not None, "Result missing for Q11"
        assert q11_res.confidence_status in ("CLARIFICATION_NEEDED", "LOW_CONFIDENCE"), (
            f"Q11 status must be CLARIFICATION_NEEDED or LOW_CONFIDENCE (got {q11_res.confidence_status})"
        )

        # 6. HIDEVS EXPERIENCE (Q12)
        q12_res = qmap_corr.get("Q12")
        assert q12_res is not None, "Result missing for Q12"
        q12_ans_lower = q12_res.answer.lower()
        for fact in ["peoplegpt", "13,000", "70%", "dave", "2.5", "github", "40%"]:
            assert fact in q12_ans_lower, f"Q12 final answer missing required fact '{fact}'"

        report = self.generate_summary(results)
        self.export_report(report)
        return report

    def generate_summary(
        self, results: list[QuestionEvaluationResult]
    ) -> EvaluationSummaryReport:
        base_res = [r for r in results if r.pipeline_type == "baseline"]
        corr_res = [r for r in results if r.pipeline_type == "corrected"]

        def avg(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        base_halluc = avg([r.hallucination_rate for r in base_res])
        corr_halluc = avg([r.hallucination_rate for r in corr_res])
        halluc_imp = (
            ((base_halluc - corr_halluc) / base_halluc * 100.0) if base_halluc > 0 else 0.0
        )

        base_recall = avg([r.retrieval_recall for r in base_res])
        corr_recall = avg([r.retrieval_recall for r in corr_res])
        recall_imp = (
            ((corr_recall - base_recall) / base_recall * 100.0) if base_recall > 0 else 0.0
        )

        base_cit = avg([r.citation_correctness for r in base_res])
        corr_cit = avg([r.citation_correctness for r in corr_res])
        cit_imp = ((corr_cit - base_cit) / base_cit * 100.0) if base_cit > 0 else 0.0

        base_abs = avg([r.appropriate_abstention for r in base_res])
        corr_abs = avg([r.appropriate_abstention for r in corr_res])
        abs_imp = ((corr_abs - base_abs) / base_abs * 100.0) if base_abs > 0 else 0.0

        return EvaluationSummaryReport(
            run_id=str(uuid.uuid4()),
            timestamp=datetime.now(UTC).isoformat(),
            total_questions=len(base_res),
            baseline_avg_hallucination_rate=round(base_halluc, 4),
            corrected_avg_hallucination_rate=round(corr_halluc, 4),
            hallucination_improvement_pct=round(halluc_imp, 2),
            baseline_avg_retrieval_recall=round(base_recall, 4),
            corrected_avg_retrieval_recall=round(corr_recall, 4),
            retrieval_recall_improvement_pct=round(recall_imp, 2),
            baseline_avg_citation_correctness=round(base_cit, 4),
            corrected_avg_citation_correctness=round(corr_cit, 4),
            citation_correctness_improvement_pct=round(cit_imp, 2),
            baseline_appropriate_abstention_rate=round(base_abs, 4),
            corrected_appropriate_abstention_rate=round(corr_abs, 4),
            abstention_improvement_pct=round(abs_imp, 2),
            baseline_avg_latency_seconds=round(avg([r.latency_seconds for r in base_res]), 4),
            corrected_avg_latency_seconds=round(avg([r.latency_seconds for r in corr_res]), 4),
            question_results=results,
        )

    def export_report(self, report: EvaluationSummaryReport) -> tuple[Path, Path]:
        """
        Exports the evaluation summary report as both structured JSON and formatted Markdown table.
        Returns paths (`root_json_path`, `root_md_path`).
        """
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        # Requirement 2: Correct repository-root path using .parents[4]
        root_dir = Path(__file__).resolve().parents[4]
        json_path = REPORTS_DIR / "evaluation_comparison_report.json"
        md_path = REPORTS_DIR / "evaluation_comparison_report.md"
        root_json_path = root_dir / "evaluation_results.json"
        root_md_path = root_dir / "evaluation_report.md"

        # Delete stale duplicate reports in services/ if present
        stale_json = root_dir / "services" / "evaluation_results.json"
        stale_md = root_dir / "services" / "evaluation_report.md"
        if stale_json.exists():
            try:
                stale_json.unlink()
            except Exception:
                pass
        if stale_md.exists():
            try:
                stale_md.unlink()
            except Exception:
                pass

        json_str = report.model_dump_json(indent=2)
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        with open(root_json_path, "w", encoding="utf-8") as f:
            f.write(json_str)

        lat_diff = (
            report.corrected_avg_latency_seconds
            - report.baseline_avg_latency_seconds
        )

        def format_metric_diff(base_val: float, corr_val: float, lower_is_better: bool = False) -> str:
            diff = corr_val - base_val
            diff_pts = diff * 100.0
            if base_val == 0.0:
                if corr_val == 0.0:
                    return "0.00 percentage points (N/A relative)"
                elif diff > 0:
                    return f"Increase from {base_val:.2%} to {corr_val:.2%} (+{diff_pts:.2f} percentage points, N/A relative)"
                else:
                    return f"Decrease from {base_val:.2%} to {corr_val:.2%} ({diff_pts:.2f} percentage points, N/A relative)"
            rel_pct = (diff / base_val) * 100.0
            direction = "reduction" if (lower_is_better and diff <= 0) else ("increase" if diff > 0 else "decrease")
            return f"{diff_pts:+.2f} percentage points ({rel_pct:+.1f}% {direction})"

        # Load golden dataset questions map for categories
        dataset = load_golden_dataset()
        q_cat_map = {q.id: q.category for q in dataset}

        halluc_str = format_metric_diff(report.baseline_avg_hallucination_rate, report.corrected_avg_hallucination_rate, lower_is_better=True)
        recall_str = format_metric_diff(report.baseline_avg_retrieval_recall, report.corrected_avg_retrieval_recall)
        cit_str = format_metric_diff(report.baseline_avg_citation_correctness, report.corrected_avg_citation_correctness)
        abs_str = format_metric_diff(report.baseline_appropriate_abstention_rate, report.corrected_appropriate_abstention_rate)

        # Render Markdown side-by-side comparison report
        md_lines = [
            "# Compass RAG Evaluation Report: Baseline vs. Self-Correcting RAG\n",
            f"**Run ID**: `{report.run_id}`  \n",
            f"**Timestamp**: `{report.timestamp}`  \n",
            f"**Total Benchmark Questions**: `{report.total_questions}` across 5 PS1 Categories\n",
            "## Summary Comparison Table\n",
            "| Metric | Baseline RAG | Self-Correcting RAG | Difference |",
            "| :--- | :---: | :---: | :---: |",
            f"| **Hallucination Rate** (Lower is better) | `{report.baseline_avg_hallucination_rate:.2%}` | `{report.corrected_avg_hallucination_rate:.2%}` | `{halluc_str}` |",
            f"| **Retrieval Recall** (Higher is better) | `{report.baseline_avg_retrieval_recall:.2%}` | `{report.corrected_avg_retrieval_recall:.2%}` | `{recall_str}` |",
            f"| **Citation Correctness** (Higher is better) | `{report.baseline_avg_citation_correctness:.2%}` | `{report.corrected_avg_citation_correctness:.2%}` | `{cit_str}` |",
            f"| **Appropriate Abstention Rate** (Higher is better) | `{report.baseline_appropriate_abstention_rate:.2%}` | `{report.corrected_appropriate_abstention_rate:.2%}` | `{abs_str}` |",
            f"| **Average Latency per Question** | `{report.baseline_avg_latency_seconds:.3f}s` | `{report.corrected_avg_latency_seconds:.3f}s` | `{lat_diff:+.3f}s` |",
            "\n## Per-Question Granular Results\n",
            "| QID | Category | Baseline Status | Corrected Status | Baseline Halluc | Corrected Halluc | Baseline Answer | Corrected Answer |",
            "| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- |",
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
            if qid not in qmap_corr:
                raise KeyError(f"Corrected evaluation result missing for question {qid}")
            c_res = qmap_corr[qid]
            cat = q_cat_map.get(qid, "General")
            b_ans = b_res.answer.replace("\n", " ")[:60]
            c_ans = c_res.answer.replace("\n", " ")[:60]
            md_lines.append(
                f"| **{qid}** | `{cat}` | `{b_res.confidence_status}` | "
                f"`{c_res.confidence_status}` | `{b_res.hallucination_rate:.1%}` | "
                f"`{c_res.hallucination_rate:.1%}` | {b_ans}... | {c_ans}... |"
            )

        md_content = "\n".join(md_lines)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        with open(root_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info("Evaluation reports exported to %s and %s", root_json_path, root_md_path)
        return root_json_path, root_md_path
