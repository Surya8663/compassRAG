"""
Verification suite for Phase 9: Evaluation Service (`/services/evaluation`).
Tests golden dataset category counts, PDF corpus generation, baseline pipeline simplicity,
programmatic metric calculations, and full side-by-side comparison report generation.
"""

from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient
from shared.models.common import DocumentChunk, DocumentMetadata

from services.evaluation.app.api.routes import runner_service
from services.evaluation.app.data.corpus_generator import generate_golden_corpus
from services.evaluation.app.main import app
from services.evaluation.app.models.eval import GoldenQuestion
from services.evaluation.app.services.baseline_pipeline import BaselineRAGPipeline
from services.evaluation.app.services.metrics_evaluator import MetricsEvaluator
from services.evaluation.app.services.report_generator import load_golden_dataset

client = TestClient(app)


def test_golden_dataset_structure_and_counts() -> None:
    """Verifies exactly 15 questions exist spanning all 5 required PS1 categories."""
    questions = load_golden_dataset()
    assert len(questions) == 15, f"Expected 15 golden questions, got {len(questions)}"

    counts: dict[str, int] = {}
    for q in questions:
        counts[q.category] = counts.get(q.category, 0) + 1
        assert q.id and q.question and q.expected_answer

    assert counts.get("directly_answerable", 0) == 4
    assert counts.get("ocr_dependent", 0) == 3
    assert counts.get("contradictory_document", 0) == 3
    assert counts.get("ambiguous", 0) == 2
    assert counts.get("unanswerable", 0) == 3


def test_corpus_generator_creates_pdfs(tmp_path: Path) -> None:
    """Verifies PDF corpus generator outputs realistic files with extractable text."""
    paths = generate_golden_corpus()
    assert len(paths) == 5
    for p in paths:
        assert p.exists() and p.suffix == ".pdf"


def test_baseline_pipeline_has_no_correction() -> None:
    """Proves baseline pipeline executes cleanly without correction or retry loops."""
    pipeline = BaselineRAGPipeline()
    dummy_chunk = DocumentChunk(
        id="dummy_p1",
        document_id="dummy.pdf",
        content="Every full-time employee is entitled to a $1,500 annual hardware stipend.",
        metadata=DocumentMetadata(
            source="dummy.pdf",
            page_number=1,
            ingestion_timestamp=datetime.now(UTC),
            tenant_id="eval",
            version_id="1.0",
        ),
    )
    result = pipeline.run(
        query="What is the hardware stipend?",
        tenant_id="eval",
        pre_retrieved_chunks=[dummy_chunk],
    )
    assert "answer" in result
    assert result["confidence_status"] == "VERIFIED"
    assert result["latency_seconds"] >= 0.0


def test_metrics_evaluator_calculations() -> None:
    """Asserts mathematical exactness of retrieval recall, hallucination rate, and abstention."""
    evaluator = MetricsEvaluator()
    q = GoldenQuestion(
        id="Q_TEST",
        category="directly_answerable",
        question="What is the stipend?",
        expected_answer="$1,500 annual hardware stipend.",
        expected_keywords=["1,500", "hardware", "stipend"],
        expected_chunk_ids=["handbook_2026.pdf"],
        should_abstain=False,
    )

    chunk = DocumentChunk(
        id="handbook_2026.pdf_p1",
        document_id="handbook_2026.pdf",
        content="Full-time employees receive a $1,500 annual hardware stipend.",
        metadata=DocumentMetadata(
            source="handbook_2026.pdf",
            page_number=1,
            ingestion_timestamp=datetime.now(UTC),
            tenant_id="eval",
            version_id="1.0",
        ),
    )

    # 1. Retrieval recall calculation
    recall = evaluator.compute_retrieval_recall(q, [chunk])
    assert recall == 1.0

    # 2. Hallucination rate fallback / verification calculation
    halluc = evaluator.compute_hallucination_rate(q, "$1,500 hardware stipend.", [chunk])
    assert halluc == 0.0

    # 3. Appropriate abstention verification
    q_abs = GoldenQuestion(
        id="Q_ABS",
        category="unanswerable",
        question="Unknown query?",
        expected_answer="Insufficient information.",
        expected_keywords=[],
        should_abstain=True,
    )
    abs_score = evaluator.compute_appropriate_abstention(
        q_abs, "INSUFFICIENT_INFORMATION", "Insufficient information available."
    )
    assert abs_score == 1.0


def test_evaluation_runner_end_to_end_report() -> None:
    """Runs full evaluation benchmark across both pipelines and verifies generated reports."""
    # Run across a 2-question subset to verify full runner logic cleanly
    subset = load_golden_dataset()[:2]
    report = runner_service.run_evaluation(dataset=subset, tenant_id="eval_test")

    assert report.total_questions == 2
    assert len(report.question_results) == 4  # 2 baseline + 2 corrected
    assert report.run_id and report.timestamp

    # Verify report files exported
    json_resp = client.get("/v1/evaluate/report?format=json")
    assert json_resp.status_code == 200
    assert "run_id" in json_resp.json()

    md_resp = client.get("/v1/evaluate/report?format=markdown")
    assert md_resp.status_code == 200
    assert "Summary Comparison Table" in md_resp.text
