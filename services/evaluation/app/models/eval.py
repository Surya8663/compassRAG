"""
Pydantic data models for the Evaluation Service (`services/evaluation`).
Defines golden dataset schema, individual metric records, and summary reports.
"""

from typing import Literal

from pydantic import BaseModel, Field


class GoldenQuestion(BaseModel):
    """Schema for a single evaluation question from `golden_dataset.yaml`."""

    id: str = Field(..., description="Unique question ID (e.g. Q1)")
    category: Literal[
        "directly_answerable",
        "ocr_dependent",
        "contradictory_document",
        "ambiguous",
        "unanswerable",
    ] = Field(..., description="PS1 classification category")
    question: str = Field(..., description="The user query text")
    expected_answer: str = Field(..., description="Human ground-truth answer")
    expected_keywords: list[str] = Field(
        default_factory=list,
        description="Mandatory keywords/facts required in answer or retrieval",
    )
    expected_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs that contain the ground truth claim",
    )
    should_abstain: bool = Field(
        default=False,
        description="True if pipeline should return UNVERIFIED/clarification",
    )


class QuestionEvaluationResult(BaseModel):
    """Evaluation metrics recorded for a single question under a specific pipeline."""

    question_id: str = Field(..., description="Question ID (Q1..Q15)")
    pipeline_type: Literal["baseline", "corrected"] = Field(
        ..., description="Pipeline path evaluated"
    )
    answer: str = Field(..., description="Synthesized output answer")
    confidence_status: str = Field(
        ..., description="Pipeline status (VERIFIED, UNVERIFIED, etc.)"
    )
    hallucination_rate: float = Field(
        ..., ge=0.0, le=1.0, description="1.0 minus verified claim fraction"
    )
    retrieval_recall: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of required facts retrieved"
    )
    citation_correctness: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of citations entailing claim"
    )
    appropriate_abstention: float = Field(
        ..., ge=0.0, le=1.0, description="1.0 if correct abstention behavior, else 0.0"
    )
    latency_seconds: float = Field(
        ..., ge=0.0, description="Wall-clock execution time in seconds"
    )
    citations_cited: list[str] = Field(
        default_factory=list, description="List of chunk IDs cited in answer"
    )


class EvaluationSummaryReport(BaseModel):
    """Aggregate comparison report between Baseline RAG and Self-Correcting RAG."""

    run_id: str = Field(..., description="Unique run identifier")
    timestamp: str = Field(..., description="Execution completion timestamp")
    total_questions: int = Field(..., description="Total questions run (15)")
    baseline_avg_hallucination_rate: float = Field(
        ..., description="Baseline average hallucination rate"
    )
    corrected_avg_hallucination_rate: float = Field(
        ..., description="Corrected average hallucination rate"
    )
    hallucination_improvement_pct: float = Field(
        ..., description="Percentage improvement in hallucination reduction"
    )
    baseline_avg_retrieval_recall: float = Field(
        ..., description="Baseline average retrieval recall"
    )
    corrected_avg_retrieval_recall: float = Field(
        ..., description="Corrected average retrieval recall"
    )
    retrieval_recall_improvement_pct: float = Field(
        ..., description="Percentage improvement in retrieval recall"
    )
    baseline_avg_citation_correctness: float = Field(
        ..., description="Baseline average citation correctness"
    )
    corrected_avg_citation_correctness: float = Field(
        ..., description="Corrected average citation correctness"
    )
    citation_correctness_improvement_pct: float = Field(
        ..., description="Percentage improvement in citation correctness"
    )
    baseline_appropriate_abstention_rate: float = Field(
        ..., description="Baseline appropriate abstention rate"
    )
    corrected_appropriate_abstention_rate: float = Field(
        ..., description="Corrected appropriate abstention rate"
    )
    abstention_improvement_pct: float = Field(
        ..., description="Percentage improvement in appropriate abstention"
    )
    baseline_avg_latency_seconds: float = Field(
        ..., description="Baseline average latency"
    )
    corrected_avg_latency_seconds: float = Field(
        ..., description="Corrected average latency"
    )
    question_results: list[QuestionEvaluationResult] = Field(
        default_factory=list, description="Per-question evaluation details"
    )
