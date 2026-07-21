/**
 * Shared TypeScript definitions matching backend domain models.
 */

export type ConfidenceStatus = "VERIFIED" | "CLARIFICATION_NEEDED" | "LOW_CONFIDENCE";

export interface DocumentMetadata {
  document_id: string;
  source: string;
  page_number: number;
  author?: string;
  section_title?: string;
  effective_date?: string;
  version?: string;
}

export interface Citation {
  chunk_id: string;
  document_id: string;
  source: string;
  page_number: number;
  quote_snippet: string;
}

export interface IngestedDocument {
  id: string;
  job_id: string;
  name: string;
  size_bytes: number;
  status: "PROCESSING" | "OCR_IN_PROGRESS" | "NEEDS_REVIEW" | "INDEXED" | "FAILED";
  progress: number;
  chunks_indexed?: number;
  created_at: string;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  confidence_status: ConfidenceStatus;
  confidence_score: number;
  reasoning?: string;
  latency_ms: number;
  trace_id?: string;
}

export interface OcrReviewItem {
  page_id: string;
  document_id: string;
  document_name: string;
  page_number: number;
  confidence_score: number;
  scanned_image_url: string;
  extracted_text: string;
  flagged_reason: string;
}

export interface EvaluationMetric {
  question_id: string;
  category: "Directly Answerable" | "OCR Dependent" | "Contradictory Policy" | "Ambiguous" | "Unanswerable";
  question: string;
  baseline_hallucinated: boolean;
  corrected_hallucinated: boolean;
  baseline_citation_correct: boolean;
  corrected_citation_correct: boolean;
  baseline_latency_ms: number;
  corrected_latency_ms: number;
}
