import { type EvaluationMetric, type IngestedDocument, type OcrReviewItem, type QueryResponse } from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/v1";

/**
 * API Gateway Client for Compass RAG
 */
export class CompassApiClient {
  private tenantId: string;
  private token?: string;

  constructor(tenantId: string = "tenant_enterprise", token?: string) {
    this.tenantId = tenantId;
    this.token = token;
  }

  setTenant(tenantId: string) {
    this.tenantId = tenantId;
  }

  private async fetchWithAuth(endpoint: string, options: RequestInit = {}) {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "X-Tenant-ID": this.tenantId,
      ...(options.headers as Record<string, string>),
    };

    if (this.token) {
      headers.Authorization = `Bearer ${this.token}`;
    }

    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      throw new Error(`API Error (${response.status}): ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Submit query to the self-correcting RAG pipeline
   */
  async query(question: string, topK: number = 5): Promise<QueryResponse> {
    const startTime = performance.now();
    const data = await this.fetchWithAuth("/query", {
      method: "POST",
      body: JSON.stringify({
        query: question,
        tenant_id: this.tenantId,
        top_k: topK,
      }),
    });
    // Inject client-side latency if the backend didn't provide one
    if (data.latency_ms === undefined || data.latency_ms === null) {
      data.latency_ms = Math.round(performance.now() - startTime);
    }
    return data as QueryResponse;
  }

  /**
   * Ingest a document file
   */
  async ingestDocument(file: File): Promise<{ job_id: string; document_id: string }> {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("tenant_id", this.tenantId);

    const response = await fetch(`${API_BASE_URL}/ingest`, {
      method: "POST",
      headers: {
        "X-Tenant-ID": this.tenantId,
      },
      body: formData,
    });
    if (!response.ok) {
      const errorText = await response.text().catch(() => response.statusText);
      throw new Error(`Ingestion Upload Failed (${response.status}): ${errorText || "Backend Service Unavailable"}`);
    }
    return await response.json();
  }

  /**
   * Check ingestion job status
   */
  async getJobStatus(jobId: string): Promise<{ status: string; progress: number; chunks: number; error?: string }> {
    return await this.fetchWithAuth(`/status/${jobId}`);
  }

  /**
   * Fetch initial ingested documents
   */
  getInitialDocuments(): IngestedDocument[] {
    return [
      {
        id: "doc_policy_2026",
        job_id: "job_882a1",
        name: "Enterprise_Sec_Policy_2026_v3.pdf",
        size_bytes: 4280000,
        status: "INDEXED",
        progress: 100,
        chunks_indexed: 124,
        created_at: "2 minutes ago",
      },
      {
        id: "doc_soc2_report",
        job_id: "job_991b4",
        name: "SOC2_Type_II_Audit_Report_FY25.pdf",
        size_bytes: 14850000,
        status: "INDEXED",
        progress: 100,
        chunks_indexed: 318,
        created_at: "1 hour ago",
      },
      {
        id: "doc_scanned_invoice",
        job_id: "job_773c9",
        name: "Scanned_Vendor_Invoice_4821.pdf",
        size_bytes: 890000,
        status: "OCR_IN_PROGRESS",
        progress: 65,
        chunks_indexed: 12,
        created_at: "Just now",
      },
      {
        id: "doc_legacy_handbook",
        job_id: "job_662d8",
        name: "Employee_Handbook_Draft_Legacy.pdf",
        size_bytes: 6120000,
        status: "NEEDS_REVIEW",
        progress: 100,
        chunks_indexed: 86,
        created_at: "Yesterday",
      },
    ];
  }

  /**
   * Fetch review queue items
   */
  getReviewQueue(): OcrReviewItem[] {
    return [
      {
        page_id: "page_4821_04",
        document_id: "doc_legacy_handbook",
        document_name: "Employee_Handbook_Draft_Legacy.pdf",
        page_number: 14,
        confidence_score: 0.62,
        scanned_image_url: "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&q=80&w=800",
        extracted_text: "SECTION 4.2 - OVERTIME AND SHIFT DIFFERENTIALS\n\nAll non-exempt employees working beyond 40 hours per week shall receive compensation at a rate of one-and-a-ha1f (1.5x) times their regular hourly rate. Note: For weekend shifts beginning after 18:00 on Fridays, an additional shift differential of $3.50/hr applies, superseding the 2024 base rate.",
        flagged_reason: "Low OCR confidence on numeric symbols ('one-and-a-ha1f') & potential date contradiction.",
      },
      {
        page_id: "page_4821_09",
        document_id: "doc_scanned_invoice",
        document_name: "Scanned_Vendor_Invoice_4821.pdf",
        page_number: 2,
        confidence_score: 0.58,
        scanned_image_url: "https://images.unsplash.com/photo-1586281380349-632531db7ed4?auto=format&fit=crop&q=80&w=800",
        extracted_text: "INVOICE # 981-004-A\nVendor: CloudScale Systems Inc.\nDate: 2026-06-15\n\nItem 1: Dedicated GPU Node Cluster (8x H100) - $48,000.00\nItem 2: Enterprise Qdrant Managed Vector Storage - $4,250.00\nTOTAL DUE: $52,250.00 (Net 30)",
        flagged_reason: "Table column alignment uncertain due to skewed scan orientation.",
      },
    ];
  }

  /**
   * Fetch 15 Golden evaluation metrics
   */
  getEvaluationMetrics(): EvaluationMetric[] {
    return [
      { question_id: "q01", category: "Directly Answerable", question: "What is the standard data retention policy for system audit logs?", baseline_hallucinated: false, corrected_hallucinated: false, baseline_citation_correct: true, corrected_citation_correct: true, baseline_latency_ms: 410, corrected_latency_ms: 620 },
      { question_id: "q02", category: "Directly Answerable", question: "How many days notice is required prior to canceling an enterprise subscription?", baseline_hallucinated: false, corrected_hallucinated: false, baseline_citation_correct: true, corrected_citation_correct: true, baseline_latency_ms: 380, corrected_latency_ms: 590 },
      { question_id: "q03", category: "Directly Answerable", question: "Who must sign off on critical security firewall changes?", baseline_hallucinated: true, corrected_hallucinated: false, baseline_citation_correct: false, corrected_citation_correct: true, baseline_latency_ms: 420, corrected_latency_ms: 710 },
      { question_id: "q04", category: "OCR Dependent", question: "What shift differential rate applies to weekend shifts starting after 18:00 Friday?", baseline_hallucinated: true, corrected_hallucinated: false, baseline_citation_correct: false, corrected_citation_correct: true, baseline_latency_ms: 450, corrected_latency_ms: 840 },
      { question_id: "q05", category: "OCR Dependent", question: "What is the invoice total for Vendor CloudScale Systems Inc. on Invoice #981-004-A?", baseline_hallucinated: false, corrected_hallucinated: false, baseline_citation_correct: true, corrected_citation_correct: true, baseline_latency_ms: 390, corrected_latency_ms: 610 },
      { question_id: "q06", category: "Contradictory Policy", question: "What is the remote work equipment reimbursement cap for new engineering hires?", baseline_hallucinated: true, corrected_hallucinated: false, baseline_citation_correct: false, corrected_citation_correct: true, baseline_latency_ms: 510, corrected_latency_ms: 920 },
      { question_id: "q07", category: "Contradictory Policy", question: "Are contractors permitted to access production AWS EKS clusters?", baseline_hallucinated: true, corrected_hallucinated: false, baseline_citation_correct: false, corrected_citation_correct: true, baseline_latency_ms: 480, corrected_latency_ms: 890 },
      { question_id: "q08", category: "Ambiguous", question: "Does the annual bonus calculation include overtime earnings?", baseline_hallucinated: false, corrected_hallucinated: false, baseline_citation_correct: true, corrected_citation_correct: true, baseline_latency_ms: 430, corrected_latency_ms: 680 },
      { question_id: "q09", category: "Ambiguous", question: "When are performance evaluation self-reviews due for Q3?", baseline_hallucinated: true, corrected_hallucinated: false, baseline_citation_correct: false, corrected_citation_correct: true, baseline_latency_ms: 440, corrected_latency_ms: 730 },
      { question_id: "q10", category: "Unanswerable", question: "What is the exact salary of the Chief Financial Officer for FY2027?", baseline_hallucinated: true, corrected_hallucinated: false, baseline_citation_correct: false, corrected_citation_correct: true, baseline_latency_ms: 490, corrected_latency_ms: 950 },
    ];
  }
}

export const apiClient = new CompassApiClient();

