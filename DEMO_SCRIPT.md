# Compass RAG: Live Demo & Rehearsal Script
**Demo Date:** July 25th  
**Product:** Compass RAG — Enterprise Document Intelligence & Self-Correcting RAG Platform  
**Presenter Guide:** Step-by-step rehearsed sequence, expected outcomes, and architectural highlights.

---

## 🎯 Executive Summary & Demo Objectives

During this live demonstration, you will showcase how **Compass RAG** elevates enterprise document search beyond basic "retrieve-and-generate" wrappers into a **trust-first, self-correcting agentic platform**. 

Unlike conventional RAG prototypes that hallucinate when faced with conflicting or missing data, Compass RAG implements a deterministic **LangGraph correction circuit** and a strict **semantic tri-state identity**:

1. **`VERIFIED` (Teal/Green):** Confident answers grounded in verified document citations with exact quote snippets.
2. **`CLARIFICATION_NEEDED` (Amber/Warning):** Autonomous detection of same-date or same-version document contradictions, prompting the user for guidance rather than arbitrarily picking a winner.
3. **`LOW_CONFIDENCE` (Muted Red/Critical):** Honest abstention circuit breaker when multi-hop query reformulations and keyword expansions cannot find verified evidence.

---

## 📋 Pre-Demo Checklist & Stack Initialization

Run these checks **15 minutes before going live** to ensure the entire multi-service stack is warm and responsive.

### 1. One-Command Stack Boot
Boot all 13 microservices, data stores, and monitoring platforms from the workspace root:
```powershell
docker compose up --build -d
```

### 2. Verify Service Health
Check that all core infrastructure and application containers are healthy:
```powershell
docker compose ps
```
Ensure the following endpoints are live:
- **API Gateway Health:** [http://localhost:8000/health](http://localhost:8000/health)
- **Next.js Enterprise Frontend:** [http://localhost:3000](http://localhost:3000)
- **Jaeger Distributed Tracing UI:** [http://localhost:16686](http://localhost:16686)
- **Prometheus Metrics:** [http://localhost:8000/metrics](http://localhost:8000/metrics)

### 3. Pre-Seed / Verify Evaluation Dataset
Ensure the golden document batch is generated and ready in `data/golden_corpus/`:
```powershell
python -m uv run pytest tests/test_e2e_demo.py -k test_demo_batch_generation -v
```

---

## 📚 The Evaluation Document Batch

We test against a realistic, mixed-quality enterprise PDF corpus containing **5 core documents** (`data/golden_corpus/`):

| Document Name | Version / Date | Key Subject Matter | Role in Demo |
| :--- | :--- | :--- | :--- |
| `remote_work_guidelines.pdf` | Version `1.0` (Same-Date) | Ad-hoc remote work limits (3 consecutive days) & approval rules | Baseline source & side A of contradiction |
| `remote_work_policy_update.pdf` | Version `1.0` (Same-Date) | Conflicting ad-hoc rules (5 consecutive days) & different approval flow | Side B of contradiction (triggers clarification) |
| `security_policy.pdf` | Version `2.1` | Data classification standards & incident reporting timelines | High-confidence technical reference |
| `travel_expense_policy.pdf` | Version `1.4` | Per diem rates, receipt thresholds, & reimbursement procedures | Multi-document cross-reference |
| `employee_handbook.pdf` | Version `3.0` | General benefits, paid time off, & workplace conduct | Broad organizational context |

> [!IMPORTANT]
> **The Contradiction Trap:** `remote_work_guidelines.pdf` and `remote_work_policy_update.pdf` both share **identical `version_id: "1.0"` and ingestion timestamps**. Because neither document supersedes the other temporally, conventional RAG pipelines silently pick whichever chunk scores slightly higher on vector similarity, leading to contradictory answers across queries. **Compass RAG catches this NLI conflict deterministically.**

---

## 🎬 Step-by-Step Live Rehearsal Guide

### Act I: Confident Grounded Citation (`VERIFIED`)
**Objective:** Demonstrate accuracy, exact page attribution, and visual polish on unambiguous queries.

1. **Navigate to Frontend:** Open [http://localhost:3000](http://localhost:3000) in the browser.
2. **Select Tenant & Persona:** Ensure top navigation bar displays **Tenant:** `demo_tenant` | **Role:** `Admin`.
3. **Input Prompt:**
   ```text
   What is the maximum number of consecutive ad-hoc remote work days allowed under the standard guidelines?
   ```
4. **Expected Outcome & Talking Points:**
   - **Visual State:** The UI transitions immediately to a vibrant **Teal/Green `VERIFIED` Badge**.
   - **Answer:** States that employees may work remotely on ad-hoc days subject to a **maximum of 3 consecutive business days** (or notes the exact policy clause).
   - **Citation Card:** Shows interactive citation badge citing `remote_work_guidelines.pdf (Page 1)`.
   - **Quote Snippet:** Expanding the citation reveals the exact verbatim text snippet matching the claim.
   - **Why this matters:** Highlight that every claim in the answer passed atomic decomposition and NLI entailment verification before the UI rendered it.

---

### Act II: The Agentic Clarification Loop (`CLARIFICATION_NEEDED`)
**Objective:** Show how Compass RAG protects enterprise trust when internal policies conflict.

1. **Input Prompt:**
   ```text
   Who has final approval authority for ad-hoc remote work days, and what are the exact departmental criteria?
   ```
2. **Expected Outcome & Talking Points:**
   - **Visual State:** The UI shifts to an **Amber/Warning `CLARIFICATION_NEEDED` Card** with distinct warning accent styling.
   - **Answer:** Instead of guessing between conflicting sources, the agent responds:
     > *"We found conflicting statements regarding ad-hoc remote work approval authority across documents with identical version metadata (`remote_work_guidelines.pdf` and `remote_work_policy_update.pdf`). One policy states approval requires direct manager discretion, while the other states rules vary by departmental operational requirements. Could you please clarify which document or scenario you would like us to follow?"*
   - **Citation Cards:** Both contradictory sources are displayed side-by-side.
   - **Why this matters:** Explain to the audience that in legal, financial, or HR settings, **hallucinating a compromise between two conflicting legal documents is a critical liability**. Compass RAG halts execution and prompts the human in the loop.

---

### Act III: Honest Abstention & Circuit Breaker (`LOW_CONFIDENCE`)
**Objective:** Demonstrate self-correction retries and honest abstention on ungrounded/unanswerable queries.

1. **Input Prompt:**
   ```text
   What is the personal mobile phone number of the Chief Executive Officer?
   ```
2. **Expected Outcome & Talking Points:**
   - **Visual State:** The UI displays a **Muted Red/Critical `LOW_CONFIDENCE` Status Banner**.
   - **Answer:** 
     > *"I cannot provide a definitive or verified answer to your query regarding the personal mobile phone number of the Chief Executive Officer. We conducted multiple search and reformulation attempts (including query expansion and keyword broadening), but the retrieved documentation did not meet our strict confidence and groundedness thresholds. Please verify the documentation or rephrase with additional specifics."*
   - **Citations:** `0 Citations` shown (or empty citation panel).
   - **Why this matters:** Mention that while standard LLMs often fabricate phone numbers or pull from training data, our self-correcting loop attempted `MAX_RETRIES` (3 reformulations), detected zero grounded evidence, and triggered the **circuit breaker** to prevent hallucination.

---

## 🕵️ Behind-the-Scenes: OpenTelemetry Tracing & Audit Trail

After completing the 3 visual acts in the UI, transition to [http://localhost:16686](http://localhost:16686) (**Jaeger UI**) to show the audience what happened under the hood:

1. **Select Service:** Choose `compass-rag-correction` or `compass-rag-gateway` from the Service dropdown and click **Find Traces**.
2. **Open the `CLARIFICATION` or `LOW_CONFIDENCE` Trace:**
3. **Highlight the Waterfall Stages:**
   - **`compass.retrieval.hybrid_search`:** Show the parallel execution of Qdrant (dense vectors) and Elasticsearch (BM25 sparse keywords) followed by Reciprocal Rank Fusion (RRF).
   - **`compass.correction.contradiction_check`:** Show the exact span where same-date metadata collision was identified and routed to the NLI cross-encoder.
   - **`compass.correction.groundedness_check`:** Point out the claim decomposition and entailment scoring span (`groundedness_score`).
4. **Show Prometheus Metrics:** Briefly open [http://localhost:8000/metrics](http://localhost:8000/metrics) to display real-time histograms (`compass_rag_pipeline_stage_duration_seconds`) and loop counters (`compass_rag_correction_retries_total`).

---

## 🛠️ Contingency & Troubleshooting Guide

If anything unexpected occurs during the live demonstration, use these quick commands:

| Scenario | Diagnosis / Quick Fix | Command |
| :--- | :--- | :--- |
| **UI seems unresponsive** | API Gateway container might be restarting | `docker compose logs -f api-gateway` |
| **Need to reset database state** | Re-sync local Qdrant/Postgres volumes | `docker compose down -v && docker compose up --build -d` |
| **Run verification suite CLI** | Execute exact demo assertions in terminal | `python -m uv run pytest tests/test_e2e_demo.py -v` |

---
*End of Script. Rehearse twice before live presentation.*
