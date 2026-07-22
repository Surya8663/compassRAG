# Compass RAG Evaluation Report: Baseline vs. Self-Correcting RAG

**Run ID**: `ccaaec9a-de6f-43f6-84bb-79e673968444`  

**Timestamp**: `2026-07-22T17:37:32.854943+00:00`  

**Total Benchmark Questions**: `12` across 5 PS1 Categories

## Summary Comparison Table

| Metric | Baseline RAG | Self-Correcting RAG | Difference |
| :--- | :---: | :---: | :---: |
| **Hallucination Rate** (Lower is better) | `0.00%` | `18.12%` | **0.0%** increase |
| **Retrieval Recall** (Higher is better) | `0.00%` | `83.33%` | `+0.0%` relative |
| **Citation Correctness** (Higher is better) | `0.00%` | `83.33%` | `+0.0%` relative |
| **Appropriate Abstention Rate** (Higher is better) | `25.00%` | `91.67%` | `+266.7%` relative |
| **Average Latency per Question** | `0.001s` | `0.271s` | `+0.270s` |

## Per-Question Granular Results

| QID | Category | Baseline Status | Corrected Status | Baseline Halluc | Corrected Halluc | Baseline Answer | Corrected Answer |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Q1** | `directly_answerable` | `UNVERIFIED` | `VERIFIED` | `0.0%` | `16.7%` | No verified context available to answer the query.... | Every full-time employee is entitled to a $1,500 annual hard... |
| **Q2** | `directly_answerable` | `UNVERIFIED` | `VERIFIED` | `0.0%` | `7.7%` | No verified context available to answer the query.... | All staff must be available for synchronous collaboration be... |
| **Q3** | `directly_answerable` | `UNVERIFIED` | `VERIFIED` | `0.0%` | `11.1%` | No verified context available to answer the query.... | Employees receive 20 days of paid annual leave per year. Up ... |
| **Q4** | `directly_answerable` | `UNVERIFIED` | `VERIFIED` | `0.0%` | `25.0%` | No verified context available to answer the query.... | The company provides a $75 monthly wellness stipend for gym ... |
| **Q5** | `ocr_dependent` | `UNVERIFIED` | `VERIFIED` | `0.0%` | `20.0%` | No verified context available to answer the query.... | OFFICIAL REIMBURSEMENT INVOICE / RECEIPT VENDOR Apex Caterin... |
| **Q6** | `ocr_dependent` | `UNVERIFIED` | `VERIFIED` | `0.0%` | `16.7%` | No verified context available to answer the query.... | OFFICIAL REIMBURSEMENT INVOICE / RECEIPT VENDOR Apex Caterin... |
| **Q7** | `unanswerable` | `UNVERIFIED` | `VERIFIED` | `0.0%` | `0.0%` | No verified context available to answer the query.... | There is insufficient information in the provided context to... |
| **Q8** | `unanswerable` | `UNVERIFIED` | `VERIFIED` | `0.0%` | `0.0%` | No verified context available to answer the query.... | There is insufficient information in the provided context to... |
| **Q9** | `contradictory_document` | `UNVERIFIED` | `VERIFIED` | `0.0%` | `5.0%` | No verified context available to answer the query.... | 1. Hotel Accommodation: When traveling on official business,... |
| **Q10** | `contradictory_document` | `UNVERIFIED` | `VERIFIED` | `0.0%` | `5.3%` | No verified context available to answer the query.... | 2. Meal Per Diem: International meal per diem is capped at $... |
| **Q11** | `ambiguous` | `UNVERIFIED` | `VERIFIED` | `0.0%` | `100.0%` | No verified context available to answer the query.... | Employees may work remotely on ad-hoc days subject to manage... |
| **Q12** | `directly_answerable` | `UNVERIFIED` | `VERIFIED` | `0.0%` | `10.0%` | No verified context available to answer the query.... | · Built PeopleGPT / Aura AI Agent · RAG pipeline over 13,000... |