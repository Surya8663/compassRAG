# Compass RAG Evaluation Report: Baseline vs. Self-Correcting RAG

**Run ID**: `8bd9be73-683d-48e7-a6d7-d214b7583d89`  

**Timestamp**: `2026-07-22T18:01:59.355668+00:00`  

**Total Benchmark Questions**: `12` across 5 PS1 Categories

## Summary Comparison Table

| Metric | Baseline RAG | Self-Correcting RAG | Difference |
| :--- | :---: | :---: | :---: |
| **Hallucination Rate** (Lower is better) | `22.02%` | `21.67%` | `-0.35 percentage points (-1.6% reduction)` |
| **Retrieval Recall** (Higher is better) | `83.33%` | `83.33%` | `+0.00 percentage points (+0.0% decrease)` |
| **Citation Correctness** (Higher is better) | `83.33%` | `83.33%` | `+0.00 percentage points (+0.0% decrease)` |
| **Appropriate Abstention Rate** (Higher is better) | `100.00%` | `100.00%` | `+0.00 percentage points (+0.0% decrease)` |
| **Average Latency per Question** | `0.089s` | `0.104s` | `+0.015s` |

## Per-Question Granular Results

| QID | Category | Baseline Status | Corrected Status | Baseline Halluc | Corrected Halluc | Baseline Answer | Corrected Answer |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Q1** | `directly_answerable` | `VERIFIED` | `VERIFIED` | `0.0%` | `0.0%` | Every full-time employee is entitled to a $1,500 annual hard... | Every full-time employee is entitled to a $1,500 annual hard... |
| **Q2** | `directly_answerable` | `VERIFIED` | `VERIFIED` | `0.0%` | `0.0%` | All staff must be available for synchronous collaboration be... | All staff must be available for synchronous collaboration be... |
| **Q3** | `directly_answerable` | `VERIFIED` | `VERIFIED` | `25.0%` | `25.0%` | Employees receive 20 days of paid annual leave per year. Up ... | Employees receive 20 days of paid annual leave per year. Up ... |
| **Q4** | `directly_answerable` | `VERIFIED` | `VERIFIED` | `0.0%` | `0.0%` | The company provides a $75 monthly wellness stipend for gym ... | The company provides a $75 monthly wellness stipend for gym ... |
| **Q5** | `ocr_dependent` | `VERIFIED` | `VERIFIED` | `33.3%` | `33.3%` | OFFICIAL REIMBURSEMENT INVOICE / RECEIPT VENDOR Apex Caterin... | OFFICIAL REIMBURSEMENT INVOICE / RECEIPT VENDOR Apex Caterin... |
| **Q6** | `ocr_dependent` | `VERIFIED` | `VERIFIED` | `33.3%` | `33.3%` | OFFICIAL REIMBURSEMENT INVOICE / RECEIPT VENDOR Apex Caterin... | OFFICIAL REIMBURSEMENT INVOICE / RECEIPT VENDOR Apex Caterin... |
| **Q7** | `unanswerable` | `LOW_CONFIDENCE` | `LOW_CONFIDENCE` | `0.0%` | `0.0%` | There is insufficient information in the provided context to... | There is insufficient information in the provided context to... |
| **Q8** | `unanswerable` | `LOW_CONFIDENCE` | `LOW_CONFIDENCE` | `0.0%` | `0.0%` | There is insufficient information in the provided context to... | There is insufficient information in the provided context to... |
| **Q9** | `contradictory_document` | `VERIFIED` | `VERIFIED` | `33.3%` | `33.3%` | 1. Hotel Accommodation: Due to increased hospitality costs, ... | 1. Hotel Accommodation: Due to increased hospitality costs, ... |
| **Q10** | `contradictory_document` | `VERIFIED` | `VERIFIED` | `25.0%` | `25.0%` | 2. Meal Per Diem: International meal per diem is updated to ... | 2. Meal Per Diem: International meal per diem is updated to ... |
| **Q11** | `ambiguous` | `CLARIFICATION_NEEDED` | `CLARIFICATION_NEEDED` | `100.0%` | `100.0%` | The documents do not establish an automatic every-Friday ent... | The documents do not establish an automatic every-Friday ent... |
| **Q12** | `directly_answerable` | `VERIFIED` | `VERIFIED` | `14.3%` | `10.0%` | · Built PeopleGPT / Aura AI Agent · RAG pipeline over 13,000... | · Built PeopleGPT / Aura AI Agent · RAG pipeline over 13,000... |