# Compass RAG Evaluation Report: Baseline vs. Self-Correcting RAG

**Run ID**: `b45b5f44-029e-4c7c-a36e-b13ee47205fb`  

**Timestamp**: `2026-07-22T16:01:37.037573+00:00`  

**Total Benchmark Questions**: `12` across 5 PS1 Categories

## Summary Comparison Table

| Metric | Baseline RAG | Self-Correcting RAG | Difference |
| :--- | :---: | :---: | :---: |
| **Hallucination Rate** (Lower is better) | `2.71%` | `0.00%` | **100.0%** reduction |
| **Retrieval Recall** (Higher is better) | `65.74%` | `0.00%` | `-100.0%` relative |
| **Citation Correctness** (Higher is better) | `100.00%` | `0.00%` | `-100.0%` relative |
| **Appropriate Abstention Rate** (Higher is better) | `75.00%` | `0.00%` | `-100.0%` relative |
| **Average Latency per Question** | `1.965s` | `0.000s` | `-1.965s` |

## Per-Question Granular Results

| QID | Category | Baseline Status | Corrected Status | Baseline Halluc | Corrected Halluc | Baseline Answer | Corrected Answer |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Q1** | `directly_answerable` | `VERIFIED` | `VERIFIED` | `0.0%` | `0.0%` | Every full-time employee is entitled to a $1,500 annual hard... | Every full-time employee is entitled to a $1,500 annual hard... |
| **Q2** | `directly_answerable` | `VERIFIED` | `VERIFIED` | `0.0%` | `0.0%` | All staff must be available for synchronous collaboration be... | All staff must be available for synchronous collaboration be... |
| **Q3** | `directly_answerable` | `VERIFIED` | `VERIFIED` | `0.0%` | `0.0%` | Employees receive 20 days of paid annual leave per year. Up ... | Employees receive 20 days of paid annual leave per year. Up ... |
| **Q4** | `directly_answerable` | `VERIFIED` | `VERIFIED` | `0.0%` | `0.0%` | The company provides a $75 monthly wellness stipend for gym ... | The company provides a $75 monthly wellness stipend for gym ... |
| **Q5** | `ocr_dependent` | `VERIFIED` | `VERIFIED` | `0.0%` | `0.0%` | 1. Hotel Accommodation: Due to increased hospitality costs, ... | 1. Hotel Accommodation: Due to increased hospitality costs, ... |
| **Q6** | `ocr_dependent` | `VERIFIED` | `VERIFIED` | `0.0%` | `0.0%` | Bangalore, India | +91 86672 34480 | iamsurya195@gmail.com [... | Bangalore, India | +91 86672 34480 | iamsurya195@gmail.com [... |
| **Q7** | `unanswerable` | `VERIFIED` | `VERIFIED` | `0.0%` | `0.0%` | Company IT Equipment Allowance Policy [handbook_2026.pdf_p1]... | Company IT Equipment Allowance Policy [handbook_2026.pdf_p1]... |
| **Q8** | `unanswerable` | `VERIFIED` | `VERIFIED` | `0.0%` | `0.0%` | Bangalore, India | +91 86672 34480 | iamsurya195@gmail.com [... | Bangalore, India | +91 86672 34480 | iamsurya195@gmail.com [... |
| **Q9** | `contradictory_document` | `VERIFIED` | `VERIFIED` | `12.5%` | `12.5%` | allowable hotel accommodation expense is now $250 per night.... | allowable hotel accommodation expense is now $250 per night.... |
| **Q10** | `contradictory_document` | `VERIFIED` | `VERIFIED` | `20.0%` | `20.0%` | 2. Meal Per Diem: International meal per diem is capped at $... | 2. Meal Per Diem: International meal per diem is capped at $... |
| **Q11** | `ambiguous` | `VERIFIED` | `VERIFIED` | `0.0%` | `0.0%` | Employees may work remotely on ad-hoc days subject to manage... | Employees may work remotely on ad-hoc days subject to manage... |
| **Q12** | `directly_answerable` | `VERIFIED` | `VERIFIED` | `0.0%` | `0.0%` | HiDevs Bangalore, India (Hybrid) [G_Surya_Resume.pdf_p1] · B... | HiDevs Bangalore, India (Hybrid) [G_Surya_Resume.pdf_p1] · B... |