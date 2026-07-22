# Compass RAG Evaluation Report: Baseline vs. Self-Correcting RAG

**Run ID**: `28134421-4fb5-4864-a4f8-7b8eae64ea25`  
**Timestamp**: `2026-07-22T14:30:07.203770+00:00`  
**Total Benchmark Questions**: `12` across 5 PS1 Categories

## Summary Comparison Table

| Metric | Baseline RAG | Self-Correcting RAG | Improvement / Difference |
| :--- | :---: | :---: | :---: |
| **Hallucination Rate** (Lower is better) | `9.74%` | `15.99%` | **+-64.2%** reduction |
| **Retrieval Recall** (Higher is better) | `82.41%` | `82.41%` | **+0.0%** relative |
| **Citation Correctness** (Higher is better) | `0.00%` | `0.00%` | **+0.0%** relative |
| **Appropriate Abstention Rate** (Higher is better) | `75.00%` | `75.00%` | **+0.0%** relative |
| **Average Latency per Question** | `0.266s` | `10.313s` | `+10.047s` |

## Per-Question Granular Results

| QID | Category | Baseline Status | Corrected Status | Baseline Halluc | Corrected Halluc |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Q1** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `14.3%` | `14.3%` |
| **Q2** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `5.6%` | `5.6%` |
| **Q3** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `0.0%` | `0.0%` |
| **Q4** | `VERIFIED` | `VERIFIED` | `LOW_CONFIDENCE` | `25.0%` | `100.0%` |
| **Q5** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `14.3%` | `14.3%` |
| **Q6** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `0.0%` | `0.0%` |
| **Q7** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `6.7%` | `6.7%` |
| **Q8** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `16.7%` | `16.7%` |
| **Q9** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `5.3%` | `5.3%` |
| **Q10** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `0.0%` | `0.0%` |
| **Q11** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `12.5%` | `12.5%` |
| **Q12** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `16.7%` | `16.7%` |