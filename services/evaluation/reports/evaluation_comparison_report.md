# Compass RAG Evaluation Report: Baseline vs. Self-Correcting RAG

**Run ID**: `6017113d-bfa1-4bd0-a8ad-fcc60778bc43`  
**Timestamp**: `2026-07-21T14:22:19.248324+00:00`  
**Total Benchmark Questions**: `15` across 5 PS1 Categories

## Summary Comparison Table

| Metric | Baseline RAG | Self-Correcting RAG | Improvement / Difference |
| :--- | :---: | :---: | :---: |
| **Hallucination Rate** (Lower is better) | `34.44%` | `0.00%` | **+100.0%** reduction |
| **Retrieval Recall** (Higher is better) | `80.00%` | `80.00%` | **+0.0%** relative |
| **Citation Correctness** (Higher is better) | `100.00%` | `100.00%` | **+0.0%** relative |
| **Appropriate Abstention Rate** (Higher is better) | `66.67%` | `66.67%` | **+0.0%** relative |
| **Average Latency per Question** | `0.000s` | `0.042s` | `+0.042s` |

## Per-Question Granular Results

| QID | Category | Baseline Status | Corrected Status | Baseline Halluc | Corrected Halluc |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Q1** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `50.0%` | `0.0%` |
| **Q2** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `50.0%` | `0.0%` |
| **Q3** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `50.0%` | `0.0%` |
| **Q4** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `50.0%` | `0.0%` |
| **Q5** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `25.0%` | `0.0%` |
| **Q6** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `25.0%` | `0.0%` |
| **Q7** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `25.0%` | `0.0%` |
| **Q8** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `16.7%` | `0.0%` |
| **Q9** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `16.7%` | `0.0%` |
| **Q10** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `16.7%` | `0.0%` |
| **Q11** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `33.3%` | `0.0%` |
| **Q12** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `33.3%` | `0.0%` |
| **Q13** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `50.0%` | `0.0%` |
| **Q14** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `50.0%` | `0.0%` |
| **Q15** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `25.0%` | `0.0%` |