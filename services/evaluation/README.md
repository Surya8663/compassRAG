# Compass RAG Evaluation Service (`/services/evaluation`)

Provides benchmark evaluation comparing our **Self-Correcting RAG Pipeline** against a **Baseline RAG Pipeline** across an authoritative 15-question Golden Dataset (`golden_dataset.yaml`).

## Features
- Programmatic PDF Corpus Generator (`corpus_generator.py`)
- 15-Question Golden Dataset spanning 5 categories (`directly_answerable`, `ocr_dependent`, `contradictory_document`, `ambiguous`, `unanswerable`)
- Side-by-side metric computation (hallucination rate, retrieval recall, citation correctness, appropriate abstention, latency)
- Automated comparison report generation in JSON and Markdown/HTML formats (`reports/`)
