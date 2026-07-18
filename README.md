# Compass RAG Monorepo

Production-grade self-correcting Retrieval-Augmented Generation (RAG) pipeline monorepo managed with `uv` workspaces and `docker-compose`.

## Architecture Overview

The system is structured as independent microservices communicating via HTTP (`/services/*`) backed by shared domain schemas and configuration (`/shared`):
- **API Gateway (`:8000`)**: Public entrypoint routing to internal services.
- **Ingestion (`:8001`)**: Document parsing, chunking, and embedding indexing.
- **Retrieval (`:8002`)**: Dense (Qdrant) and sparse (Elasticsearch) chunk search.
- **Correction (`:8003`)**: Self-correcting evaluation and query refinement.
- **Generation (`:8004`)**: LLM response synthesis from verified context.

## Local Development Quickstart

1. Install `uv` (`pip install uv`).
2. Copy configuration:
   ```bash
   cp .env.example .env
   ```
3. Sync dependencies:
   ```bash
   uv sync
   ```
4. Boot local infrastructure and all services via Docker:
   ```bash
   docker compose -f infrastructure/docker-compose.yml up --build -d
   ```
5. Verify health checks across all services:
   ```bash
   uv run python scripts/test_health_checks.py
   ```
