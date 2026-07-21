FROM python:3.12-slim

WORKDIR /app

# Install system dependencies needed for PyMuPDF / report rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy workspace configuration and shared library
COPY pyproject.toml uv.lock ./
COPY shared/ ./shared/
COPY services/retrieval/ ./services/retrieval/
COPY services/correction/ ./services/correction/
COPY services/generation/ ./services/generation/
COPY services/evaluation/ ./services/evaluation/

# Install dependencies using uv
RUN uv sync --frozen --package compass-rag-evaluation

# Expose evaluation API port
EXPOSE 8005

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]
