FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /workspace

# Install system dependencies needed for PyMuPDF / report rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Enable bytecode compilation and non-interactive sync
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy workspace pyproject.toml, uv.lock, shared module, and all services
COPY pyproject.toml uv.lock ./
COPY shared/ ./shared/
COPY services/ ./services/

# Sync dependencies for this package without dev deps
RUN uv sync --package compass-rag-evaluation --no-dev

WORKDIR /workspace/services/evaluation

# Expose evaluation API port
EXPOSE 8005

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]
