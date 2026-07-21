FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /workspace

# Enable bytecode compilation and non-interactive sync
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy workspace pyproject.toml, uv.lock, shared module, and all services
COPY pyproject.toml uv.lock ./
COPY shared/ ./shared/
COPY services/ ./services/

# Sync dependencies for this package without dev deps
RUN uv sync --package compass-rag-retrieval --no-dev

# Set working directory to the service root so that `app.main:app` resolves cleanly
WORKDIR /workspace/services/retrieval

EXPOSE 8002

CMD ["uv", "run", "--package", "compass-rag-retrieval", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
