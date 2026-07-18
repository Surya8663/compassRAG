FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /workspace

# Enable bytecode compilation and non-interactive sync
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy workspace pyproject.toml and shared module
COPY pyproject.toml ./
COPY shared/ ./shared/

# Copy target service
COPY services/correction/ ./services/correction/

# Sync dependencies for this package without dev deps
RUN uv sync --package compass-rag-correction --no-dev

# Set working directory to the service root so that `app.main:app` resolves cleanly
WORKDIR /workspace/services/correction

EXPOSE 8003

CMD ["uv", "run", "--package", "compass-rag-correction", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
