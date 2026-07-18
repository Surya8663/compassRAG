FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /workspace

# Enable bytecode compilation and non-interactive sync
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy workspace pyproject.toml and shared module
COPY pyproject.toml ./
COPY shared/ ./shared/

# Copy target service
COPY services/api-gateway/ ./services/api-gateway/

# Sync dependencies for this package without dev deps
RUN uv sync --package compass-rag-api-gateway --no-dev

# Set working directory to the service root so that `app.main:app` resolves cleanly
WORKDIR /workspace/services/api-gateway

EXPOSE 8000

CMD ["uv", "run", "--package", "compass-rag-api-gateway", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
