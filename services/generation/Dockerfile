FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /workspace

# Enable bytecode compilation and non-interactive sync
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy workspace pyproject.toml and shared module
COPY pyproject.toml ./
COPY shared/ ./shared/

# Copy target service
COPY services/generation/ ./services/generation/

# Sync dependencies for this package without dev deps
RUN uv sync --package compass-rag-generation --no-dev

# Set working directory to the service root so that `app.main:app` resolves cleanly
WORKDIR /workspace/services/generation

EXPOSE 8004

CMD ["uv", "run", "--package", "compass-rag-generation", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
