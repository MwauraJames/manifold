# ./app/Dockerfile
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CONCURRENT_DOWNLOADS=4 \
    UV_HTTP_TIMEOUT=120 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH="/srv/app/models" \
    PATH="/srv/app/.venv/bin:$PATH" \
    PORT=8000

RUN useradd --create-home --uid 1000 appuser

WORKDIR /srv/app

# 1. Install dependencies first
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache

# 2. ONLY pre-bake the lightweight Sparse (Qdrant/bm25) ONNX model (~25MB)
RUN mkdir -p /srv/app/models && \
    python -c "from fastembed import SparseTextEmbedding; \
    SparseTextEmbedding('Qdrant/bm25', cache_dir='/srv/app/models')"

# 3. Copy application code 
COPY --chown=appuser:appuser . .

# Ensure appuser owns everything
RUN chown -R appuser:appuser /srv/app

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]