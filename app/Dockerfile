# ./app/Dockerfile
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CONCURRENT_DOWNLOADS=4 \
    UV_HTTP_TIMEOUT=120 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH="/srv/app/models" \
    HF_HOME="/srv/app/models" \
    PATH="/srv/app/.venv/bin:$PATH"

RUN useradd --create-home --uid 1000 appuser

WORKDIR /srv/app

# 1. Install dependencies first (cached unless pyproject.toml or uv.lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache

# 2. Pre-bake FastEmbed Dense (bge-large-en-v1.5) & Sparse (Qdrant/bm25) ONNX models
# so Cloud Run cold starts never have to download models at runtime
RUN mkdir -p /srv/app/models && \
    python -c "from fastembed import TextEmbedding, SparseTextEmbedding; \
    TextEmbedding('BAAI/bge-large-en-v1.5', cache_dir='/srv/app/models'); \
    SparseTextEmbedding('Qdrant/bm25', cache_dir='/srv/app/models')" && \
    chown -R appuser:appuser /srv/app

# 3. Copy application code
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

# Cloud Run injects $PORT (defaults to 8000)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]