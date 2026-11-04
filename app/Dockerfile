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
    PATH="/srv/app/.venv/bin:$PATH" \
    PORT=7860

RUN useradd --create-home --uid 1000 appuser

WORKDIR /srv/app

# 1. Install dependencies first (cached unless pyproject.toml or uv.lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache

# 2. Pre-bake FastEmbed Dense (bge-large-en-v1.5) & Sparse (Qdrant/bm25) ONNX models
RUN mkdir -p /srv/app/models && \
    python -c "from fastembed import TextEmbedding, SparseTextEmbedding; \
    TextEmbedding('BAAI/bge-large-en-v1.5', cache_dir='/srv/app/models'); \
    SparseTextEmbedding('Qdrant/bm25', cache_dir='/srv/app/models')"

# 3. Copy application code (excluding models/ and secrets via .dockerignore)
COPY --chown=appuser:appuser . .

# Ensure appuser owns virtual environment, baked models, and source files
RUN chown -R appuser:appuser /srv/app

USER appuser

EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]