# Lives in app/ and is built with context ./app (see docker-compose.yml)
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CONCURRENT_DOWNLOADS=4 \
    UV_HTTP_TIMEOUT=120 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 1000 appuser

WORKDIR /srv/app

# 1. Dependencies first: rebuilt only when pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,id=uv-api,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 2. Application code (everything not excluded by .dockerignore).
COPY . .

RUN mkdir -p models \
    && chown appuser:appuser models

ENV PATH="/srv/app/.venv/bin:$PATH"
USER appuser

EXPOSE 8000

# One worker on purpose: ingestion status lives in a per-process dictionary.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]