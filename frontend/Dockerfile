# Lives in frontend/ and is built with context ./frontend (see docker-compose.yml)
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Same uid as the backend image, so the bind-mounted data/ folder stays writable.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /srv/frontend

# 1. Dependencies first: rebuilt only when pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,id=uv-frontend,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
# 2. Application code (everything not excluded by .dockerignore, including .streamlit/).
COPY . .

# Mount point for the SQLite database (accounts, subjects, documents, chat history).
RUN mkdir -p data && chown appuser:appuser data

ENV PATH="/srv/frontend/.venv/bin:$PATH"
USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]