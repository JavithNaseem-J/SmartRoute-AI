FROM node:22-bookworm-slim AS frontend

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

FROM python:3.10-slim AS runtime

ARG COMMIT_SHA
ARG BUILD_TIME
ARG RENDER_GIT_COMMIT

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    SMARTROUTE_COMMIT_SHA="${COMMIT_SHA}" \
    SMARTROUTE_BUILD_TIME="${BUILD_TIME}" \
    PATH="/app/.venv/bin:/root/.local/bin:$PATH"

LABEL org.opencontainers.image.revision="${COMMIT_SHA}" \
      org.opencontainers.image.created="${BUILD_TIME}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .
COPY --from=frontend /app/frontend/dist ./frontend/dist

RUN if [ -n "$COMMIT_SHA" ]; then \
        printf '%s\n' "$COMMIT_SHA" > /app/.commit_sha; \
    elif [ -n "$RENDER_GIT_COMMIT" ]; then \
        printf '%s\n' "$RENDER_GIT_COMMIT" > /app/.commit_sha; \
    else \
        printf '%s\n' "unknown" > /app/.commit_sha; \
    fi \
    && if [ -n "$BUILD_TIME" ]; then \
        printf '%s\n' "$BUILD_TIME" > /app/.build_time; \
    else \
        date -u +'%Y-%m-%dT%H:%M:%SZ' > /app/.build_time; \
    fi

RUN python scripts/train_classifier.py

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p data/documents data/embeddings models/classifiers logs \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
