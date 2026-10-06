#!/usr/bin/env sh
set -eu

if [ "${RUN_MIGRATIONS:-true}" = "true" ]; then
  echo "Running database migrations..."
  if ! alembic upgrade head; then
    echo "WARNING: alembic migration failed or was skipped. Continuing to start the server..."
  fi
fi

exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-10000}"
