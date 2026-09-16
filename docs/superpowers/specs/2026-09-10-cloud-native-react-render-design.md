# SmartRoute AI Cloud Native Consolidation Design

Date: 2026-09-10

## Goal

Consolidate SmartRoute AI into one Render web service that serves a compiled React frontend and the FastAPI backend from the same origin. The result should be closer to a cloud native app: one deployable container, dynamic port binding, externalized runtime dependencies, explicit health and readiness behavior, and CI that validates the actual deployment artifact.

## Original State

At the time this design was written, the repository had a Python FastAPI backend and a Streamlit dashboard. Render was configured with two web services, both pointing at the same multi-stage Dockerfile. Because Docker builds the final stage by default, both Render services could end up running the dashboard stage unless the platform was configured outside the repo. The dashboard also wrote uploaded files to its own local filesystem, while the API indexed files from the API container filesystem. That made split deployment fragile.

There was no React frontend scaffold at that point. The requested React prompt box therefore required creating a frontend workspace rather than only copying a component into an existing app.

## Chosen Approach

Build a new React, TypeScript, Tailwind, and shadcn-style frontend under `frontend/`, then serve its production build from FastAPI. FastAPI remains the only runtime process in the Render service and owns all API endpoints, streaming, auth checks, file upload ingestion, static asset serving, SPA fallback, and health checks.

This was the recommended path because it satisfies the one-service, same-origin deployment target without running an additional dashboard process beside FastAPI.

## Architecture

Runtime flow:

```text
Browser
  -> React SPA assets from FastAPI
  -> same-origin /v1/* API calls
  -> same-origin /health and /ready probes
FastAPI
  -> PostgreSQL, Redis, Qdrant, OpenRouter, Hugging Face, optional observability providers
```

Render will build one Docker image. The container will run Uvicorn on `0.0.0.0:${PORT}`. The image will include the compiled frontend assets and Python backend dependencies.

## Frontend

Create a production React app with Vite, TypeScript, Tailwind, and local shadcn-compatible UI primitives. Integrate the prompt box as the primary query input, adapted for SmartRoute behavior:

- Submit prompts to existing `/v1/query` or `/v1/query/stream` endpoints.
- Add abort support for streaming requests.
- Add document upload controls that match the RAG file types the backend can index.
- Keep Search, Think, and Canvas modes as UI metadata unless backend support is added later.
- Move unsafe DOM style injection into CSS.
- Fix recorder interval typing and lifecycle behavior.

The initial visual direction should be a focused operations console rather than a marketing page: dense, readable, fast to scan, with polished controls and restrained motion.

## Backend

FastAPI will serve static frontend assets when present, fall back to `index.html` for non-API routes, and preserve all existing `/v1/*` behavior. Add or adjust:

- Dynamic `PORT` handling in local entrypoints and Docker command.
- `/ready` for dependency readiness.
- `/health` for cheap liveness.
- Multipart upload endpoint that stores incoming documents where the indexer can read them in the same container.
- Same-origin defaults so production no longer depends on split CORS or `API_URL`.

Document storage uses Supabase Storage as the durable object store, with document metadata in Supabase-backed Postgres and vector chunks in Qdrant. The local `data/documents` path remains only as a compatibility/manual ingestion path for the legacy `/v1/index` endpoint.

## Render

Replace the two-service blueprint with one web service. Remove duplicate dashboard environment variables and fixed port assumptions. Keep secrets externalized. If `preDeployCommand` remains in use, note that Render only supports it on paid instance types.

## CI/CD

Replace the split API/dashboard image pipeline with one pipeline that checks:

- Python lint/type/test commands already used by the repo.
- Frontend install, typecheck/lint, and production build.
- Single Docker image build for the Render artifact.

Deploy should remain gated to the main branch and use configured secrets only. No dashboard mutation or push is part of this implementation.

## Docs And Cleanup

Update README, `.env.example`, `.devcontainer`, Render docs/config, and active OpenSpec specs to match the single-service architecture. Legacy dashboard and compatibility Docker artifacts can be removed after the React app reaches functional parity for query, streaming, document upload/indexing, stats, budget, and document management flows.

## Verification

Minimum verification before handoff:

- Backend tests that can run in the local environment.
- Frontend build or typecheck.
- Docker build command if Docker daemon is available.
- Static serving smoke check when feasible.
- Clear note for any verification blocked by missing services or unavailable Docker daemon.

## Constraints

Do not commit or push changes. Preserve existing user edits in the dirty worktree. Do not delete deployment files until the replacement path is implemented and verified.
