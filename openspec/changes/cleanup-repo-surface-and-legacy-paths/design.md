## Context

SmartRoute-AI now runs as a single Render web service: the root `Dockerfile` builds the React frontend, serves it from FastAPI, and exposes `/v1/*` API routes from the same origin. The repository still contains earlier migration surfaces:

- `app.py` Streamlit dashboard and its runtime dependencies.
- `Dockerfile.api` compatibility deployment file.
- `/v1/index` and `DOCUMENTS_DIR` local document ingestion path.
- Generated/legacy `requirements*.txt` alongside `pyproject.toml` and `uv.lock`.
- Missing or ignored `data/models/reference_centroids.npy` despite the ML router spec requiring precomputed centroids.
- Oversized frontend/backend modules that mix unrelated responsibilities.

The cleanup must be phased because several candidates are safe repo hygiene, while others remove legacy runtime paths and require parity checks.

## Goals / Non-Goals

**Goals:**

- Produce one ordered cleanup path instead of repeatedly rediscovering micro-cleanup items.
- Start with low-risk hygiene and dead-file removals.
- Remove legacy UI/deployment/dependency surfaces only after replacement paths are verified.
- Make routing artifacts deterministic in production.
- Improve module boundaries without changing user-facing behavior.
- Keep the repo reviewer-friendly for a production-grade portfolio project.

**Non-Goals:**

- No feature redesign of chat, RAG, analytics, Langfuse, Supabase, Qdrant, or routing strategy.
- No database schema changes unless a cleanup task explicitly proves one is needed.
- No production secret or Render dashboard changes from code alone.
- No removal of compatibility paths until CI and local smoke checks pass.

## Decisions

### Decision 1: Use phased cleanup, not one large refactor

Cleanup will be implemented in phases:

1. Safe hygiene and dead tracked files.
2. Legacy Streamlit/compatibility deployment removal.
3. Dependency slimming.
4. Classifier and centroid build correctness.
5. Backend/frontend boundary refactors.

**Rationale:** This prevents a safe file cleanup from being bundled with risky deployment or architecture changes.

**Alternative considered:** One large cleanup PR. Rejected because a failure would be hard to isolate and would make CI/debugging slower.

### Decision 2: Treat React/FastAPI as the only active UI/runtime path

`app.py` and `Dockerfile.api` will be removed only after React/FastAPI parity is confirmed for chat, streaming, document upload/list/delete/clear, analytics, and budget views.

**Rationale:** The current deployment config and README already point at the single-service architecture, but retaining a parity gate avoids removing a working fallback too early.

### Decision 3: Keep dependency ownership in `pyproject.toml` + `uv.lock`

`requirements.txt` and `requirements-dev.txt` will be removed or clearly regenerated from uv, not maintained as independent dependency sources.

**Rationale:** Docker and CI already use `uv sync`; duplicate manifests drift and increase dependency confusion.

### Decision 4: Make centroid behavior explicit

The ML routing implementation must either:

- generate `data/models/reference_centroids.npy` during a deterministic build/training step,
- track it intentionally with a `.gitignore` exception, or
- remove the semantic centroid feature and update specs/docs.

**Rationale:** The current code silently runs with zero semantic features when the ignored artifact is absent, which conflicts with the existing ML router spec.

### Decision 5: Refactor boundaries after removal work

`api/main.py`, `frontend/src/App.tsx`, and `frontend/src/components/ui/ai-prompt-box.tsx` will be split after legacy cleanup, not before.

**Rationale:** Removing dead/legacy branches first makes the refactor smaller and reduces churn.

## Risks / Trade-offs

- **[Risk] Removing `app.py` breaks an untracked external Streamlit deployment** → Mitigation: verify Render/service settings and README before deletion.
- **[Risk] Removing dependencies breaks transitive/runtime behavior** → Mitigation: remove in small batches, regenerate lockfile, run backend type/lint/tests and frontend build.
- **[Risk] Classifier retraining changes routing behavior** → Mitigation: make training deterministic or stop retraining during Docker build before touching model artifacts.
- **[Risk] `/v1/index` removal breaks manual local workflows** → Mitigation: document `/v1/documents/upload` as the canonical path and keep/remove only after Streamlit removal.
- **[Risk] Large refactors create merge conflicts** → Mitigation: defer boundary refactors until cleanup phases are complete.

## Migration Plan

1. Apply Phase 1 hygiene: remove unused frontend files, fix `.gitignore`, ignore `graphify-out/`, align `.vscode` with pytest, clean stale env docs.
2. Run lint/type/tests/build.
3. Confirm React/FastAPI parity and remove Streamlit + compatibility Docker surfaces.
4. Remove dependencies made obsolete by Phase 2 and regenerate lockfile.
5. Resolve classifier/centroid ownership.
6. Deprecate/remove local ingestion compatibility if no longer needed.
7. Split oversized backend/frontend modules.

Rollback is standard Git revert per phase. Each phase should be independently reviewable.

## Open Questions

- Should `reference_centroids.npy` be committed as a small build artifact or generated during image build?
- Should `/v1/index` remain as a hidden/manual admin endpoint, or be removed entirely?
- Should OpenSpec archive files remain in the repo for process transparency, or be summarized for portfolio readability?
