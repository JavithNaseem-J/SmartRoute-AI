## Why

SmartRoute-AI has completed several production hardening passes, but the repository still carries legacy UI, deployment, dependency, and local-ingestion surfaces from earlier architectures. Cleaning these up now will make the portfolio easier to review, reduce install/build weight, and prevent future fixes from touching stale paths.

## What Changes

- Remove or retire unused frontend demo/type/util files that are not imported by the app.
- Align local tooling and repo hygiene with the active pytest + uv + Graphify workflow.
- Retire legacy Streamlit and Render compatibility surfaces after confirming React/FastAPI parity.
- Slim runtime dependencies that only support removed or unused paths.
- Resolve the classifier/centroid artifact source-of-truth so routing behavior is deterministic in production.
- Deprecate the local `/v1/index` document ingestion path once cloud upload remains the only supported user-facing ingestion flow.
- Refactor oversized backend/frontend modules after the safe cleanup phases are complete.
- Update README/OpenSpec/docs so they describe the current single-service architecture rather than historical migration paths.

No public API behavior should be removed until its replacement is verified and documented. Destructive cleanup must be phased and validated by CI after each stage.

## Capabilities

### New Capabilities

- None. This change cleans and tightens existing structural, routing, and pipeline capabilities.

### Modified Capabilities

- `structural-debt-resolution`: Extend the existing cleanup contract to cover legacy UI/deployment removal, repo hygiene, dependency-source consolidation, stale requirements exports, and documented environment-variable ownership.
- `ml-router-enhancements`: Clarify how `reference_centroids.npy` is produced, tracked, or removed so semantic routing features cannot silently run zeroed in production.
- `pipeline-refactoring`: Extend the refactoring scope to route/service/component boundaries for `api/main.py`, `frontend/src/App.tsx`, and `frontend/src/components/ui/ai-prompt-box.tsx`.

## Impact

- Backend: `api/main.py`, `src/retrieval/indexer.py`, `src/documents/storage.py`, `src/pipeline/inference.py`, `src/cost/tracker.py`.
- Frontend: `frontend/src/App.tsx`, `frontend/src/components/ui/ai-prompt-box.tsx`, `frontend/src/components/ui/demo.tsx`, `frontend/src/types/chat.ts`, `frontend/src/lib/utils.ts`.
- Deployment/build: `Dockerfile`, `Dockerfile.api`, `render.yaml`, `.dockerignore`, `.gitignore`, `.vscode/settings.json`, CI workflows.
- Dependencies: `pyproject.toml`, `uv.lock`, `requirements.txt`, `requirements-dev.txt`, frontend package metadata as needed.
- Docs/specs: `README.md`, `docs/*`, `openspec/specs/*`.
- Operational risk: dependency removals and legacy endpoint removals require CI, Docker smoke test, and production parity checks before merge/deploy.
