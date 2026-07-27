## Why

An independent audit of SmartRoute-AI identified 15 structural issues (6 High, 6 Medium, 3 Low severity) covering broken runtime paths, duplicate deployment configuration, stale planning artifacts, dead training code, and redundant abstractions. These issues create operational risk in production (broken evaluation path, wrong migration sequencing, missing env vars) and slow development velocity (inconsistent dependency management, two copies of every OpenSpec skill, near-identical CI workflows). Resolving them now clears the runway for reliable production deployment.

## What Changes

**High Severity (Production Risk)**
- Fix `_run_single()` in `src/evaluation/ragas_eval.py` to `await` the async `retriever.retrieve()` call instead of calling it synchronously.
- Remove `alembic upgrade head &&` from `Dockerfile.api` command and `docker-compose.yml` web service startup; keep migrations **only** in the Render `preDeployCommand` and a standalone migration job.
- Consolidate env-var contracts: create a single `.env.example` as source of truth, remove legacy `NVIDIA_API_KEY`, `GROQ_API_KEY`, `SMARTROUTE_API_KEY` keys from `render.yaml`, `docker-compose.yml`, and `api/main.py`, replacing with `OPENROUTER_API_KEY`, `SUPABASE_JWT_SECRET`, `HF_TOKEN`, Qdrant, Redis, and DB vars.
- Archive two stale duplicate planning changes (`fix-structural-debt`, `structural-debt-fixes`). ✓ Done in this proposal step.
- Fix `scripts/train_classifier.py` to actually consume the CSV rows from `generate_training_data.py` instead of immediately resetting queries/labels.

**Medium Severity (Dev Velocity)**
- Merge `.github/workflows/ci.yml` and `.github/workflows/deploy.yml` into one workflow with explicit `test`, `build-api`, `build-dashboard`, and `deploy` jobs.
- Make `pyproject.toml` + `uv.lock` the single dependency source of truth; remove/generate `requirements.txt` and `requirements-dev.txt` only for legacy deployment targets.
- Merge `Dockerfile` and `Dockerfile.api` into a single multi-stage `Dockerfile` with shared base, separate `api` and `dashboard` final stages.
- Delete duplicate `.agent/workflows/opsx-*.md` files (keep `.agent/skills/` as canonical).
- Extract shared generation/finalization helpers from `run()` and `astream_run()` in `src/pipeline/inference.py`.
- Move Qdrant sparse-model detection into a shared utility in `src/core/dependencies.py`.
- Remove unused `classification.active_classifier`, `complexity_rules`, `quality`, `max_latency`, and `emergency_stop` sections from `config/routing.yaml`.

**Low Severity (Cleanup)**
- Remove `mock_key` Hugging Face cache blob from the repo and add model/cache dirs to `.gitignore`.
- Collapse `src/routing/base_classifier.py` single-method abstraction into `src/routing/classifier.py`.
- Remove unused `DocumentIndexer.retriever`, `_executor`, and retrieval `persist_dir`/`max_distance` fields.

## Capabilities

### New Capabilities
- `structural-debt-resolution`: Cross-cutting fixes to evaluation correctness, deployment config, env-var contracts, training pipeline, CI, and codebase cleanliness.

### Modified Capabilities
- `rag-retrieval-grounding`: `ragas_eval.py` async fix changes how evaluation exercises retrieval.
- `document-deletion-management`: env-var unification affects deployment config for this capability.

## Impact

- `src/evaluation/ragas_eval.py`: Bug fix (await missing).
- `Dockerfile.api`, `docker-compose.yml`, `render.yaml`: Migration sequencing correction.
- `.env.example`, `render.yaml`, `docker-compose.yml`, `api/main.py`: Env-var contract unification.
- `scripts/generate_training_data.py`, `scripts/train_classifier.py`: Training pipeline fix.
- `.github/workflows/ci.yml`, `.github/workflows/deploy.yml`: Merged into single workflow.
- `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`: Dependency consolidation.
- `Dockerfile`, `Dockerfile.api`: Merged into multi-stage.
- `.agent/workflows/opsx-*.md`: Removed duplicate workflow files.
- `src/pipeline/inference.py`: Refactored shared helpers.
- `src/core/dependencies.py`, `src/retrieval/indexer.py`, `src/retrieval/retriever.py`: Sparse-model detection moved to shared utility.
- `config/routing.yaml`: Unused config sections removed.
- `mock_key/`, `.gitignore`: Cache blob removed.
- `src/routing/base_classifier.py`: Collapsed into classifier.
- `src/retrieval/indexer.py`, `src/retrieval/retriever.py`: Dead fields removed.
