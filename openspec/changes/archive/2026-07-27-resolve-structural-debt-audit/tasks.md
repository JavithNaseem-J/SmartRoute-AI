## 1. High Severity — Runtime Correctness

- [x] 1.1 Fix `_run_single()` in `src/evaluation/ragas_eval.py` to reuse `result["sources"]` from `pipeline.run()` as the RAGAS context instead of calling `retriever.retrieve()` synchronously.
- [x] 1.2 Remove `alembic upgrade head &&` from `Dockerfile.api` CMD so migrations no longer run at container startup.
- [x] 1.3 Remove `alembic upgrade head &&` from the `web` service `command` in `docker-compose.yml`; add a dedicated one-shot `migrate` service that runs `alembic upgrade head` on startup.
- [x] 1.4 Verify `render.yaml` `preDeployCommand` is the sole migration trigger (no other changes needed unless it is missing).

## 2. High Severity — Env-Var Contract Unification

- [x] 2.1 Rewrite `.env.example` to list every required runtime variable: `OPENROUTER_API_KEY`, `SUPABASE_JWT_SECRET`, `HF_TOKEN`, `QDRANT_URL`, `QDRANT_API_KEY`, `REDIS_URL`, `DATABASE_URL`, with inline comments explaining each.
- [x] 2.2 Remove `NVIDIA_API_KEY`, `GROQ_API_KEY`, `SMARTROUTE_API_KEY` from `render.yaml`, `docker-compose.yml`, and any remaining references in `api/main.py` or `src/utils/security.py`.
- [x] 2.3 Add all required vars to `render.yaml` `envVars` block if not already present.

## 3. High Severity — Training Pipeline Fix

- [x] 3.1 In `scripts/train_classifier.py`, keep the CSV-loaded rows by appending them to `queries`/`labels` rather than resetting those lists after loading.
- [x] 3.2 Verify end-to-end: run `generate_training_data.py` then `train_classifier.py` and confirm the loaded CSV rows appear in the final training set.

## 4. Medium Severity — CI/CD Consolidation

- [x] 4.1 Create a single `.github/workflows/ci.yml` with jobs: `test` (lint + pytest), `build-api` (Docker build API target), `build-dashboard` (Docker build dashboard target), `deploy` (gated on main branch).
- [x] 4.2 Delete `.github/workflows/deploy.yml`.
- [x] 4.3 Ensure `build-api` and `build-dashboard` push to the correct registries (matching what the original two workflows did).

## 5. Medium Severity — Dependency Consolidation

- [x] 5.1 Add any dependencies present in `requirements.txt` / `requirements-dev.txt` but missing from `pyproject.toml` (e.g., `aiohttp`, `fastembed`, `langchain-huggingface`).
- [x] 5.2 Run `uv lock` to regenerate `uv.lock` after updating `pyproject.toml`.
- [x] 5.3 Update Dockerfile(s) to use `uv sync` from `pyproject.toml` instead of `pip install -r requirements.txt`.
- [x] 5.4 Delete `requirements.txt` and `requirements-dev.txt` (or replace with `uv export`-generated outputs and mark them as generated).

## 6. Medium Severity — Merge Dockerfiles

- [x] 6.1 Create a single `Dockerfile` with stages: `base` (common Python/uv setup, deps install), `api` (API entrypoint), `dashboard` (Streamlit entrypoint).
- [x] 6.2 Delete `Dockerfile.api`.
- [x] 6.3 Update `docker-compose.yml` and CI to build with `--target api` or `--target dashboard` as appropriate.

## 7. Medium Severity — Remove Duplicate Workflow Files

- [x] 7.1 Delete `.agent/workflows/opsx-apply.md`, `opsx-archive.md`, `opsx-explore.md`, `opsx-propose.md`, `opsx-sync.md`, `opsx-update.md` (the `.agent/skills/` files are canonical).
- [x] 7.2 Verify the IDE still resolves slash-commands after deletion (skills in `.agent/skills/` are loaded directly).

## 8. Medium Severity — Inference Pipeline Deduplication

- [x] 8.1 Extract shared logic (model loading, history/message setup, token counting, cost calculation, memory update, response payload construction) from `run()` and `astream_run()` in `src/pipeline/inference.py` into private helper methods.
- [x] 8.2 Refactor both `run()` and `astream_run()` to call those helpers.

## 9. Medium Severity — Shared Sparse-Model Detection

- [x] 9.1 Add a `get_sparse_vector(query)` helper to `src/core/dependencies.py` that encapsulates the `_sparse_embedding_model` attribute check and query embedding.
- [x] 9.2 Update `src/retrieval/indexer.py` and `src/retrieval/retriever.py` to use the shared helper instead of duplicating the attribute check.

## 10. Medium Severity — Remove Unused Routing Config

- [x] 10.1 Delete unused sections from `config/routing.yaml`: `classification.active_classifier`, `complexity_rules`, `quality`, `max_latency`, `emergency_stop`.
- [x] 10.2 Verify `src/routing/router.py` does not reference these keys at runtime (grep check).

## 11. Low Severity — Cleanup

- [x] 11.1 Remove `mock_key/` directory (HuggingFace cache blob) from the repository.
- [x] 11.2 Add `mock_key/`, `*.cache`, `model_cache/`, `hf_cache/` to `.gitignore` to prevent future re-check-in.
- [x] 11.3 Delete `src/routing/base_classifier.py` and inline its single method directly in `src/routing/classifier.py`.
- [x] 11.4 Remove unused fields from `DocumentIndexer` and `DocumentRetriever`: `retriever`, `_executor` (module-level), `persist_dir` (retriever), `max_distance`.
