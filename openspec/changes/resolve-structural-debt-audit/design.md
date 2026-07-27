## Context

SmartRoute-AI has grown organically from a prototype to a near-production system. During that growth, structural debt accumulated in three areas: (1) runtime correctness issues where async code paths were mixed with sync calls, (2) deployment configuration drift where different infra files disagree on migrations and env vars, and (3) codebase cleanliness issues (dead code, duplicate files, stale artifacts) that slow development.

## Goals / Non-Goals

**Goals:**
- Fix every High-severity issue from the audit: eval async path, migration sequencing, env-var contract, stale planning, training pipeline.
- Reduce developer confusion by eliminating duplicate files (two Dockerfiles, two CI workflows, two OpenSpec skill layers, two structural-debt changes).
- Establish `pyproject.toml` + `uv.lock` as the single dependency source of truth.
- Prune dead code that silently diverged from actual runtime behaviour.

**Non-Goals:**
- Adding new features or capabilities beyond what the audit specifies.
- Rewriting the inference pipeline or routing logic.
- Changing public API contracts or data schemas.

## Decisions

### Decision 1: Single Migration Path — Render preDeployCommand only
**Choice:** Remove `alembic upgrade head &&` from `Dockerfile.api` CMD and docker-compose `command`; keep migrations only in `render.yaml` `preDeployCommand`.
**Rationale:** The spec `openspec/specs/database-migrations/spec.md` explicitly states migrations must not run during web boot. Running migrations in the container startup causes race conditions on horizontal scale and fails fast on first boot if the DB isn't yet reachable.
**Alternative considered:** A separate `migrate` service in docker-compose — acceptable but adds complexity; the preDeployCommand pattern is already present and correct.

### Decision 2: Single `.env.example` as env-var contract
**Choice:** Write a comprehensive `.env.example` listing every required runtime variable (`OPENROUTER_API_KEY`, `SUPABASE_JWT_SECRET`, `HF_TOKEN`, `QDRANT_URL`, `QDRANT_API_KEY`, `REDIS_URL`, `DATABASE_URL`). Remove legacy `NVIDIA_API_KEY`, `GROQ_API_KEY`, `SMARTROUTE_API_KEY` from all deployment configs.
**Rationale:** All deployment targets (`render.yaml`, `docker-compose.yml`, CI secrets) should derive from one contract. Legacy keys are neither referenced at runtime (confirmed by code grep) nor valid for current providers.

### Decision 3: Fix eval async — reuse pipeline result context
**Choice:** In `_run_single()`, instead of calling `retriever.retrieve()` again, pass `result["sources"]` already present in the `pipeline.run()` return value as the retrieved context for RAGAS evaluation.
**Rationale:** Avoids a duplicate retrieval call and keeps eval consistent with what the pipeline actually used. Simpler and more accurate.

### Decision 4: Merge CI workflows into one
**Choice:** Replace `ci.yml` and `deploy.yml` with a single `ci.yml` that has explicit jobs: `test` → `build-api` → `build-dashboard` → `deploy`. The deploy job is gated on `main` branch only.
**Rationale:** Running lint/tests twice doubles CI minutes and confuses which workflow is authoritative. A single file with job-level gates is idiomatic.

### Decision 5: pyproject.toml + uv.lock as single dependency source
**Choice:** Remove `requirements.txt` and `requirements-dev.txt`. Update `Dockerfile` stages to use `uv sync` or `uv export` to install from `pyproject.toml`.
**Rationale:** Already using `uv` as the package manager. Maintaining parallel dependency files creates drift (confirmed by audit: `aiohttp`, `fastembed` missing from `pyproject.toml`).

### Decision 6: Single multi-stage Dockerfile
**Choice:** Merge `Dockerfile` and `Dockerfile.api` into one `Dockerfile` with stages: `base` → `api` / `dashboard`.
**Rationale:** Near-identical files diverge silently. Multi-stage builds are the standard pattern and add zero overhead.

### Decision 7: Remove `.agent/workflows/opsx-*.md` duplicates
**Choice:** Delete the six `.agent/workflows/opsx-*.md` files. Keep `.agent/skills/openspec-*/SKILL.md` as canonical.
**Rationale:** The workflow files are near-copies of the skills. Having two instruction sources causes confusion about which is authoritative. The IDE already loads skills from `.agent/skills/`.

### Decision 8: Fix training pipeline — keep CSV rows
**Choice:** In `train_classifier.py`, after loading the CSV, append the loaded rows to `queries`/`labels` rather than resetting them. Remove the reset that discards the generated data.
**Rationale:** The audit confirms `generate_training_data.py` writes a CSV that is immediately discarded. The fix is a one-line change.

## Risks / Trade-offs

- [Risk] Removing `alembic upgrade head` from Docker CMD could break local dev that relies on auto-migration → Mitigation: Document `uv run alembic upgrade head` as a manual step in `README.md` and add it to docker-compose as a one-shot `migrate` service.
- [Risk] Merging `requirements.txt` into `pyproject.toml` only could break external CI that pips from requirements files → Mitigation: Add a `uv export --output-file requirements.txt` step to CI for any legacy target that needs it, and document in `README.md`.
- [Risk] Deleting `.agent/workflows/` files could break any user bookmark/shortcut to those paths → Mitigation: Low risk; slash-commands load from skills not workflow files at runtime.
