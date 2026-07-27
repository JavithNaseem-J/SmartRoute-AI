# structural-debt-resolution Specification

## Purpose
TBD - created by archiving change resolve-structural-debt-audit. Update Purpose after archive.
## Requirements
### Requirement: Evaluation path uses async retrieval
The system SHALL correctly await `pipeline.retriever.retrieve()` in `_run_single()` of `src/evaluation/ragas_eval.py`, or SHALL reuse context from the `pipeline.run()` result, so evaluation produces correct retrieved context.

#### Scenario: RAGAS evaluation retrieves context
- **WHEN** `_run_single()` is called with a sample question
- **THEN** the retrieved context is populated from the pipeline result without raising an `AttributeError` or `TypeError` due to unawaited coroutine

---

### Requirement: Migrations run only in pre-deploy, not container startup
The system SHALL NOT run `alembic upgrade head` as part of the web container startup command in `Dockerfile.api` or `docker-compose.yml`.
Migrations SHALL only be invoked via `render.yaml` `preDeployCommand` (for cloud) and a dedicated one-shot `migrate` service in `docker-compose.yml` (for local).

#### Scenario: Container starts without running migrations
- **WHEN** the API container starts (CMD is executed)
- **THEN** no `alembic upgrade head` command runs as part of startup

#### Scenario: Pre-deploy step runs migrations
- **WHEN** Render deploys the application
- **THEN** `alembic upgrade head` runs exactly once via `preDeployCommand` before the web service starts

---

### Requirement: Single canonical env-var contract
The system SHALL define all required runtime environment variables in `.env.example` as the single source of truth.
Deployment configs (`render.yaml`, `docker-compose.yml`) SHALL reference only variables defined in `.env.example`.
Legacy keys `NVIDIA_API_KEY`, `GROQ_API_KEY`, and `SMARTROUTE_API_KEY` SHALL be removed from all deployment configs.

#### Scenario: Runtime uses OPENROUTER_API_KEY
- **WHEN** `src/models/openrouter_model.py` initializes a model
- **THEN** it reads `OPENROUTER_API_KEY` (not `NVIDIA_API_KEY` or `GROQ_API_KEY`)

#### Scenario: All required vars documented
- **WHEN** a developer reads `.env.example`
- **THEN** every variable required at runtime (`OPENROUTER_API_KEY`, `SUPABASE_JWT_SECRET`, `HF_TOKEN`, `QDRANT_URL`, `QDRANT_API_KEY`, `REDIS_URL`, `DATABASE_URL`) is listed with a comment

---

### Requirement: Training pipeline consumes generated data
The system SHALL use the CSV rows loaded from `data/training/synthetic_queries.csv` when training the complexity classifier in `scripts/train_classifier.py`, rather than discarding them.

#### Scenario: Synthetic data is included in training
- **WHEN** `train_classifier.py` loads `synthetic_queries.csv`
- **THEN** the loaded query/label pairs are appended to (not replaced by) the built-in training set before the classifier is fitted

---

### Requirement: Single CI/CD workflow
The system SHALL use a single GitHub Actions workflow file that runs `test` → `build-api` → `build-dashboard` → `deploy` jobs in dependency order.
The `deploy` job SHALL only run on the `main` branch.

#### Scenario: Tests run on every push
- **WHEN** any branch is pushed
- **THEN** the `test` job runs lint and pytest

#### Scenario: Deployment only from main
- **WHEN** a non-main branch push triggers CI
- **THEN** the `deploy` job is skipped

---

### Requirement: pyproject.toml is sole dependency source
The system SHALL declare all runtime and dev dependencies exclusively in `pyproject.toml`. `requirements.txt` and `requirements-dev.txt` SHALL either be removed or generated from `pyproject.toml` via `uv export`, not maintained manually.

#### Scenario: Dockerfile installs from pyproject.toml
- **WHEN** the Docker image is built
- **THEN** `uv sync` (or equivalent) installs dependencies from `pyproject.toml`, not from a separate `requirements.txt`

---

### Requirement: Single multi-stage Dockerfile
The system SHALL have exactly one `Dockerfile` with a shared `base` stage and separate `api` and `dashboard` final stages. The standalone `Dockerfile.api` SHALL be removed.

#### Scenario: API image built from single Dockerfile
- **WHEN** `docker build --target api` is run
- **THEN** the API service image is produced correctly

#### Scenario: Dashboard image built from single Dockerfile
- **WHEN** `docker build --target dashboard` is run
- **THEN** the Streamlit dashboard image is produced correctly

