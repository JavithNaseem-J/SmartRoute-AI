# structural-debt-resolution Specification

## Purpose
Keep deployment, dependency, and structural contracts aligned with the current single-service SmartRoute AI architecture.

## Requirements

### Requirement: Evaluation path uses async retrieval
The system SHALL correctly await `pipeline.retriever.retrieve()` in `_run_single()` of `src/evaluation/ragas_eval.py`, or SHALL reuse context from the `pipeline.run()` result, so evaluation produces correct retrieved context.

#### Scenario: RAGAS evaluation retrieves context
- **WHEN** `_run_single()` is called with a sample question
- **THEN** the retrieved context is populated from the pipeline result without raising an `AttributeError` or `TypeError` due to unawaited coroutine

---

### Requirement: Migrations run only in pre-deploy, not container startup
The system SHALL NOT run `alembic upgrade head` as part of the web container startup command.
Migrations SHALL only be invoked via `render.yaml` `preDeployCommand` for cloud deployments or manually by an operator for local development.

#### Scenario: Container starts without running migrations
- **WHEN** the web container starts
- **THEN** no `alembic upgrade head` command runs as part of startup

#### Scenario: Pre-deploy step runs migrations
- **WHEN** Render deploys the application on a plan that supports pre-deploy commands
- **THEN** `alembic upgrade head` runs exactly once via `preDeployCommand` before the web service starts

---

### Requirement: Single canonical env-var contract
The system SHALL define all required runtime environment variables in `.env.example` as the single source of truth.
Deployment configs (`render.yaml`) SHALL reference only variables defined in `.env.example`.
Legacy keys `NVIDIA_API_KEY`, `GROQ_API_KEY`, and `SMARTROUTE_API_KEY` SHALL be removed from all deployment configs.

#### Scenario: Runtime uses OPENROUTER_API_KEY
- **WHEN** `src/models/openrouter_model.py` initializes a model
- **THEN** it reads `OPENROUTER_API_KEY` and uses `APP_PUBLIC_URL` for provider referer metadata when supplied

#### Scenario: All required vars documented
- **WHEN** a developer reads `.env.example`
- **THEN** every variable required at runtime is listed with a comment

---

### Requirement: Training pipeline consumes generated data
The system SHALL use the CSV rows loaded from `data/training/synthetic_queries.csv` when training the complexity classifier in `scripts/train_classifier.py`, rather than discarding them.

#### Scenario: Synthetic data is included in training
- **WHEN** `train_classifier.py` loads `synthetic_queries.csv`
- **THEN** the loaded query/label pairs are appended to the built-in training set before the classifier is fitted

---

### Requirement: Single CI/CD workflow
The system SHALL use a single GitHub Actions workflow file that runs backend tests, frontend build, single Docker validation, and main-branch deploy in dependency order.
The `deploy` job SHALL only run on the `main` branch.

#### Scenario: Tests run on every pull request
- **WHEN** a pull request targets `main`
- **THEN** backend lint/type/test checks, frontend build, and Docker validation run

#### Scenario: Deployment only from main
- **WHEN** a non-main branch push triggers CI
- **THEN** the `deploy` job is skipped

---

### Requirement: pyproject.toml is sole Python dependency source
The system SHALL declare Python runtime and dev dependencies in `pyproject.toml`. `requirements.txt` and `requirements-dev.txt` SHALL either be removed or generated from `pyproject.toml` via `uv export`, not maintained manually.

#### Scenario: Dockerfile installs from pyproject.toml
- **WHEN** the Docker image is built
- **THEN** `uv sync` installs Python dependencies from `pyproject.toml`

---

### Requirement: Single service Dockerfile
The system SHALL have exactly one active `Dockerfile` that builds the React frontend and runs the FastAPI application as the only web process.

#### Scenario: Render image built from single Dockerfile
- **WHEN** `docker build .` is run
- **THEN** a single image containing the compiled frontend and FastAPI service is produced correctly

#### Scenario: Container binds to platform port
- **WHEN** the container starts with `PORT` set by the platform
- **THEN** Uvicorn binds to that port and serves both the React app and `/v1/*` API routes

---

### Requirement: Durable cloud document storage
Uploaded documents SHALL be stored in Supabase Storage, with metadata stored in Supabase-backed Postgres and retrieval chunks stored in Qdrant.

#### Scenario: Document upload is durable
- **WHEN** a user uploads a supported PDF, TXT, or MD file through `/v1/documents/upload`
- **THEN** the API uploads the object to Supabase Storage, records metadata in Postgres, and indexes chunks in Qdrant using the Storage object path as durable source metadata

#### Scenario: Document deletion removes cloud object
- **WHEN** a user deletes a document through `/v1/documents/{filename}`
- **THEN** the API deletes the Supabase Storage object, purges matching Qdrant vectors, flushes stale semantic cache entries, and marks the metadata record deleted
