## ADDED Requirements

### Requirement: Safe cleanup phases are independently verifiable
The system SHALL group cleanup work into independently verifiable phases so low-risk hygiene changes are not coupled to deployment, dependency, or architecture refactors.

#### Scenario: Phase cleanup validation
- **WHEN** a cleanup phase is implemented
- **THEN** the relevant lint, type, unit, frontend, and Docker checks for that phase pass before the next phase begins

### Requirement: Repository hygiene excludes local analysis artifacts
The repository SHALL ignore local analysis/tool outputs that are not part of the application source of truth.

#### Scenario: Graphify output remains local
- **WHEN** Graphify generates `graphify-out/`
- **THEN** Git does not show it as an untracked artifact intended for commit

### Requirement: Development tooling matches the active test runner
Project editor settings SHALL match the active pytest-based test workflow.

#### Scenario: VS Code test discovery
- **WHEN** a developer opens the repository in VS Code
- **THEN** the checked-in settings do not disable pytest in favor of unittest

### Requirement: Legacy UI surfaces are removed after React parity
The repository SHALL remove the legacy Streamlit dashboard only after the React/FastAPI application supports the same production demo flows.

#### Scenario: React parity before Streamlit removal
- **WHEN** `app.py` is removed
- **THEN** the React/FastAPI app supports chat, streaming, RAG upload, document listing, document deletion, clear-all, cost analytics, and budget display

### Requirement: Compatibility Dockerfiles are not kept after deployment migration
The repository SHALL keep one active Dockerfile for the single-service Render deployment unless a second Dockerfile is referenced by a live deployment target.

#### Scenario: Render uses single Dockerfile
- **WHEN** `render.yaml` and the live Render service point to the root `Dockerfile`
- **THEN** obsolete compatibility Dockerfiles are removed from the repository

### Requirement: Runtime dependencies match active runtime paths
Production dependencies SHALL support active runtime paths only; packages used exclusively by removed legacy paths SHALL be removed from production dependency groups.

#### Scenario: Streamlit dependency cleanup
- **WHEN** the legacy Streamlit dashboard is removed
- **THEN** `streamlit`, `plotly`, and `pandas` are no longer production dependencies unless another active runtime path imports them

### Requirement: Local document ingestion compatibility is explicit
The local `DOCUMENTS_DIR` and `/v1/index` ingestion path SHALL either be documented as a supported manual/admin compatibility path or removed after cloud upload becomes the only supported ingestion flow.

#### Scenario: Cloud upload is canonical
- **WHEN** the application documents the canonical RAG ingestion path
- **THEN** `/v1/documents/upload` with Supabase Storage, Postgres metadata, and Qdrant indexing is identified as the supported user-facing flow

### Requirement: Environment examples describe real configuration
`.env.example` SHALL document only environment variables that the application reads or that deployment tooling requires.

#### Scenario: Budget configuration source is accurate
- **WHEN** budget limits are configured from `config/routing.yaml`
- **THEN** `.env.example` does not advertise unused `DAILY_BUDGET`, `WEEKLY_BUDGET`, or `MONTHLY_BUDGET` variables unless the code reads them

#### Scenario: Optional runtime knobs are documented
- **WHEN** code reads optional variables such as `MAX_DOCUMENT_UPLOAD_BYTES`, `WEBHOOK_URL`, `DEMO_TOKEN_TTL_SECONDS`, or OTEL exporter options
- **THEN** `.env.example` documents them or intentionally explains that they are advanced optional settings

### Requirement: Dependency manifests have one source of truth
Python dependency manifests SHALL be generated from or synchronized with `pyproject.toml` and `uv.lock`, not maintained as independent conflicting sources.

#### Scenario: Requirements files are removed or generated
- **WHEN** `requirements.txt` or `requirements-dev.txt` exists
- **THEN** it is either generated from uv metadata or explicitly retained for a documented legacy target
