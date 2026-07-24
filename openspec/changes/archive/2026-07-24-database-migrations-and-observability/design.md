## Context

The SmartRoute-AI backend requires true production readiness. While the architectural decoupling (Streamlit split, Qdrant native hybrid search) is complete, there is a lack of automated deployment checks (CI/CD), strict database schema management (migrations), and observability. This change introduces these essential enterprise features. 

## Goals / Non-Goals

**Goals:**
- Automate database migrations using Alembic, applying existing migrations (e.g. `68f612722729_create_query_logs_table.py`) on app startup or deployment.
- Integrate Langfuse natively into the pipeline for end-to-end trace logging and observability of LLM cost/latency/quality metrics.
- Establish a robust CI/CD pipeline using GitHub Actions to enforce code quality (flake8/black) and run pytest on all PRs.

**Non-Goals:**
- Setting up complex multi-environment CD (Continuous Deployment) infrastructure like Kubernetes or Terraform. We will limit the pipeline to Continuous Integration (CI).
- Replacing Qdrant or Supabase; we will use the existing infrastructure choices.

## Decisions

- **Alembic for Migrations**: Alembic is already configured in `alembic.ini`. We will ensure it is run automatically either through a pre-start script in Docker or documented for standard deployment.
- **Langfuse over Sentry for AI Telemetry**: Sentry is great for error tracking, but Langfuse is purpose-built for LLM telemetry, cost tracking, and prompt tracing, which fits the `SmartRoute-AI` cost-optimization mandate perfectly.
- **GitHub Actions for CI**: It is the industry standard for open-source and modern enterprise projects. A standard `.github/workflows/ci.yml` will be added.

## Risks / Trade-offs

- **Risk**: Langfuse integration might add latency to critical API paths.
  **Mitigation**: Use Langfuse's async integration or background tasks (which Langfuse SDK handles naturally via background threads) to avoid blocking FastAPI workers.
- **Risk**: Automated Alembic migrations on startup in a multi-instance deployment can cause race conditions.
  **Mitigation**: For this phase, we will provide a `run_migrations.sh` or integrate it as a single init-container step in Docker Compose.
