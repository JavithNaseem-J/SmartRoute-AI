## 1. Database Migrations

- [x] 1.1 Review the existing Alembic migration file `68f612722729_create_query_logs_table.py` for correctness.
- [x] 1.2 Modify `docker-compose.yml` to run Alembic migrations automatically on startup (e.g. using a `command` override for the `api` container, or creating an init container).
- [x] 1.3 Validate that `query_logs` and other required tables are successfully created in the Supabase/PostgreSQL instance when the API starts.

## 2. Observability & Alerting

- [x] 2.1 Add an API `/health` GET route in `api/main.py` that verifies the connection status to PostgreSQL, Redis, and Qdrant.
- [x] 2.2 Install `langfuse` in `requirements.txt` and integrate the Langfuse callback handler into the Langchain inference pipeline.
- [x] 2.3 Ensure API responses stream seamlessly without being blocked by Langfuse observability telemetry.

## 3. CI/CD Pipeline

- [x] 3.1 Create `.github/workflows/deploy.yml` using GitHub Actions.
- [x] 3.2 Configure jobs for format checking (`black`, `isort`), testing (`pytest`), and building/pushing a Docker image to GitHub Container Registry.
- [x] 3.3 Configure the deployment step to trigger a webhook on Render (using `RENDER_DEPLOY_HOOK_URL`) once the image is pushed.
- [x] 3.4 Add jobs in `ci.yml` to run `flake8` and `black` for code linting and formatting.
- [x] 3.5 Add a job to test the Docker image build `docker-compose build` to ensure the environment is fully reproducible.
