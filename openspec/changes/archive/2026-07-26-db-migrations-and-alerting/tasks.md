## 1. Setup Alembic

- [x] 1.1 Install Alembic (if missing) and initialize the Alembic environment (`alembic init alembic`)
- [x] 1.2 Configure `alembic.ini` and `alembic/env.py` to connect to the database via SQLAlchemy
- [x] 1.3 Update `alembic/env.py` to import SQLAlchemy models so autogenerate works correctly

## 2. Baseline Migration

- [x] 2.1 Run `alembic revision --autogenerate -m "Initial schema baseline"` to create the first migration
- [x] 2.2 Verify the generated migration accurately reflects the current state of the database models
- [x] 2.3 Run `alembic upgrade head` to apply/stamp the baseline migration on the development database

## 3. Implement Alerting

- [x] 3.1 Create an `alerting.py` utility in `src/utils/` to handle posting webhook messages (e.g., Slack)
- [x] 3.2 Add environment variables for the webhook URLs in `.env.example` and the configuration module
- [x] 3.3 Integrate the alerting utility into FastAPI's global exception handler in `api/main.py`
- [x] 3.4 Update `.github/workflows/deploy.yml` and `ci.yml` to trigger a webhook notification if steps fail

## 4. Verification

- [x] 4.1 Run the FastAPI server locally and simulate an unhandled exception to verify the webhook is triggered
- [x] 4.2 Review CI/CD pipeline locally to ensure the webhook step is syntactically valid
- [x] 4.3 Run `alembic current` and `alembic history` to confirm the baseline is set correctly (e.g., syntax error on a dummy branch) to verify CI alerting works
