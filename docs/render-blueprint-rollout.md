# Render Deployment Rollout

Use this rollout for the single-service Render app after the CI-to-Render deployment workflow is merged.

## Canonical Blueprint

The root `render.yaml` defines one Docker web service:

- `smartroute-ai`

Do not delete or recreate the live service as part of this pipeline change. The service must keep `autoDeploy: false` so GitHub Actions decides which exact commit is safe to deploy.

## Required Render Environment

Confirm these environment variables are ready in Render:

- `OPENROUTER_API_KEY`
- `SUPABASE_JWT_SECRET`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_STORAGE_BUCKET`
- `HF_TOKEN`
- `DATABASE_URL`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `REDIS_URL`
- `APP_PUBLIC_URL`
- optional LangFuse/OTEL keys

Use:

```env
SUPABASE_STORAGE_BUCKET=smartroute-documents
APP_PUBLIC_URL=https://<new-render-service-url>
```

## Rollout Steps

1. Apply or sync the root `render.yaml` to the existing `smartroute-ai` service.
2. Confirm the service uses the root `Dockerfile`.
3. Confirm the service health check path is `/health`.
4. Confirm automatic deploys are disabled.
5. Create or copy the service deploy hook URL.
6. Add the deploy hook to GitHub as the `RENDER_DEPLOY_HOOK_URL` secret.
7. Add the public app URL to GitHub as the `PRODUCTION_BASE_URL` repository variable or `production` environment variable.
8. Run migrations manually before deploying if they have not already been applied:

```bash
uv run alembic upgrade head
```

9. Push to `main` and let GitHub Actions run the `CI` workflow.
10. Confirm `Deploy Render` starts only after `CI` succeeds.
11. Confirm the deployment artifact shows `/version` returning the exact deployed SHA.
12. Check `https://<service>/health`.
13. Check `https://<service>/ready`.
14. Open the React app at the service URL.
15. Upload a small `.txt` file and confirm it appears in Supabase Storage.
16. Ask a RAG query against the uploaded file.

## Notes

- The Blueprint uses `plan: free`, so it does not include `preDeployCommand`.
- Run `uv run alembic upgrade head` manually before deploying whenever migrations change.
- The container runs one Uvicorn process and binds to Render's injected `$PORT`.
- The deploy workflow calls the Render deploy hook with `ref=<exact-commit-sha>` and then verifies production `/version`.
