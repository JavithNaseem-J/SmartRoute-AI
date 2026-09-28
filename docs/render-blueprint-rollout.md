# Render Deployment Rollout

Use this rollout for the single-service Render app after the CI-to-Render deployment workflow is merged.

## Canonical Blueprint

The root `render.yaml` defines a Docker web service and a private Redis-compatible Render Key Value service:

- `smartroute-ai`
- `smartroute-redis`

Do not delete or recreate the live service as part of this pipeline change. The service must keep `autoDeploy: false` so GitHub Actions decides which exact commit is safe to deploy.

## Required Render Environment

Confirm these environment variables are ready in Render:

- `LLM_PROVIDER` (`groq` or `openrouter`)
- `LLM_API_KEY` (the selected provider's key)
- `SUPABASE_JWT_SECRET`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_STORAGE_BUCKET`
- `RERANKER_MODE` (`local` by default)
- `DATABASE_URL`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `APP_PUBLIC_URL`
- optional LangFuse/OTEL keys

The Blueprint supplies `REDIS_URL` from `smartroute-redis`. Do not enter a separate external Redis URL. The local reranker does not need an API key.

Use:

```env
SUPABASE_STORAGE_BUCKET=smartroute-documents
APP_PUBLIC_URL=https://<new-render-service-url>
LLM_PROVIDER=groq
RERANKER_MODE=local
```

## Rollout Steps

1. Apply or sync the root `render.yaml` to the existing `smartroute-ai` service.
2. Confirm `smartroute-redis` is available in the Singapore region and `REDIS_URL` references its internal connection string.
3. Confirm the web service uses the root `Dockerfile`.
4. Confirm the service health check path is `/health`.
5. Confirm automatic deploys are disabled.
6. Create or copy the service deploy hook URL.
7. Add the deploy hook to GitHub as the `RENDER_DEPLOY_HOOK_URL` secret.
8. Add the public app URL to GitHub as the `PRODUCTION_BASE_URL` repository variable or `production` environment variable.
9. Validate the selected provider and configured models from a secure environment:

```bash
uv run python scripts/provider_predeploy.py
```

10. Push to `main` and let GitHub Actions run the `CI` workflow.
11. Confirm `Deploy Render` starts only after `CI` succeeds.
12. Confirm the deployment artifact shows `/version` returning the exact deployed SHA.
13. Check `https://<service>/health`.
14. Check `https://<service>/ready` and confirm every component is `ok`.
15. Open the React app at the service URL.
16. Confirm startup logs show Alembic at the latest revision.
17. Upload a small `.txt` file and confirm it appears in Supabase Storage.
18. Ask a RAG query and open its citation button to verify filename and location.

## Notes

- The Blueprint uses `plan: free`, so it does not include `preDeployCommand`.
- The free Key Value plan is non-persistent and can restart. Use a persistent paid plan when budget-counter durability is a production requirement.
- `scripts/start_api.sh` applies idempotent Alembic migrations before Uvicorn starts because the free plan has no pre-deploy command.
- The container runs one Uvicorn process and binds to Render's injected `$PORT`.
- The deploy workflow calls the Render deploy hook with `ref=<exact-commit-sha>` and then verifies production `/version`.
