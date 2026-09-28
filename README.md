# SmartRoute-AI

SmartRoute-AI is a multi-tenant LLM routing and document-question-answering application. It combines a FastAPI backend, React chat interface, LightGBM complexity routing, Qdrant retrieval, Redis budget enforcement and memory, PostgreSQL analytics, and citation-aware streamed answers.

Live deployment: [smartroute-ai-ev2b.onrender.com](https://smartroute-ai-ev2b.onrender.com/)

## Current Behavior

- One active LLM provider at a time: `groq` or `openrouter`.
- One authentication variable for either provider: `LLM_API_KEY`.
- Internal routes use logical tiers: `economy`, `balanced`, `quality`, and `fallback`.
- `quality_first` always selects the active provider's `quality` tier, regardless of classifier confidence.
- Provider failures are terminal errors or explicit fallback attempts; error text is never presented as an LLM answer.
- Document records, Qdrant filters, semantic cache entries, cost analytics, and daily budget counters are tenant-scoped.
- RAG answers include validated `[C#]` markers. The frontend renders compact citation buttons with filename, page or section, and an evidence excerpt.
- Deleted documents are excluded from retrieval by filtering Qdrant with the authenticated user's current active storage paths.

## Provider Configuration

Set exactly one provider and one key:

```env
LLM_PROVIDER=groq
LLM_API_KEY=gsk_...
```

To use OpenRouter instead:

```env
LLM_PROVIDER=openrouter
LLM_API_KEY=sk-or-v1-...
```

Both providers use `LLM_API_KEY`; the provider is never guessed from the key prefix.

Model IDs and pricing are defined in `config/models.yaml`. Groq uses production `openai/gpt-oss-20b` and `openai/gpt-oss-120b` endpoints. Run this after changing provider configuration:

```bash
uv run python scripts/provider_predeploy.py
```

The command authenticates with the active provider and verifies every configured model ID. It does not generate text.

## RAG Pipeline

1. Upload PDF, UTF-8/UTF-16/Windows-1252 TXT, or Markdown.
2. Store the original object in Supabase Storage.
3. Extract readable text and split it into 500-character chunks with 50-character overlap.
4. Create local `BAAI/bge-small-en-v1.5` dense embeddings with FastEmbed.
5. Upsert tenant and source metadata into Qdrant, then verify the indexed chunk count.
6. At query time, filter retrieval by authenticated user and active document storage paths.
7. Rerank candidates, assemble `[C#]` context, and validate generated citation markers against retrieved evidence.

Dense retrieval is the production default:

```env
ENABLE_SPARSE_EMBEDDINGS=false
RERANKER_MODE=local
```

Set `ENABLE_SPARSE_EMBEDDINGS=true` only when the sparse FastEmbed model is available and the Qdrant collection has a compatible `sparse` vector definition. `RERANKER_MODE=local` uses deterministic keyword overlap after Qdrant retrieval; `RERANKER_MODE=disabled` preserves Qdrant order. Neither mode needs a separate reranker API key.

Scanned image-only PDFs require OCR before upload. The application does not currently perform OCR.

## Routing And Evaluation

The complexity classifier uses 19 lexical and semantic features. Its committed artifact is built with scikit-learn `1.7.2` and artifact schema version `2`.

Current reproducible classifier evaluation:

| Measure | Result |
|---|---:|
| Training examples | 540 |
| Unique training examples | 540 |
| Held-out human-authored examples | 30 |
| Train/evaluation overlap | 0 |
| Held-out accuracy | 96.67% |
| Held-out macro F1 | 0.9666 |

The held-out set is intentionally small, so these values are regression signals rather than a broad real-world performance claim. Training metadata and the confusion matrix are stored in `models/classifiers/complexity_classifier.metrics.json`.

The version-controlled RAG benchmark in `data/evaluation/rag_eval.json` checks retrieval hit rate, reciprocal rank, and citation metadata without consuming LLM tokens:

```bash
uv run python scripts/run_eval.py
```

An optional RAGAS harness remains in `src/evaluation/ragas_eval.py` for judge-based evaluation against documents already indexed for a dedicated evaluation user.

## Budget And Analytics

- Redis uses an atomic tenant-scoped daily reservation key.
- Redis failure is fail-closed: paid inference returns `budget_unavailable` instead of bypassing the limit.
- The daily hard limit is configured in `config/routing.yaml`.
- Weekly and monthly values are analytics/reporting limits; they are not currently atomic hard gates.
- PostgreSQL query logs include `user_id`; `/v1/stats`, `/v1/savings`, and `/v1/budget` return only the authenticated tenant's data.
- Token counts use provider usage for non-streaming calls when available. Streaming estimates tokens with `len(text) // 4`, so streaming cost remains approximate.

## API

Business endpoints require a JWT bearer token.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Cheap process liveness |
| `GET` | `/ready` | Redis, Qdrant, PostgreSQL, pipeline, and provider configuration readiness |
| `GET` | `/version` | Deployment commit and build identity |
| `POST` | `/v1/query` | Non-streaming query |
| `POST` | `/v1/query/stream` | SSE query stream with terminal success/failure payload |
| `POST` | `/v1/query/batch` | Up to ten concurrent queries |
| `GET` | `/v1/models` | Active provider and logical-to-physical model mapping |
| `GET` | `/v1/stats` | Tenant-scoped usage analytics |
| `GET` | `/v1/savings` | Tenant-scoped baseline comparison |
| `GET` | `/v1/budget` | Tenant-scoped budget status and enforcement state |
| `POST` | `/v1/documents/upload` | Store, index, and verify documents |
| `GET` | `/v1/documents` | List active tenant documents |
| `DELETE` | `/v1/documents/{filename}` | Remove one document from vectors, storage, and metadata |
| `DELETE` | `/v1/documents` | Clear all active tenant documents |

Non-streaming failures use non-200 HTTP statuses. Streaming responses cannot change HTTP status after headers are sent, so they end with `done.result.success=false` and a stable error code.

## Local Setup

Requirements: Python 3.10, Node.js 22, PostgreSQL/Supabase, Redis, and Qdrant.

```bash
uv sync --all-groups
cd frontend
npm ci
cd ..

cp .env.example .env
# Fill in provider and infrastructure credentials.

uv run alembic upgrade head
uv run python scripts/provider_predeploy.py
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
cd frontend
npm run dev -- --port 5173
```

## Verification

```bash
uv run pytest
uv run ruff check src api tests scripts
uv run ruff format --check src api tests scripts
uv run mypy src/ api/
uv run python scripts/run_eval.py

cd frontend
npm run lint
npm run typecheck
npm run build
npx playwright install chromium
npm run test:e2e
```

The browser suite mocks provider traffic and verifies Quality First selection, active-document isolation, streamed citation details, terminal stream failures, and mobile viewport containment. It consumes no LLM tokens.

## Deployment

`render.yaml` declares one Docker web service and a private Render Key Value service. The Blueprint injects the Key Value internal connection string as `REDIS_URL`; no Redis URL needs to be copied into Render manually. The free Key Value plan is suitable for this demonstration but loses cache, memory, and budget-counter state when it restarts. Use a persistent paid plan for production budget enforcement.

`scripts/start_api.sh` applies Alembic migrations before Uvicorn starts. The image builds the frontend and packages the committed classifier artifact; it does not retrain during deployment.

Set `LLM_PROVIDER` and `LLM_API_KEY` in Render after deployment configuration is updated. `/health` can remain healthy during a dependency outage, while `/ready` correctly returns `503` until Redis, Qdrant, PostgreSQL, and the provider configuration are ready.

## Known Limits

- The classifier's held-out set has only 30 examples.
- Local reranking is lexical, not a local neural cross-encoder.
- Sparse retrieval requires more memory and a compatible Qdrant collection schema.
- Streaming token accounting is approximate.
- Uploaded scanned PDFs need external OCR.
- Provider availability and model catalogs can change; run the predeploy validator whenever provider models are updated.

## License

This project is licensed under the [MIT License](LICENSE).
