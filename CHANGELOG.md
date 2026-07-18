# Changelog

All notable changes to SmartRoute-AI are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [2.1.0] — 2026-07-18

### Added
- **API versioning** — all business endpoints are now served under `/v1/` prefix
  (`/v1/query`, `/v1/stats`, `/v1/budget`, `/v1/savings`, `/v1/models`, `/v1/memory/{id}`)
- **Startup env-var validator** — app exits immediately on boot with a clear,
  actionable error message if any required cloud credential is missing
- **Pre-commit hooks** — `ruff` linting and formatting enforced on every `git commit`
- **mypy type checking** added to CI pipeline (runs in parallel with lint)

### Changed
- Legacy unversioned routes (`/query`, `/stats`, `/budget`) kept as deprecated
  aliases; hidden from `/docs`; will be removed in v3.0
- `FastAPI(lifespan=...)` replaces deprecated `@app.on_event("startup")`

### Fixed
- CI test job was missing `fakeredis` and `pytest-asyncio` — GitHub Actions
  pipeline was broken on test collection

---

## [2.0.0] — 2026-07-17

### Added
- **Full async architecture** — `InferencePipeline.run()` and all API endpoints
  are now `async def`, eliminating thread pool starvation under concurrent load
- **Qdrant Cloud** as the sole vector store (`QDRANT_URL` + `QDRANT_API_KEY` required)
- **Supabase PostgreSQL** as the sole database backend (`DATABASE_URL` required)
- **Upstash Redis** as the sole cache and budget store (`REDIS_URL` required)
- `BaseLLM` abstract interface for provider-agnostic LLM integration
- `GroqModel` upgraded to use `AsyncGroq` with exponential-backoff retry
- `render.yaml` for 1-click free-tier deployment to Render
- `tests/conftest.py` with global `fakeredis` and Qdrant mocks — CI never
  makes real cloud calls

### Removed
- **SQLite** fallback from `CostTracker` — cloud-only
- **ChromaDB** fallback from `VectorStore` — cloud-only; package removed from
  `requirements.txt`
- **In-memory dict** fallback from `ConversationMemory` and `RetrievalCache`
- **`run_in_executor` wrappers** from FastAPI endpoints — replaced by direct `await`

### Changed
- `BudgetManager` uses atomic Redis `INCRBYFLOAT` instead of SQLite transactions
- `ConversationMemory` is now shared across workers via Redis (multi-worker safe)
- All cloud services now **fail fast with actionable error messages** if env vars
  are absent rather than silently degrading

---

## [1.0.0] — 2026-07-11

### Added
- Initial production architecture: FastAPI + LangChain RAG pipeline
- Intelligent query routing via LightGBM classifier (complexity: simple/moderate/complex)
- Three routing strategies: `cost_optimized`, `quality_first`, `balanced`
- Hybrid retrieval: dense (ChromaDB) + sparse (BM25) with cross-encoder reranker
- `CostTracker` with SQLite backend for usage analytics
- `BudgetManager` with daily/weekly/monthly spending caps
- `ConversationMemory` with FIFO eviction for multi-turn sessions
- `RetrievalCache` (LRU) to avoid redundant vector DB queries
- API key authentication with `secrets.compare_digest` (timing-attack safe)
- Rate limiting via `slowapi` (30 req/min for query endpoints)
- Regex-based prompt injection guardrails
- GitHub Actions CI pipeline: lint → security → tests → Docker build
- Kubernetes manifests (`k8s/`) and `docker-compose.yml`
- Streamlit dashboard (`app.py`) for cost/usage visualisation
