# SmartRoute-AI

**Cost-optimized LLM inference gateway with ML-based query routing and RAG.**

🚀 **Live Deployment:** [Click Here](https://smartroute-dashboard.onrender.com/)

---

## Problem

Every LLM call goes through the same expensive, high-capability model even when the question is _"What is Python?"_ — a query any 9B model can answer correctly. SmartRoute-AI fixes that.

### Features
A **LightGBM classifier** (19 lexical + semantic features, `n_estimators=50`, `max_depth=4`, `num_leaves=15`) reads each incoming query and routes it to the smallest model that can handle it:

| Complexity | Model (`cost_optimized` strategy) | Max latency target |
|---|---|---|
| simple | `nvidia/nemotron-nano-9b-v2:free` | 3.0 s |
| medium | `openai/gpt-oss-20b:free` | 5.0 s |
| complex | `google/gemma-4-31b-it:free` | 10.0 s |

Uncertain `complex` classifications (confidence < 0.75) are demoted to `medium` via cost-biased hysteresis — conservative by design. Three routing strategies are configurable at query time: `cost_optimized` (default), `quality_first`, `balanced`.

On top of routing the system also provides:
- **RAG pipeline** — Qdrant native hybrid search (dense + sparse RRF fusion), HuggingFace cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with keyword-overlap local fallback, 500-token chunks with 50-token overlap.
- **Semantic cache** — query embeddings stored in Qdrant, payloads in Redis; hits return instantly without calling any LLM. Cache similarity threshold: `0.95`. TTL: 7 days.
- **Multi-turn memory** — conversation history stored in Redis, injected per `session_id`.
- **Budget enforcement** — atomic `INCRBYFLOAT` on Redis; daily $10, weekly $50, monthly $200 hard limits (configurable in `config/routing.yaml`). Alerts at 80% of any limit.
- **Guardrails** — 21 regex patterns for prompt injection blocking; 500-character hard query limit.
- **Full observability** — OpenTelemetry traces (OTLP, gRPC or HTTP) → LangFuse, per-query cost/token/latency logged to Supabase PostgreSQL.
- **RAGAS evaluation** — `faithfulness`, `answer_relevancy`, `context_recall`, `context_precision`; pass threshold set at 0.70 in `scripts/run_eval.py`.

---

## System Architecture

```mermaid
graph TD
    Client([Client Application]) -->|HTTPS Request| Gateway[API Gateway]
    
    subgraph Security & Ingestion Layer
        Gateway --> Guardrails[Input Guardrails]
        Guardrails --> CacheCheck{Semantic Cache Hit?}
    end

    CacheCheck -- Yes --> CacheStore[(Redis Cache)]
    CacheStore -->|Instant Response| Client

    CacheCheck -- No --> Router[Query Complexity Router]

    subgraph Routing & Budget Layer
        Router --> Hysteresis[Hysteresis Adjuster]
        Hysteresis --> BudgetCheck{Within Budget?}
        BudgetCheck -- Exceeded --> Fallback[Fallback Model]
        BudgetCheck -- Approved --> StrategyEngine[Routing Strategy Engine]
    end

    subgraph Retrieval Augmented Generation
        StrategyEngine --> RAGCheck{RAG Enabled?}
        RAGCheck -- Yes --> VectorSearch[Hybrid Vector Search]
        VectorSearch --> Reranker[Re-Ranker]
        Reranker --> ContextAssembler[Context Assembler]
        RAGCheck -- No --> ContextAssembler
    end

    subgraph Resilience & LLM Execution
        ContextAssembler --> CircuitBreaker[Circuit Breaker]
        CircuitBreaker --> LLMGateway[LLM Gateway]
        Fallback --> LLMGateway
        LLMGateway --> ResponseGen[Response Generator]
    end

    subgraph Observability & Analytics
        ResponseGen --> Database[(PostgreSQL Database)]
        ResponseGen --> Tracing[LangFuse Tracing]
        ResponseGen --> CacheWrite[Cache Updater]
    end

    ResponseGen --> Client
```

---

## End-to-End Query Lifecycle & State Machine

```mermaid
sequenceDiagram
    autonumber
    participant Client as Client Application
    participant Gateway as API Gateway
    participant Cache as Semantic Cache
    participant Router as Complexity Router
    participant Budget as Budget Manager
    participant RAG as Vector RAG Engine
    participant LLM as LLM Gateway
    participant DB as PostgreSQL Database

    Client->>Gateway: Submit Query Request
    Gateway->>Gateway: Sanitize & Validate Input
    Gateway->>Cache: Lookup Query Embedding
    alt Cache Hit
        Cache-->>Gateway: Return Cached Payload
        Gateway-->>Client: Stream Cached Response
    else Cache Miss
        Gateway->>Router: Classify Query Complexity
        Router-->>Gateway: Return Complexity & Confidence
        Gateway->>Budget: Validate Budget Limit
        Budget-->>Gateway: Budget Approved
        opt RAG Retrieval Enabled
            Gateway->>RAG: Execute Hybrid Vector Search
            RAG->>RAG: Re-Rank Context Chunks
            RAG-->>Gateway: Return Retrieved Context
        end
        Gateway->>LLM: Dispatch Query to Routed Model
        LLM-->>Gateway: Stream Response Tokens
        Gateway-->>Client: Forward Response Stream
        Gateway->>DB: Log Metrics & Token Costs
        Gateway->>Cache: Update Semantic Cache
    end
```

---

## RESTful API Endpoints

All business endpoints are versioned under `/v1` and require JWT Bearer Authentication (`Authorization: Bearer <TOKEN>`). Rate limiting is enforced per IP via SlowAPI.

| HTTP Method | Endpoint | Rate Limit | Auth Required | Description |
|---|---|---|---|---|
| `GET` | `/health` | Unthrottled | No | Health check probe for Docker/Render container status and DB/Redis connectivity. |
| `GET` | `/` | Unthrottled | No | Root endpoint returning service status, version (`2.0.0`), and endpoint index. |
| `POST` | `/v1/query` | `30/min` | Yes | Synchronous end-to-end inference processing query routing, budget, RAG, and execution. |
| `POST` | `/v1/query/stream` | `30/min` | Yes | Server-Sent Events (SSE) streaming endpoint returning metadata, tokens, and cost breakdown. |
| `POST` | `/v1/query/batch` | `10/min` | Yes | Concurrent batch execution processing up to 10 queries per request payload. |
| `GET` | `/v1/stats` | `60/min` | Yes | Aggregate cost, token consumption, latency, and model breakdown analytics over $N$ days. |
| `GET` | `/v1/savings` | `60/min` | Yes | Calculates total financial cost savings relative to a $0.15 baseline LLM cost per query. |
| `GET` | `/v1/budget` | `60/min` | Yes | Upstash Redis budget status detailing daily ($10), weekly ($50), and monthly ($200) utilization. |
| `GET` | `/v1/models` | Unthrottled | Yes | Returns list of configured OpenRouter models and currently initialized model instances. |
| `DELETE` | `/v1/memory/{session_id}` | Unthrottled | Yes | Flushes conversation turn history for a given multi-turn session ID from Redis. |
| `POST` | `/v1/index` | Unthrottled | Yes | Triggers background document chunking and hybrid vector indexing for files in `data/documents`. |
| `GET` | `/v1/documents` | Unthrottled | Yes | Lists all indexed document files currently stored in `data/documents`. |
| `DELETE` | `/v1/documents/{filename}` | Unthrottled | Yes | Deletes a document file, purges matching Qdrant vector points, and flushes semantic cache. |
| `DELETE` | `/v1/documents` | Unthrottled | Yes | Clears all documents, resets the Qdrant vector collection, and flushes Redis cache. |

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI 0.109+, Uvicorn, SlowAPI rate limiter (30 req/min query, 10 req/min batch) |
| LLM routing | LightGBM 4.0+, scikit-learn 1.4+, fastembed (`BAAI/bge-small-en-v1.5`) |
| LLM inference | OpenRouter via `AsyncOpenAI` (base_url: `https://openrouter.ai/api/v1`) |
| RAG | LangChain, Qdrant Cloud (hybrid dense+sparse), HuggingFace Inference API |
| Semantic cache | Qdrant + Upstash Redis (7-day TTL) |
| Cost/budget DB | Supabase PostgreSQL via SQLAlchemy + Alembic migrations |
| Observability | OpenTelemetry SDK → LangFuse (OTLP HTTP/gRPC), structured JSON logs |
| Auth | HS256 JWT (`PyJWT`) |
| Dashboard | Streamlit 1.31+, Plotly |
| CI/CD | GitHub Actions → GHCR Docker images → Render deploy hook |
| Python | 3.10 (pinned in `.python-version` and `pyproject.toml`) |

---

## Key Metrics

| Metric | Value | Description / Benchmark |
|---|---|---|
| **Classifier Test Accuracy** | **95.49%** | Evaluated on 731 stratified test queries (3,651 total dataset) |
| **Routing Macro F1-Score** | **0.9550** | `simple`: 0.975 \| `medium`: 0.941 \| `complex`: 0.949 |
| **RAGAS Quality Threshold** | **0.70+** | Minimum score across Faithfulness, Relevancy, Recall, & Precision |

---

## Setup

```bash
# 1. Clone and enter
git clone https://github.com/JavithNaseem-J/SmartRoute-AI.git
cd SmartRoute-AI

# 2. Install (Python 3.10 required)
pip install uv
uv pip install -e .

# 3. Configure environment
cp .env.example .env
# Fill in .env — required keys:
#   OPENROUTER_API_KEY   → https://openrouter.ai
#   SUPABASE_JWT_SECRET  → any string ≥ 32 chars
#   HF_TOKEN             → https://huggingface.co/settings/tokens
#   DATABASE_URL         → postgresql://... (Supabase free tier works)
#   REDIS_URL            → redis://... (Upstash free tier works)
#   QDRANT_URL           → https://... (Qdrant Cloud free tier works)
#   QDRANT_API_KEY       → from Qdrant Cloud dashboard

# 4. Run DB migrations
alembic upgrade head

# 5. Train / retrain the complexity classifier
python scripts/train_classifier.py
# Saves model to models/classifiers/complexity_classifier.pkl

# 6. Start the API backend
uvicorn api.main:app --host 0.0.0.0 --port 8000
# or: python -m api.main

# 7. Start the Streamlit dashboard (separate terminal)
streamlit run app.py        # → http://localhost:8501

# 8. Run the test suite
pytest tests/ -v

# 9. Run RAGAS RAG evaluation (requires indexed documents)
python scripts/run_eval.py


```

### Docker

```bash
docker compose up --build
# API → http://localhost:8000   Dashboard → http://localhost:8501
```

The `Dockerfile` has two targets (`api`, `dashboard`). The build step automatically runs `scripts/train_classifier.py`, so the classifier is baked into the image. Health checks run every 30 s with a 60 s start-up grace period.

---

## Deployment

Deployed on **Render** (Singapore region, free plan) via `render.yaml` — two services: `smartroute-api` (port 8000) and `smartroute-dashboard` (port 8501). `alembic upgrade head` runs as a pre-deploy command on every deploy.

GitHub Actions CI (`.github/workflows/ci.yml`):
1. Lint with `ruff`
2. Type-check with `mypy`
3. Test with `pytest`
4. Build and push Docker images to GHCR
5. Trigger Render deploy hook

---

## Project Structure

```
SmartRoute-AI/
├── api/main.py                  # FastAPI app — all /v1/* routes (rate-limited, JWT-gated)
├── app.py                       # Streamlit dashboard (inference console, cost analytics, budget)
├── config/
│   ├── routing.yaml             # Strategy definitions, budget limits, reference queries
│   └── models.yaml              # Model registry with cost per 1k tokens
├── src/
│   ├── routing/                 # LightGBM classifier, feature extraction (19 features), router
│   ├── pipeline/inference.py    # Full async pipeline orchestration
│   ├── retrieval/               # Qdrant indexer, hybrid retriever, HF reranker, semantic cache
│   ├── models/                  # OpenRouter async LLM wrapper (retry + circuit breaker)
│   ├── cost/                    # CostTracker (Supabase) + BudgetManager (Redis)
│   ├── memory/                  # Conversation history (Redis)
│   ├── evaluation/              # RAGAS eval harness
│   └── utils/                   # Guardrails, circuit breaker, alerting, OTel tracing
├── scripts/
│   ├── train_classifier.py      # LightGBM training + evaluation + save to .pkl
│   ├── generate_training_data.py
│   ├── generate_centroids.py
│   └── run_eval.py              # RAGAS evaluation runner
├── tests/                       # 9 pytest test files
├── data/
│   ├── training/synthetic_queries.csv
│   └── documents/               # Drop PDFs / TXTs here for RAG indexing
├── models/classifiers/          # complexity_classifier.pkl (231 KB, pre-trained)
├── alembic/                     # DB migration scripts
├── Dockerfile                   # Multi-stage build (base → api / dashboard)
├── docker-compose.yml
└── render.yaml                  # One-click Render deployment blueprint
```

---

## Future Work

- **Real benchmark numbers** — classifier accuracy was measured on 3,651 synthetic & rule-generated samples. Evaluating on a held-out human-labeled query set will give real-world domain confidence intervals.
- **Paid model cost tracking** — currently all three models are on `:free` tiers; the cost-per-1k fields in `config/models.yaml` are all 0.0 and will need updating once paid tiers are used.
- **Token-accurate counting** — `count_tokens` uses `len(text) // 4` (a heuristic). Replace with `tiktoken` or a model-specific tokenizer for precise billing.
- **Streaming batch endpoint** — `/v1/query/batch` returns full JSON; streaming SSE is not yet supported for batch.
- **Multi-collection RAG** — all indexed documents share a single `smartroute_docs` Qdrant collection; namespace isolation per user/project would enable multi-tenant deployments.

---

## License

This project is licensed under the [MIT License](file:///f:/DSML/SmartRoute-AI/LICENSE).
