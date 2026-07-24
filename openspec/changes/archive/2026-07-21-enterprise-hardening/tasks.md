## 1. Decoupled Database Migrations

- [x] 1.1 Remove `alembic upgrade head` from `Dockerfile.api` `CMD`
- [x] 1.2 Update `render.yaml` to include `preDeployCommand: "alembic upgrade head"` under the `smartroute-api` web service configuration

## 2. API Resilience via Tenacity

- [x] 2.1 Add `tenacity` dependency to `requirements.txt` and `pyproject.toml`
- [x] 2.2 Decorate the asynchronous Generation and Streaming methods in `src/models/nvidia_model.py` with `@retry` (exponential backoff)
- [x] 2.3 Add a fallback/exception handling block to gracefully return standard 503 error payloads to clients upon maximum retry exhaustion
- [x] 2.4 Update tests in `test_router.py` or create a `test_resilience.py` to assert retry logic

## 3. JWT-based Client Authentication

- [x] 3.1 Update `src/utils/security.py` to validate JWTs from `Authorization: Bearer <token>` headers instead of matching `SMARTROUTE_API_KEY`
- [x] 3.2 Add PyJWT cryptography dependencies if necessary to decode and verify Supabase signatures locally
- [x] 3.3 Update `api/main.py` dependencies to enforce the new JWT authentication mechanism
- [x] 3.4 Update `tests/test_api.py` to reflect JWT token expectations instead of standard API keys

## 4. Semantic Caching (Redis + Qdrant)

- [x] 4.1 Define the `SemanticCache` class inside `src/retrieval/cache.py` (or create a new `semantic_cache.py`)
- [x] 4.2 Initialize a `semantic-cache` vector collection in the Qdrant Cloud cluster setup
- [x] 4.3 Wrap the LLM execution flow in `src/pipeline/inference.py` to embed incoming queries, check `SemanticCache` for cosine similarity > 0.95, and return early on hits
- [x] 4.4 Set up an asynchronous task to insert LLM response payloads into Redis and vector embeddings into Qdrant on cache misses
- [x] 4.5 Write comprehensive unit tests in `tests/test_semantic_cache.py` to validate cache hits and misses
