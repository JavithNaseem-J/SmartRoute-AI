## Context

The SmartRoute-AI microservice architecture is moving from a prototype to an enterprise-grade production environment. The current state is vulnerable to auto-scaling race conditions (migrations in `Dockerfile.api`), incurs unnecessary inference costs for duplicate queries, lacks resilience against transient upstream API failures, and uses a single static API key for all clients. 

## Goals / Non-Goals

**Goals:**
- Eliminate migration race conditions and potential database corruption during horizontal scaling.
- Dramatically reduce API costs and latency for repeat queries by intercepting them before LLM inference.
- Ensure the API returns a graceful response or retries automatically when LLM providers fail.
- Implement granular, revocable client authentication.

**Non-Goals:**
- We are not rewriting the core routing algorithm or the `InferencePipeline` class architecture; we are wrapping/decorating it.
- We are not implementing a custom OAuth provider; we are leveraging Supabase JWTs which are natively supported by our existing infrastructure.

## Decisions

**1. Decoupling Migrations via Render Release Commands**
- *Decision*: Remove `alembic upgrade head` from `Dockerfile.api` and define it as the `preDeployCommand` in `render.yaml`.
- *Rationale*: Render guarantees the Release Command finishes before *any* new web instances are started or traffic is routed. This natively solves the multi-instance scaling race condition without complex Redis distributed locks.

**2. Semantic Caching via Upstash Redis + Qdrant**
- *Decision*: Introduce a `SemanticCache` class that embeds incoming queries via the existing `Embedder` and queries a dedicated Qdrant collection (`semantic-cache`). If cosine similarity > 0.95, it retrieves the cached response payload from Redis.
- *Rationale*: Storing vectors in Qdrant enables semantic matching (e.g., "What's smartroute?" matches "What is SmartRoute?"). Storing the actual large JSON response payloads in Redis prevents bloating the vector database and allows for native TTL expiration.

**3. API Resilience via Tenacity**
- *Decision*: Use the `tenacity` library to wrap the `agenerate` and `astream` methods in `NvidiaModel`.
- *Rationale*: `tenacity` provides robust exponential backoff decorators out-of-the-box (`@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))`). It is cleaner and more configurable than writing custom asyncio loops.

**4. Authentication via Supabase JWTs**
- *Decision*: Replace the static `SMARTROUTE_API_KEY` with a FastAPI dependency that decodes Supabase JWTs.
- *Rationale*: We already use Supabase for Postgres. Supabase provides a fully-featured Auth API. Validating their JWTs locally is fast (requires no database dip) and enables instant revocation at the auth provider level.

## Risks / Trade-offs

- [Risk] Semantic caching might return incorrect answers if the context drastically changes but the query is identical. → **Mitigation**: Implement a low TTL on the cache (e.g., 24 hours) or invalidate cache entries when underlying documents are updated.
- [Risk] Exponential backoff on LLM calls might hold the FastAPI connection open for too long (e.g., 10 seconds), causing timeouts. → **Mitigation**: Ensure FastAPI timeout configurations and client-side timeouts are aligned with the max retry budget.
- [Risk] Embedding the query for semantic caching adds latency (~200ms) to *every* query, even cache misses. → **Mitigation**: Use a blazing-fast, small embedding model (we already use HuggingFace Inference API) to minimize overhead.
