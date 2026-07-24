## Why

The current SmartRoute-AI architecture functions well under prototype load but contains multiple single points of failure, scaling bottlenecks, and cost inefficiencies that prevent it from being a true enterprise-grade system. Specifically, race conditions in database migrations during auto-scaling, high LLM inference costs due to duplicate identical queries, unhandled external API fragility, and coarse-grained monolithic security keys pose significant operational risks. We must address these to achieve zero-downtime scaling, near-zero latency/cost for repeat queries, resilient API integrations, and robust client access management.

## What Changes

- Decouple database migrations from the API container startup (`Dockerfile.api` `CMD`) into a dedicated pre-deployment or release script to prevent database race conditions and lock-contention when horizontal auto-scaling triggers.
- Introduce a Semantic Caching layer (via Upstash Redis + Qdrant) that intercepts incoming queries, computes similarity against previous queries, and short-circuits to return cached LLM responses if the cosine similarity exceeds a high threshold (e.g., > 0.95).
- Integrate the `tenacity` library to wrap external LLM and Embedding API calls in resilient exponential backoff/retry loops, gracefully failing over to smaller fallback models when primary providers (NVIDIA/Groq/HuggingFace) experience transient outages.
- **BREAKING**: Transition from a single, hardcoded `SMARTROUTE_API_KEY` to granular, scalable client authentication. Clients will authenticate via Supabase JWTs (JSON Web Tokens) or dynamic API keys, enabling targeted revocation without globally impacting system availability.

## Capabilities

### New Capabilities
- `semantic-caching`: Intercepts and caches LLM query/response pairs semantically using vector similarity to eliminate duplicate inference costs and reduce latency to ~15ms.
- `resilient-api-retry`: Implements automated exponential backoff and LLM model failovers using Tenacity to ensure 99.9% uptime despite third-party provider instability.
- `scalable-auth-jwt`: Enforces client-specific JWT-based authentication via Supabase, replacing the static global API key pattern.

### Modified Capabilities
- `database-migrations`: Modifies the application deployment requirements to run Alembic database migrations externally (via Render Release Commands) instead of at API container boot.

## Impact

- **API & Clients**: Breaking change for existing clients; all clients must migrate from the static `SMARTROUTE_API_KEY` to requesting and supplying standard JWT bearer tokens.
- **Dependencies**: New dependency `tenacity` will be added to the project for exponential backoff decorators.
- **Infrastructure**: Render deployment configuration (`render.yaml`) will be updated to utilize `preDeployCommand`/Release commands.
- **Cost & Latency**: Drastic reduction in API token usage for duplicate queries, alongside significantly reduced P95 latency profiles due to semantic cache hits.
