# SmartRoute-AI Production Reliability and Provider Design

## Objective

Make SmartRoute-AI defensible and reliable as a production-style portfolio system while
supporting one active OpenAI-compatible LLM provider at a time. The first supported
providers are GroqCloud and OpenRouter. The design also repairs the routing, evaluation,
tenant isolation, retrieval, error handling, testing, and documentation gaps found in the
end-to-end audit.

## Configuration Contract

The active provider is selected explicitly. API keys are never detected by prefix.

```env
LLM_PROVIDER=groq
LLM_API_KEY=replace-me
```

`LLM_PROVIDER` selects a provider definition with a base URL and logical model-tier map.
`LLM_API_KEY` authenticates only with that provider. An OpenRouter key cannot authenticate
with GroqCloud and a GroqCloud key cannot authenticate with OpenRouter. A temporary
`OPENROUTER_API_KEY` fallback is allowed only when `LLM_PROVIDER=openrouter`, with a
deprecation warning.

Routing uses provider-independent tiers: `economy`, `balanced`, `quality`, and `fallback`.
Each provider maps those tiers to model IDs it supports. Provider configuration is validated
at startup and exposed in readiness without revealing credentials.

## Runtime Architecture

`ModelManager` resolves a logical tier through the active provider configuration and creates
an `OpenAICompatibleModel`. The model adapter owns OpenAI-compatible synchronous generation,
streaming, token usage, retry policy, and per-model circuit breaking. Provider exceptions are
raised as typed runtime failures; they are never emitted as ordinary answer text.

The router returns a logical tier rather than a provider-specific model ID. Cost Optimized
may use confidence-based escalation. Balanced may move upward but never below its configured
tier. Quality First always selects the `quality` tier and cannot be demoted by classifier
confidence, budget fallback, or provider-error fallback. With only one active provider,
fallback means another configured model at that provider, not another provider.

## Failure Contract

Non-streaming provider exhaustion returns a failed pipeline result and an appropriate API
status. Streaming emits one terminal failed `done` event after discarding any partial failed
attempt. A failed fallback cannot be recorded, cached, billed, or displayed as a successful
answer. The frontend uses the terminal result's `success` and `answer` fields instead of
inventing `Response received.`.

Readiness reports LLM configuration, Postgres, Qdrant, and Redis separately. Redis remains a
required production dependency because it provides hard budget enforcement and conversation
memory. When Redis is unavailable, readiness fails and the budget response explicitly states
that enforcement is unavailable; it must not claim hard-limit protection.

## Tenant Isolation

Every query log includes `user_id`. Statistics, savings, and budget reporting require the
authenticated user ID and filter all database aggregates by it. A migration adds the column
and indexes the timestamp/user combination. Existing rows remain unowned and are excluded
from tenant views.

Documents, vectors, cache entries, memory, and citations retain the current user isolation.

## Classifier and Evaluation

The training corpus is deduplicated before splitting. Test-specific prompt injection is
removed. Query text groups are kept wholly within one split, and the evaluation set contains
independently authored examples rather than template duplicates. Training writes a versioned
artifact bundle containing the model, scaler, encoder, feature schema, dependency versions,
dataset fingerprint, split counts, and metrics.

CI validates artifact compatibility and feature order. Published metrics come only from the
held-out evaluation report committed with the artifact. If a valid retrained artifact cannot
be produced in the current environment, routing falls back to deterministic rules and the
documentation makes that limitation explicit.

## Retrieval and Reranking

Dense retrieval is always available through FastEmbed and Qdrant. Sparse vectors are an
explicit optional mode. When enabled, upload verification confirms both dense and sparse
vectors, and retrieval uses Qdrant fusion. Readiness and response diagnostics state whether
retrieval is `dense` or `hybrid`; documentation does not call dense-only operation hybrid.

The reranker has an explicit mode: `local`, `huggingface`, or `disabled`. External reranking
has bounded timeouts and a tested local fallback. Tests await all async mocks and verify
ordering, failure fallback, and diagnostics.

## Tests and Evaluation

Backend tests cover provider selection, missing credentials, model-tier resolution, Quality
First invariants, primary/fallback stream failure, tenant-scoped telemetry, classifier split
isolation, artifact metadata, dense/hybrid diagnostics, and reranker fallback.

Playwright tests cover authentication, strategy selection, RAG request payloads, upload/list/
delete UI behavior, citations, provider failure display, analytics, and mobile layout. Browser
tests use deterministic API interception in CI; bounded live probes remain separate.

A small version-controlled RAG evaluation dataset tests answer relevance, context precision,
context recall, faithfulness, citation filename/page accuracy, and no-source honesty. The
deterministic retrieval/citation checks run in CI; optional RAGAS judge scoring requires a
configured judge provider and is reported separately.

## Deployment and Compatibility

`.env.example`, Render configuration, CI, and the README use `LLM_PROVIDER` and `LLM_API_KEY`.
Application version comes from project metadata and is shared by `/health`, `/ready`, and
package metadata. Deployment continues to expose commit SHA and build time.

The final verification gate is: backend tests, lint, formatting, mypy, migration check,
frontend lint/typecheck/unit tests/build/audit, Playwright, classifier leakage checks, RAG
evaluation, local API acceptance, and bounded production probes. Production provider and
Redis checks are expected to remain blocked until the user updates those secrets.

## Out of Scope

- Concurrent multi-provider failover or multiple active provider keys.
- Key-prefix provider detection.
- Creating or purchasing external Redis, GroqCloud, or OpenRouter accounts.
- Claiming external judge scores when no judge API was executed.
