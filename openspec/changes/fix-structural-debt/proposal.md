## Why

The SmartRoute-AI codebase contains several instances of structural debt, including inconsistent patterns (e.g., using local JSON for BM25 while using Qdrant for dense vectors), duplicate logic (loading local embedding models redundantly), and dead code. Addressing these now will improve maintainability, reduce memory footprint, and ensure the system runs properly in a distributed cloud environment.

## What Changes

- Migrate the BM25 sparse index from a local JSON file to a persistent cloud store (or rely on Qdrant's sparse vectors) to ensure hybrid search works across distributed workers.
- Extract the SentenceTransformer embedding model into a shared singleton service so it isn't instantiated multiple times in memory.
- Standardize Qdrant database access to use `AsyncQdrantClient` across both `VectorStore` and `SemanticCache`, removing synchronous blocking code.
- Remove custom Langfuse header parsing in the OpenTelemetry tracing setup in favor of standard OTEL environment variables.
- Standardize Redis connection handling to use async clients across `ConversationMemory` and `BudgetManager`.
- Remove dead code such as `check_budget_full()` and `should_alert()` from `budget.py`.

## Capabilities

### New Capabilities
- None. This is a refactoring and technical debt reduction change.

### Modified Capabilities
- None. System behavior and requirements remain unchanged, only internal structural debt is being fixed.

## Impact

- `src/retrieval/retriever.py`, `src/retrieval/indexer.py`: BM25 logic updated.
- `src/retrieval/embedder.py`, `src/routing/features.py`: Embedding model logic unified.
- `src/retrieval/vector_store.py`, `src/retrieval/semantic_cache.py`: Qdrant client unified.
- `src/utils/tracing.py`: Removed custom Langfuse logic.
- `src/memory/conversation.py`, `src/cost/budget.py`: Redis connections unified.
- Overall reduced RAM usage and improved cloud readiness.
