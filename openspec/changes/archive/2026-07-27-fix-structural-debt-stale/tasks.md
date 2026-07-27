## 1. Dead Code Removal

- [x] 1.1 Remove `check_budget_full` and `should_alert` from `src/cost/budget.py`

## 2. Tracing Cleanup

- [x] 2.1 Remove Langfuse specific header parsing logic from `src/utils/tracing.py`

## 3. Centralize Redis Connections

- [x] 3.1 Create a shared Redis client utility (e.g. `src/utils/redis.py`)
- [x] 3.2 Update `src/memory/conversation.py` to use the shared async Redis client
- [x] 3.3 Update `src/cost/budget.py` to use the shared async Redis client
- [x] 3.4 Update `src/retrieval/semantic_cache.py` to use the shared async Redis client

## 4. Singleton Embedding Model

- [x] 4.1 Create `src/models/embeddings.py` exposing a singleton `SentenceTransformer`
- [x] 4.2 Update `src/routing/features.py` to use the singleton
- [x] 4.3 Update `src/retrieval/embedder.py` to use the singleton

## 5. Unify Qdrant Async Clients

- [x] 5.1 Update `src/retrieval/vector_store.py` to use `AsyncQdrantClient` instead of the synchronous Langchain wrapper
- [x] 5.2 Update `src/retrieval/retriever.py` to support fully async dense retrieval without using `loop.run_in_executor`

## 6. BM25 Migration

- [x] 6.1 Refactor BM25 indexing in `src/retrieval/retriever.py` or `src/retrieval/indexer.py` to persist payloads in Redis instead of a local JSON file
