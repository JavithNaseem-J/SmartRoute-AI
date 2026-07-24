## Context

The audit identified structural debt, including inconsistent patterns (mixing local and cloud storage, mixing async and sync clients) and duplicate logic (instantiating models twice). These inconsistencies waste resources and break cloud compatibility. We need to refactor these areas to ensure the system is ready for distributed environments.

## Goals / Non-Goals

**Goals:**
- Unify Redis connections to use async clients.
- Unify Qdrant connections to use async clients.
- Centralize embedding model instantiation.
- Migrate BM25 to a non-local, persistent storage mechanism (or use Qdrant sparse vectors).
- Remove unused code.

**Non-Goals:**
- Major architecture changes.
- Replacing underlying database technologies (Redis, Qdrant).
- Changing business logic or routing thresholds.

## Decisions

- **Embedding Model Singleton:** We will create `src/models/embeddings.py` (or a similar location) to host a singleton factory for the SentenceTransformer model. `Embedder` and `FeatureExtractor` will both import this shared instance, saving RAM and initialization time.
- **BM25 Persistence:** Instead of saving BM25 corpora to `data/bm25_index.json`, we will store the corpus payload in Redis or leverage Qdrant sparse vectors. Given the existing Qdrant/Redis dependencies, we'll store the serialized documents in Redis for cloud-wide availability.
- **Qdrant Async Unification:** We will refactor `VectorStore` to use `AsyncQdrantClient` instead of the synchronous wrapper, aligning it with `SemanticCache` and the async `InferencePipeline`.
- **Tracing Simplification:** We will drop the manual Langfuse basic auth encoding in `src/utils/tracing.py`. Users can pass `OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic <base64>"` via environment variables natively.

## Risks / Trade-offs

- **Risk:** Modifying Qdrant clients from sync to async requires refactoring how `DocumentRetriever` invokes it.
  - **Mitigation:** We will update `_retrieve_dense` and `_retrieve_hybrid` in `retriever.py` to be `async` and remove the `loop.run_in_executor` wrap in `inference.py`.
- **Risk:** Serializing the BM25 corpus to Redis might be slow if the corpus is huge.
  - **Mitigation:** The chunk size is relatively small and can be compressed. Redis string caching is fast.
