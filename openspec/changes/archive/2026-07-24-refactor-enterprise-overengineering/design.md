## Context

Currently, the Streamlit frontend imports backend components (`InferencePipeline`, `DocumentIndexer`), creating a monolithic setup that contends for database and Redis resources. Additionally, the retrieval and reranking systems load massive memory-heavy assets locally (`BM25Retriever` corpus stored in Redis and the PyTorch `cross-encoder` in CPU RAM), blocking horizontal auto-scaling and preventing deployment to lightweight cloud containers.

## Goals / Non-Goals

**Goals:**
- Transition the architecture to a pure microservices model (FastAPI backend + REST frontend).
- Replace local PyTorch CPU execution with async external inference API calls.
- Replace local `BM25Retriever` memory indexing with native Qdrant hybrid search capabilities.

**Non-Goals:**
- No changes to routing behavior or model assignments.
- No changes to user-facing capabilities in the Streamlit UI, other than its backend data sourcing.

## Decisions

1. **Decouple Streamlit via REST**: Streamlit will rely exclusively on the `/v1/query` endpoint for inference rather than directly initializing an `InferencePipeline`. This enforces separation of concerns.
2. **Qdrant Native Hybrid Search**: Rather than fetching a large BM25 JSON corpus from Redis to memory on startup, we will configure the Qdrant client to perform native hybrid search (Dense + Sparse). 
3. **HuggingFace Reranker API**: `DocumentReranker` will use an async `aiohttp` call to a HuggingFace Inference API for the cross-encoder reranking, completely eliminating `sentence-transformers` and `torch` from the `requirements.txt`.

## Risks / Trade-offs

- **Risk**: Streamlit relies on `requests.post` and may experience network latency rather than local execution speed.
  - *Mitigation*: The backend handles streaming and async correctly; we will utilize standard `httpx` or `requests` streaming in the UI to minimize perceived latency.
- **Risk**: Using Qdrant for BM25/Sparse requires Qdrant v1.7.0+ with sparse vectors enabled. 
  - *Mitigation*: The current Qdrant Cloud setup supports this by default.
