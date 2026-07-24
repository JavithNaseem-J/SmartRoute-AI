## Why

The SmartRoute-AI codebase contains critical architectural flaws (overengineering) that currently prevent the system from being enterprise production-ready. Specifically, the Streamlit frontend operates as a monolithic backend, the local hybrid RAG solution relies on massive in-memory and Redis text caching (causing OOMs at scale), and the embedded PyTorch reranker consumes substantial local CPU and RAM. Addressing these will enable true microservice deployment, horizontal scaling, and reduced resource contention.

## What Changes

- **Streamlit Decoupling**: Remove direct imports of `InferencePipeline` and `DocumentIndexer` from `app.py`. Rewrite the UI to interact exclusively with the FastAPI backend via REST endpoints (`/v1/query`).
- **Hybrid Search Modernization**: Delete the custom `BM25Retriever` implementation and Redis text-blob corpus caching in `src/retrieval/retriever.py`. Delegate RRF (Reciprocal Rank Fusion) and hybrid sparse/dense search natively to Qdrant.
- **Reranker Offloading**: Remove the local PyTorch `cross-encoder/ms-marco-MiniLM-L-6-v2` execution from `src/retrieval/reranker.py`. Refactor it to call an external inference API (e.g., HuggingFace or Cohere). Remove `sentence-transformers` from requirements.

## Capabilities

### New Capabilities
- None. This is an architectural hardening and debt-reduction change.

### Modified Capabilities
- None. System behavior and functional requirements remain unchanged; only internal structural debt and overengineering are being fixed.

## Impact

- `app.py`: Will become a true lightweight frontend UI.
- `src/retrieval/retriever.py`: Massive reduction in memory footprint and complexity.
- `src/retrieval/reranker.py`: Eliminated PyTorch dependency; replaced with async HTTP client.
- `requirements.txt`: Removed `sentence-transformers` (and transitively PyTorch/Torchvision), heavily reducing Docker image size.
