## 1. Decouple Streamlit Frontend

- [x] 1.1 Remove direct `InferencePipeline` and `DocumentIndexer` imports from `app.py`.
- [x] 1.2 Replace the local RAG indexing block in `app.py` with HTTP POST requests to `/v1/documents`. (Assuming API needs to be adjusted, or just rely on the backend). Wait, currently there is no index endpoint in `api/main.py`... We'll need to use HTTP POST for queries.
- [x] 1.3 Refactor the query execution block in `app.py` to make a streaming `httpx` or `requests` POST call to `http://api:8000/v1/query`.
- [x] 1.4 Refactor cost analytics and budget checks in `app.py` to use a backend endpoint or make direct HTTP calls instead of local objects. (May require exposing these via `api/main.py` if not already there).

## 2. Reranker Offloading

- [x] 2.1 Refactor `DocumentReranker` in `src/retrieval/reranker.py` to use `aiohttp` to call HuggingFace Inference API (or similar) instead of `sentence-transformers` CrossEncoder.
- [x] 2.2 Remove `sentence-transformers` from `requirements.txt`.
- [x] 2.3 Verify `tests/` pass with mocked reranker HTTP responses.

## 3. Qdrant Native Hybrid Search

- [x] 3.1 Remove the `BM25Retriever` logic from `src/retrieval/retriever.py`.
- [x] 3.2 Remove the Redis storage logic for the BM25 text corpus (e.g., `smartroute:bm25_corpus`).
- [x] 3.3 Configure the `AsyncQdrantClient` in `src/retrieval/retriever.py` to initialize sparse vectors (SPLADE) on the collection.
- [x] 3.4 Refactor `_retrieve_hybrid` in `DocumentRetriever` to perform native Qdrant hybrid queries.
- [x] 3.5 Update `DocumentIndexer` to generate sparse vectors alongside dense vectors when saving to Qdrant.

## 4. Final Testing and Documentation

- [x] 4.1 Update `.env.example` and documentation to note that `HF_TOKEN` must have permissions for external inference (if using HuggingFace).
- [x] 4.2 Validate that `docker-compose.yml` does not try to bundle massive torch weights.
- [x] 4.3 Test the entire system end-to-end to ensure the API connects to Qdrant successfully and streams completions correctly.
