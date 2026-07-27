## 1. Extract Hardcoded Routing Config

- [x] 1.1 Create `config/routing.yaml` (if it doesn't already exist or expand it) and move the `reference_queries` dictionary from `src/routing/features.py` into it.
- [x] 1.2 Update `FeatureExtractor.__init__` in `src/routing/features.py` to read `reference_queries` from `config/routing.yaml`.

## 2. Deduplicate Semantic Feature Math

- [x] 2.1 Update `extract()` in `src/routing/features.py` to call `self.batch_extract_vectors([query])[0]` rather than implementing its own max cosine similarity math.
- [x] 2.2 Delete the redundant local math calculation in `extract()`.
- [x] 2.3 Run tests to verify the classifier and features extraction still yield the same behavior.

## 3. Decouple BM25 from VectorStore

- [x] 3.1 Extract `save_bm25_index` and `load_bm25_documents` out of `src/retrieval/vector_store.py`.
- [x] 3.2 Move these methods to `src/retrieval/retriever.py` (e.g. inside `DocumentRetriever`).
- [x] 3.3 Update any usages of the local BM25 index save/load to call the new retriever methods.

## 4. Standardize OTEL Tracing Config

- [x] 4.1 In `src/utils/tracing.py`, remove the `is_langfuse = "langfuse" in endpoint.lower()` detection logic.
- [x] 4.2 Read `OTEL_EXPORTER_OTLP_PROTOCOL` from the environment.
- [x] 4.3 Fallback to `grpc` if no protocol is specified, but explicitly support `http/protobuf` to return the HTTP exporter.

## 5. Verification

- [x] 5.1 Run all tests via `pytest tests/` to ensure nothing is broken.
- [x] 5.2 Validate that BM25 local JSON fallback works correctly during queries.
