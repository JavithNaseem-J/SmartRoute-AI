## 1. Indexer & Core Retrieval Deletion Logic

- [x] 1.1 Add `adelete_document(filename)` and `aclear_all_documents()` methods to `DocumentIndexer` in `src/retrieval/indexer.py` with Qdrant vector filter deletion and cache flushing.

## 2. API Endpoints

- [x] 2.1 Add `GET /v1/documents`, `DELETE /v1/documents/{filename}`, and `DELETE /v1/documents` endpoints to `api/main.py`.

## 3. Dashboard UI & Verification

- [x] 3.1 Update `app.py` Knowledge Base tab to show an interactive document manager with delete buttons.
- [x] 3.2 Add test cases in `tests/test_api.py` for document listing and deletion endpoints.
- [x] 3.3 Run pytest to verify all tests pass.
