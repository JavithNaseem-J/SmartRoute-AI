## Why

Users currently have to manually navigate the codebase and delete files from the `data/documents` directory to remove indexed documents. Furthermore, deleting a local file manually leaves orphaned vector embeddings in Qdrant and stale entries in Redis semantic cache. To make SmartRoute-AI production-ready for enterprise deployment, users need full document lifecycle management (listing, deleting single files, clearing all documents) directly from the API and Streamlit UI with complete vector & cache synchronization.

## What Changes

- **Document Management API**: Add `GET /v1/documents`, `DELETE /v1/documents/{filename}`, and `DELETE /v1/documents` endpoints in `api/main.py`.
- **Vector & Cache Sync on Deletion**: Implement document vector deletion in Qdrant matching payload `metadata.source` and invalidate Redis semantic cache upon document deletion.
- **Streamlit UI Document Management**: Add a interactive document list section in `app.py` with individual **🗑️ Delete** buttons and a **Clear Knowledge Base** action.
- **Production Hardening**: Ensure proper error handling, logging, and graceful cache invalidation when document state changes.

## Capabilities

### New Capabilities
- `document-deletion-management`: Complete document lifecycle management (list, delete document file, delete document vectors from Qdrant, invalidate cache) via API and UI.

### Modified Capabilities
<!-- None -->

## Impact

- `api/main.py`: Added document listing and deletion API endpoints.
- `src/retrieval/indexer.py`: Added `adelete_document` and `aclear_all_documents` methods that delete local files and Purge Qdrant points.
- `app.py`: Added document management UI in the Knowledge Base tab.
- `tests/test_api.py`: Added test cases for document listing and deletion endpoints.
