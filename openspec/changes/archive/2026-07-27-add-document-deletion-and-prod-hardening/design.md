## Context

Document management in SmartRoute-AI currently supports uploading and indexing (`/v1/index`), but lacks deletion primitives. If a user uploads an outdated or incorrect document, removing it requires local file deletion and leaves orphaned vectors in Qdrant and stale cached answers in Redis.

## Goals / Non-Goals

**Goals:**
- Provide native `DELETE /v1/documents/{filename}` and `DELETE /v1/documents` endpoints.
- Synchronously purge corresponding Qdrant vector points filtered by `metadata.source` or payload filename.
- Automatically invalidate Redis semantic cache upon document deletion.
- Add an interactive document manager in the Streamlit dashboard (`app.py`).

**Non-Goals:**
- Multi-tenant user permission scoping for individual document ownership (handled at API key level).

## Decisions

### Decision 1: Qdrant Point Filtering for Document Deletion
- Use `AsyncQdrantClient.delete(collection_name, points_selector=models.Filter(...))` matching `metadata.source` containing the target filename.

### Decision 2: Automatic Semantic Cache Flushing
- Call `redis.flushdb()` or clear semantic cache keys whenever documents are deleted to prevent stale cached Q&A responses from being served.

### Decision 3: Streamlit Interactive Document Manager
- Render an expander or table in the Knowledge Base tab showing all indexed documents with file size and an inline **🗑️ Delete** button.

## Risks / Trade-offs

- [Risk] Deleting vectors from a large Qdrant collection could take time → Mitigation: Operations are executed asynchronously with proper FastAPI exception handling.
