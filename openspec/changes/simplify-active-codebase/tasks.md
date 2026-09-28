## Deletion-first cleanup

- [x] Confirm callers and record baseline tests.
- [x] Remove unused inference wrappers, direct `tqdm`, and redundant Render settings.
- [x] Remove old source-string citation fallback and its unused state/helper file.
- [x] Remove unsent composer attachments, image preview, dialog wrapper, and dialog dependency; keep document upload.
- [x] Verify upload, citations, backend, frontend, offline RAG, and net line count.
- [x] Remove external Hugging Face reranker and old `OPENROUTER_API_KEY` fallback; keep both LLM providers on `LLM_API_KEY` and test the supported modes.
- [ ] Verify production readiness and migration separately; local services are unavailable.
