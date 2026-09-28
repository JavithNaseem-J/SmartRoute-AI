## Decisions

1. Delete only code with confirmed callers and behavior. Frontend `sources` was used only to synthesize old citations; current answers provide structured citations. Removing this fallback prevents old source labels from appearing as evidence.
2. The composer previously accepted image and document files as chat attachments, but `onSend` passed only text to the API. Keep document upload through `onUploadDocument`; remove attachment state, image preview, dialog wrappers, and their package.
3. Keep both active LLM providers, RAG, tenant, budget, citation, and document-delete paths. The user selected local Qdrant/FastEmbed retrieval and one LLM key, so remove the external reranker and old key fallback. Live deployment readiness remains unverified.
4. Count tracked insertions/deletions and new OpenSpec lines. Run backend, frontend, and browser checks. Deployment readiness remains a separate check.

## Risks

- Older saved chats with only `sources` no longer show generated citation buttons. Their answer text remains, and new answers carry structured citations.
- Upload picker and drag/drop must still invoke the document endpoint; browser tests cover both the upload and citation behavior.
- A live service still configured with only the old key or external reranker mode will fail startup; set `LLM_API_KEY` and `RERANKER_MODE=local` before deployment.
