## 1. Safe hygiene and dead frontend files

- [x] 1.1 Add `graphify-out/` to `.gitignore` and keep Graphify analysis output untracked.
- [x] 1.2 Fix `.gitignore` so `uv.lock` is not listed as an ignored file while Docker/CI depend on it.
- [x] 1.3 Align `.vscode/settings.json` with pytest or remove misleading unittest-only settings.
- [x] 1.4 Remove unused frontend file `frontend/src/components/ui/demo.tsx` after confirming no imports.
- [x] 1.5 Remove unused frontend file `frontend/src/types/chat.ts` or migrate active App message/session types into it.
- [x] 1.6 Remove unused frontend file `frontend/src/lib/utils.ts` or adopt its helpers consistently.
- [x] 1.7 Update `.env.example` to remove unused budget env vars or explicitly wire/document their source.
- [x] 1.8 Document optional runtime knobs read by code: `MAX_DOCUMENT_UPLOAD_BYTES`, `WEBHOOK_URL`, `DEMO_TOKEN_TTL_SECONDS`, and OTEL headers/protocol.
- [x] 1.9 Run backend lint/type/tests and frontend lint/type/build for the hygiene phase.

## 2. Legacy UI and deployment surface removal

- [x] 2.1 Verify React/FastAPI parity for chat, streaming, RAG upload, document list/delete/clear, cost analytics, and budget display.
- [x] 2.2 Verify Render service and `render.yaml` use the root `Dockerfile`, not `Dockerfile.api`.
- [x] 2.3 Remove `app.py` after parity verification.
- [x] 2.4 Remove `Dockerfile.api` after deployment verification.
- [x] 2.5 Update README and docs to remove legacy Streamlit and compatibility Docker instructions.
- [ ] 2.6 Run CI-equivalent checks and Docker smoke build after legacy removal.
  - CI-equivalent local checks passed; Docker smoke build is blocked locally because Docker Desktop is not running (`dockerDesktopLinuxEngine` pipe missing).

## 3. Dependency source and runtime slimming

- [x] 3.1 Remove `streamlit`, `plotly`, and `pandas` from production dependencies after `app.py` removal.
- [x] 3.2 Remove unused direct dependencies `tiktoken` and `pydantic-settings` if CI confirms no usage.
- [x] 3.3 Remove `langchain-qdrant` if no runtime import or framework integration needs it.
- [x] 3.4 Investigate whether the `langchain` meta package is still required beyond installed subpackages.
- [x] 3.5 Remove or regenerate `requirements.txt` and `requirements-dev.txt` from uv instead of maintaining independent manifests.
- [ ] 3.6 Regenerate `uv.lock` and run backend and Docker checks.
  - `uv.lock`, backend lint/type/tests, and frontend lint/type/build passed; Docker smoke build is blocked locally because Docker Desktop is not running.

## 4. Classifier and centroid build correctness

- [x] 4.1 Decide whether `reference_centroids.npy` is committed, generated during build, or removed as a feature.
- [x] 4.2 Implement the chosen centroid artifact ownership path.
- [x] 4.3 Make classifier training deterministic for Docker builds or stop retraining during production image builds.
- [ ] 4.4 Verify router tests and production Docker build use the intended classifier and centroid artifacts.
  - Router tests passed; Docker smoke build is blocked locally because Docker Desktop is not running.

## 5. Local ingestion compatibility cleanup

- [x] 5.1 Decide whether `/v1/index` remains as a supported manual/admin endpoint or is removed.
- [x] 5.2 If removed, delete local `DOCUMENTS_DIR` indexing path and update docs/env examples.
- [x] 5.3 Remove unused synchronous `DocumentIndexer` wrappers only after local ingestion decisions are complete.
- [x] 5.4 Verify document upload, delete, clear-all, and RAG retrieval tests still pass.

## 6. Backend and frontend boundary refactors

- [ ] 6.1 Split `api/main.py` into focused app setup, system/auth, query, analytics, documents, and frontend-static modules.
  - Started safely by moving API schemas/typed upload records into `api/schemas.py` while preserving existing route behavior and test monkeypatch compatibility.
  - Moved system health/version/readiness routes into `api/system_routes.py`; query, analytics, document, and frontend-static route extraction remains a follow-up because those routes share mutable `pipeline` test fixtures.
- [ ] 6.2 Preserve existing route paths and response contracts while moving code.
  - Verified preserved paths for the extracted system routes with API tests.
- [ ] 6.3 Split `frontend/src/App.tsx` into session, document, streaming query, sidebar, chat, and analytics units.
  - Started safely by moving chat/session types and session persistence helpers out of `App.tsx`.
- [x] 6.4 Split `frontend/src/components/ui/ai-prompt-box.tsx` into reusable UI primitives and SmartRoute-specific prompt controls.
- [ ] 6.5 Run full backend/frontend validation after each boundary refactor.
  - Validation after partial Phase 6 split passed: backend lint/type/tests and frontend lint/type/build.

## 7. Documentation and archive polish

- [x] 7.1 Update README architecture tree after cleanup phases complete.
- [x] 7.2 Decide whether OpenSpec archived changes remain in-repo or are summarized for portfolio readability.
  - Decision: keep this OpenSpec change active in-repo until Docker smoke verification and the remaining deeper route/component split are completed.
- [ ] 7.3 Update current OpenSpec specs through archive/sync after implementation completes.
  - Not archived yet because Docker smoke verification is blocked locally and deeper route/component extraction remains intentionally deferred.
