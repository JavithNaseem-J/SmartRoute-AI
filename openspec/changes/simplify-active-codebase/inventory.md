## Evidence

| Candidate | Caller check | Decision |
| --- | --- | --- |
| `frontend/src/lib/format.ts` | Citation fallback plus two labels used only by `App.tsx` | Delete file; keep two labels in app |
| Chat `sources` state | Only used by citation fallback | Remove; retain structured `citations` |
| Composer image/attachment state | `App.handleSendMessage` accepts only text | Remove; retain document upload callback |
| Dialog wrapper/dependency | Only used by image preview | Remove wrapper and package |
| Pipeline statistics/savings wrappers | No tracked callers | Remove; API uses tenant-aware tracker |
| `tqdm` direct dependency | No tracked direct import | Remove direct declaration |
| Render `DOCUMENTS_DIR`, `PYTHONUNBUFFERED` | No runtime read; Dockerfile sets latter | Remove redundant entries |
| External reranker and legacy provider key | User selected local reranking and one active `LLM_API_KEY`; runtime paths isolated | Remove branches, tests, and docs together; retain both LLM providers |

Baseline: 118 backend tests passed. `.env` was not edited. The pre-existing `.env.example` edit is preserved.

Final local checks: 117 backend tests, Ruff, mypy, frontend lint/typecheck/build, 8 passing browser checks (2 viewport skips), offline RAG evaluation, and OpenSpec validation. Runtime source is net -304 lines; all cleanup changes including this OpenSpec change are net -448 lines by Git's diff, excluding the pre-existing `.env.example` edit. Production readiness remains unverified.
