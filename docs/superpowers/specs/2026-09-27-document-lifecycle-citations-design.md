# SmartRoute Document Lifecycle and Citation Design

Date: 2026-09-27

## Goal

Make document answers trustworthy after uploads and deletions, and replace the large source list in each assistant message with compact, claim-level citations that identify the supporting document and page or section.

## Problems Confirmed

The database-backed sidebar and Qdrant can disagree. `adelete_document` currently catches Qdrant errors and returns `False`, while the API still marks the database record deleted. The sidebar then hides the document even though its vectors remain retrievable. Retrieval filters by user only, so any orphan vectors for that user can still ground future answers.

The current `sources` response is a list of display strings. It describes retrieved chunks, but it does not map an answer claim to evidence, is difficult to validate, and forces the frontend to parse presentation text. The frontend renders those strings as a large source card inside the assistant bubble.

## Chosen Approach

Use active-document allowlisting for retrieval and verified vector deletion for lifecycle correctness. Add structured citations while temporarily preserving the existing `sources` field for backward compatibility with cached responses and saved browser sessions. Render inline citation markers plus a compact citation control instead of the current source card.

This approach is preferred over source-only badges because it maps evidence to claims. It is preferred over a permanent citations sidebar because SmartRoute is a compact chat application and does not need a research-workspace layout.

## Document Lifecycle

### Retrieval

Before document retrieval, load the user's active document records from PostgreSQL and collect their durable `storage_path` values. Qdrant search must require both:

- `metadata.user_id` equals the authenticated user.
- `metadata.source` belongs to the active storage-path allowlist.

If the user has no active documents, return no document context without querying Qdrant. This immediately prevents already orphaned vectors from appearing in answers, even before they are physically removed.

### Single-document deletion

Deletion runs in this order:

1. Resolve the active database record for the authenticated user.
2. Delete matching Qdrant points with `wait=True`.
3. Count matching points and require the result to be zero.
4. Invalidate the user's semantic answer cache.
5. Delete the stored object.
6. Mark the database record deleted.
7. Reload retriever readiness.

Qdrant failures must propagate to the API. The API must not report success or hide the database record when vector deletion failed.

### Clear-all deletion

Clear-all must delete and verify all Qdrant points for the authenticated user before marking database records deleted. Cache invalidation remains user-scoped. Failures return an error instead of a false success response.

### Existing orphan vectors

Active-source filtering makes existing orphan vectors non-retrievable immediately. A reconciliation operation may physically remove old orphans, but it is not required on the query hot path and must never delete another user's vectors.

## Citation Data Contract

Each retrieved chunk receives a stable marker such as `C1` and carries structured metadata:

```json
{
  "id": "C1",
  "filename": "Resume.pdf",
  "page": 3,
  "section": null,
  "excerpt": "Short supporting text from the retrieved chunk"
}
```

Rules:

- PDF pages are one-based in API and UI responses.
- TXT and Markdown citations use a section when available and otherwise omit page information.
- Excerpts are bounded supporting snippets, not entire chunks.
- Internal storage paths and credentials are not exposed to the browser.
- Citations are deduplicated by document, page or section, and supporting excerpt.

The API adds `citations` to normal and streaming query responses. The legacy `sources` list remains during migration so old semantic-cache entries and browser sessions still load.

## Grounded Generation

Retrieved context is labeled with citation IDs before it is sent to the model. The system instruction requires the model to place a citation marker after each factual claim supported by uploaded content, for example:

```text
The cover letter targets a backend engineering role. [C1]
```

After generation, the backend removes citation markers that do not correspond to retrieved evidence. The UI receives only validated citation objects. A citation means the referenced chunk was supplied as evidence; it is not presented as independent proof that the model's interpretation is correct.

## Frontend Experience

Assistant answers keep small inline markers such as `[1]`. Beneath the answer, a compact citation button row shows one icon per cited source. The current large `Sources` card is removed.

Selecting a marker or citation icon opens a small anchored detail panel with:

- Filename.
- Page number or section, when available.
- A short supporting excerpt.

The panel closes when the citation is selected again or when another citation is selected. Controls have accessible labels, keyboard focus states, and stable dimensions. On narrow screens the detail panel stays inside the message width.

Old saved messages that contain only string `sources` are converted to basic filename/page citations in the browser, without excerpts, so chat history remains usable.

## Components and Boundaries

- `src/retrieval/retriever.py`: active-source Qdrant filtering and structured citation construction.
- `src/retrieval/indexer.py`: strict, verified vector deletion.
- `src/pipeline/inference.py`: active-document lookup, citation-aware context, model instructions, and citation validation.
- `api/schemas.py`: structured citation response schema.
- `api/main.py`: deletion error behavior and lifecycle ordering.
- `frontend/src/types/chat.ts`: citation type stored with assistant messages.
- A focused frontend citation component: answer marker rendering, compact icons, and the detail panel.
- Existing tests: tenant filtering, stale-source exclusion, deletion verification, response contracts, streaming metadata, and legacy-source fallback.

## Error Handling

- Database lookup failure: fail the retrieval request safely instead of searching all user vectors.
- Qdrant deletion or verification failure: return an explicit document-deletion error and keep the database record active.
- Citation metadata missing: render the filename only; never invent a page.
- Invalid model citation marker: remove it from the answer and do not create a citation object for it.
- No validated citations: show no citation controls.

## Verification

- Backend unit tests for active-source filters and empty active-document behavior.
- Backend tests proving Qdrant deletion errors propagate and database records are not marked deleted.
- Backend tests for one-based PDF pages, citation validation, and streaming citation payloads.
- Frontend typecheck, lint, and production build.
- Playwright checks at desktop and mobile widths for inline markers, citation controls, panel positioning, text containment, and removal of the old source card.
- Full Python test suite, Ruff, and mypy.

## Non-goals

- Building a full document viewer in this change.
- Generating artificial page numbers for TXT or Markdown files.
- Adding a permanent research sidebar.
- Replacing Qdrant, FastEmbed, or the existing hybrid retrieval strategy.
