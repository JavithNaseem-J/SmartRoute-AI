## Why

The first cleanup split existing code into more files and increased total lines. This change instead removes proven dead UI paths, compatibility code, and unused dependencies. It must reduce net source lines without changing active document, chat, retrieval, or citation behavior.

## What Changes

- Remove frontend source-string citation fallback; use structured citations only.
- Remove composer image previews and unsent file chips. Document selection and drop still upload supported files to the knowledge base.
- Remove the now-unused formatting module, dialog wrapper, and dialog package.
- Remove unused inference forwarding methods, one unused direct Python dependency, and redundant Render settings.
- Retire the external Hugging Face reranker and the old OpenRouter key fallback. Keep GroqCloud and OpenRouter selectable through `LLM_PROVIDER` and `LLM_API_KEY`; preserve `.env` and the user's `.env.example` edits.

## Capabilities

### Modified Capabilities

- `pipeline-refactoring`: Retain inference and SSE contracts while removing unused wrappers.
- `structural-debt-resolution`: Require caller evidence and a net line reduction for this cleanup.

## Impact

Frontend composer, citations, dependency lockfile, inference helpers, configuration, and focused tests. No public API route or storage format changes.
