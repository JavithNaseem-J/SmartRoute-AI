## Context

Following a repository audit, 12 instances of structural code debt, duplicate logic, dead code, and broken script/CI configuration were identified. These issues range from broken script imports and GitHub Actions job dependency mismatches to redundant query logging in `InferencePipeline` and duplicate keyword lists across classifiers.

## Goals / Non-Goals

**Goals:**
- Fix high-severity broken scripts (`scripts/generate_training_data.py`, `validate_bm25.py`) and CI workflow dependencies (`.github/workflows/ci.yml`).
- Eliminate duplicate logic in pipeline query logging, routing keyword definitions, Qdrant sparse model initialization, and Streamlit UI components.
- Reduce unnecessary disk I/O in cost estimation and clean up unused abstractions.

**Non-Goals:**
- Modifying API endpoint contracts or external response payloads.
- Altering core routing algorithms or ML classifier behavior.

## Decisions

1. **Delete `validate_bm25.py` & Fix `scripts/generate_training_data.py`**:
   - *Rationale*: `validate_bm25.py` targets removed legacy BM25 classes. `generate_training_data.py` points to a deleted `src.models.nvidia_model` module. Updating `generate_training_data.py` to `OpenRouterModel` restores functionality.
2. **Centralize Shared Keywords in `src/routing/keywords.py`**:
   - *Rationale*: `HeuristicClassifier` and `FeatureExtractor` both define complex, medium, technical, and reasoning keywords. Placing them in a shared module ensures consistent classification logic across ML and heuristic modes.
3. **Unify `InferencePipeline` Query Logging**:
   - *Rationale*: `_prepare_context`, `run`, and `astream_run` repeat 10+ lines of `asyncio.to_thread(self.tracker.log_query, ...)` boilerplate. A single `_log_query_metrics()` helper keeps metrics logic DRY.
4. **Cache Pricing Data in `BudgetManager`**:
   - *Rationale*: Reading `models.yaml` from disk on every query estimation is inefficient. Reading it once at initialization eliminates redundant file reads.

## Risks / Trade-offs

- **[Risk]**: Refactoring `InferencePipeline` logging could miss edge-case attributes in streaming error paths.
  - *Mitigation*: Ensure `_log_query_metrics()` handles optional parameters cleanly and verify with unit tests.
- **[Risk]**: Centralizing Qdrant sparse model configuration might affect indexing or retrieval initialization order.
  - *Mitigation*: Initialize sparse model setup once in `src/core/dependencies.py` client getters.
