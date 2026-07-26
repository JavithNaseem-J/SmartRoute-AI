## Why

A comprehensive structural audit revealed 12 instances of code quality debt, dead code, duplicated logic, and broken CI configuration. Addressing these issues will eliminate latent runtime bugs, remove obsolete scripts, reduce code duplication, and improve long-term maintainability.

## What Changes

- **Fix Broken Scripts & CI**:
  - Update `scripts/generate_training_data.py` to import `OpenRouterModel` instead of deleted `NvidiaModel`.
  - Fix invalid `docker-build-pr` job reference in `.github/workflows/ci.yml` (replace with `docker-validate`).
  - Delete obsolete `validate_bm25.py` script.
- **Consolidate Duplicated Logic**:
  - Unify cost/latency tracking in `src/pipeline/inference.py` into a single `_log_query()` helper.
  - Consolidate classification keywords shared by `HeuristicClassifier` and `FeatureExtractor` into a central module.
  - Centralize Qdrant sparse model initialization in `src/core/dependencies.py`.
  - Extract reusable document upload UI component in `app.py`.
  - Streamline retry/circuit-breaker logic in `src/models/openrouter_model.py`.
- **Remove Over-engineering & Standardize Patterns**:
  - Cache pricing configuration in `BudgetManager.__init__` instead of reading `models.yaml` from disk per estimation.
  - Clean up unused `__call__` method in `AsyncCircuitBreaker`.
  - Standardize logger import in `src/utils/tracing.py`.
  - Fix docstring mismatch in `src/models/base.py`.

## Capabilities

### New Capabilities

*(None)*

### Modified Capabilities

- `ci-cd`: Correct invalid job reference in GitHub Actions workflow dependencies.
- `pipeline-refactoring`: Refactor internal pipeline logging and dependency initialization for cleaner architecture.

## Impact

- **Codebase**: Removes dead code (`validate_bm25.py`), fixes imports in scripts, and eliminates ~100+ redundant lines of duplicate code.
- **CI/CD**: Fixes job dependency resolution in `.github/workflows/ci.yml`.
- **Runtime API**: Zero breaking changes to public FastAPI or Streamlit user interfaces.
