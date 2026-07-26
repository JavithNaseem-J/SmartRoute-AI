## 1. High-Severity Fixes & Dead Code Cleanup

- [x] 1.1 Update `scripts/generate_training_data.py` to import `OpenRouterModel` instead of deleted `NvidiaModel`
- [x] 1.2 Delete dead test validation script `validate_bm25.py`
- [x] 1.3 Fix invalid job name reference `docker-build-pr` -> `docker-validate` in `.github/workflows/ci.yml`

## 2. Core Codebase Refactoring & De-duplication

- [x] 2.1 Refactor `InferencePipeline` in `src/pipeline/inference.py` to use a centralized `_log_query()` helper
- [x] 2.2 Create `src/routing/keywords.py` and refactor `HeuristicClassifier` and `FeatureExtractor` to share keyword sets
- [x] 2.3 Centralize Qdrant sparse model initialization inside `src/core/dependencies.py`
- [x] 2.4 Refactor `app.py` Streamlit document uploading UI into a reusable helper function

## 3. Performance & Clean Code Optimization

- [x] 3.1 Refactor `OpenRouterModel` API calls to streamline retries and circuit breaker checks
- [x] 3.2 Cache model pricing dictionary in `BudgetManager.__init__` in `src/cost/budget.py`
- [x] 3.3 Remove unused `__call__` decorator method from `AsyncCircuitBreaker` in `src/utils/circuit_breaker.py`
- [x] 3.4 Update `src/utils/tracing.py` to import `logger` directly from `src.utils.logger`
- [x] 3.5 Clean up docstrings and unused method notes in `src/models/base.py`

## 4. Verification & Testing

- [x] 4.1 Run full test suite with `pytest` to ensure zero regressions
