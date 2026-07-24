## 1. LLM Interface Refactoring

- [x] 1.1 Update `BaseLLM` interface in `src/models/base.py` so `agenerate` and `astream` accept `messages: List[Dict]` instead of `prompt`, `context`, and `history`.
- [x] 1.2 Extract `_build_messages` from `src/models/openrouter_model.py` and move it into `src/pipeline/inference.py`.
- [x] 1.3 Update `OpenRouterModel.agenerate` and `astream` to accept `messages` and pass them directly to the API without internal modification.

## 2. Pipeline Refactoring

- [x] 2.1 Extract common setup steps (guardrails, semantic cache, route, budget, retrieve) in `src/pipeline/inference.py` into a new `_prepare_context` async method.
- [x] 2.2 Refactor `InferencePipeline.run` to call `_prepare_context`, assemble the standard message list, and execute `agenerate`.
- [x] 2.3 Refactor `InferencePipeline.astream_run` to call `_prepare_context`, assemble the standard message list, and execute `astream`.

## 3. Testing and Verification

- [x] 3.1 Update all tests in `tests/` that mock `BaseLLM` methods to use the new `messages` signature.
- [x] 3.2 Run `pytest tests/` to verify that the refactored pipeline routes and executes queries successfully.
