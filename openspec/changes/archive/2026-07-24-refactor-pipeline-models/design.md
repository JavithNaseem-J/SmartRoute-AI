## Context

The `InferencePipeline` in `src/pipeline/inference.py` currently duplicates approximately 50 lines of setup logic across its standard (`run`) and streaming (`astream_run`) execution paths. At the same time, the `BaseLLM` interface in `src/models/base.py` takes raw string representations of `prompt`, `context`, and `history`. This requires each underlying LLM wrapper (like `OpenRouterModel`) to independently implement prompt formatting (e.g., `_build_messages`), violating DRY and making it harder to add new providers.

## Goals / Non-Goals

**Goals:**
- Extract the standard inference setup steps (guardrails, cache, route, budget, retrieval) into a shared internal `_prepare_context` method.
- Refactor the `BaseLLM` interface so `agenerate` and `astream` accept a pre-formatted list of `dict` messages (e.g. `[{"role": "user", "content": "..."}]`) instead of separate strings.
- Move the `_build_messages` function out of `OpenRouterModel` and into the Pipeline layer.

**Non-Goals:**
- No changes to the underlying routing classification math, budget algorithms, or retrieval accuracy.
- We are not adding new model providers in this change.

## Decisions

- **Pipeline Context extraction**: We will extract the common setup code into `async def _prepare_context(...)` within `InferencePipeline`. It will handle the entire lifecycle up until the actual model call, returning the `model_id`, `routing_decision`, `context`, and `sources` so both `run` and `astream_run` can simply execute the model call and track costs.
- **Prompt Standardization**: The logic inside `_build_messages` (currently in `OpenRouterModel`) will be extracted into the pipeline. The pipeline will assemble the final list of dictionaries, meaning `BaseLLM` implementations simply pass the `messages` object straight to their respective APIs.

## Risks / Trade-offs

- **Risk**: Modifying the signature of `BaseLLM.agenerate()` and `astream()` will break tests or mocks that expect the old signature. 
  - **Mitigation**: We will execute a strict update across the `tests/` directory to fix mock signatures.
- **Trade-off**: The `InferencePipeline` becomes responsible for prompt structuring.
  - **Mitigation**: This is an acceptable separation of concerns, as it forces the system to conform to a standard schema before it reaches the provider adapters.
