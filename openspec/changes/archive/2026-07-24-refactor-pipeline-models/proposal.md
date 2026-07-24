## Why

The orchestration pipeline (`src/pipeline/inference.py`) and model abstractions (`src/models/base.py`) contain significant structural debt: duplicate setup logic and abstraction leaks. Resolving these issues will make the core inference flow more resilient to change and easier to test. It will also make adding new LLM providers trivial without duplicating prompt construction logic across multiple client classes.

## What Changes

- Abstract common pipeline setup steps (semantic caching, routing, budget checks, vector retrieval) into a unified setup method in `InferencePipeline` that yields a context object, rather than duplicating it across `run` and `astream_run`.
- Simplify the `BaseLLM` interface to accept standard standard message arrays (`[{"role": "user", ...}]`) instead of disparate prompt/context/history string parameters.
- Move prompt template construction (system prompts, user message formatting) out of specific model clients (like `OpenRouterModel`) and into the `InferencePipeline` layer.

## Capabilities

### New Capabilities
- `pipeline-refactoring`: A new internal spec tracking the architectural separation of concerns for the inference pipeline and LLM abstraction.

### Modified Capabilities
None.

## Impact

- `src/pipeline/inference.py` (significant refactoring of `run` and `astream_run`)
- `src/models/base.py` (interface definition change)
- `src/models/openrouter_model.py` (removing prompt construction logic)
- `src/models/model_manager.py` (minor tweaks if initialization is affected)
