## Why

The current SmartRoute-AI pipeline is completely hardcoded to rely on a single provider (NVIDIA NIM). When NVIDIA NIM is overloaded, the entire application fails because there is no fallback provider mechanism. Furthermore, the budget tracker contains hardcoded errors (e.g. referencing non-existent fallback models and incorrect config keys) that break cost estimations and budget fallback routes, causing silent failures or crashes. The code also contains performance bottlenecks like recreating thread pools per query and doing synchronous database writes in an async loop.

## What Changes

- Add multi-provider support by integrating free, high-performance fallback options (e.g., Groq, Google Gemini) when the primary provider (NVIDIA) is overloaded.
- Fix the budget fallback model name in `inference.py` to reference a model that actually exists in `models.yaml`.
- Fix the cost estimation bug in `budget.py` that incorrectly looks for the `"groq_models"` key instead of `"nvidia_models"`.
- Clean up performance issues: instantiate `ThreadPoolExecutor` once instead of per-query, remove redundant `validate_query` calls, and avoid blocking async code with synchronous SQLAlchemy commits.
- Fix deprecated `asyncio.get_event_loop()` usage.

## Capabilities

### New Capabilities
- `multi-provider-fallback`: The system will gracefully failover to secondary LLM providers (e.g., Groq, Gemini) when the primary provider experiences rate limits, timeouts, or service overloads.
- `budget-fixes`: Budget enforcement logic needs updating to accurately query model config keys and route to existing fallback models when budgets are exhausted.
- `pipeline-performance`: Retrieval thread pooling and database event logging performance will be optimized for better concurrency and responsiveness.

### Modified Capabilities

## Impact

- **Models layer**: Modifies `BaseLLM` and adds new providers, updates `ModelManager` to handle fallback logic.
- **Config**: Modifies `models.yaml` and `.env` to include new provider configurations.
- **Pipeline & Cost layers**: Modifies `inference.py`, `retriever.py`, `budget.py`, and `tracker.py` for bug fixes and performance improvements.
