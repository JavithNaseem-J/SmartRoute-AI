## 1. Multi-Provider LLM Integration

- [x] 1.1 Create `src/models/groq_model.py` containing `GroqModel` (implements `BaseLLM`)
- [x] 1.2 Create `src/models/gemini_model.py` containing `GeminiModel` (implements `BaseLLM`)
- [x] 1.3 Update `src/models/model_manager.py` to load new models and support provider fallback logic
- [x] 1.4 Update `config/models.yaml` to include configurations for `groq_models` and `gemini_models`

## 2. Budget and Cost Tracking Fixes

- [x] 2.1 Fix `budget.py:147` in `estimate_query_cost` to correctly loop over provider keys (`nvidia_models`, `groq_models`, etc.) rather than only checking `groq_models` for an NVIDIA ID
- [x] 2.2 Update `inference.py:105` to route to a valid fallback model (e.g. `nemotron_nano_8b`) instead of the non-existent `llama_3_1_8b`

## 3. Pipeline Performance Optimizations

- [x] 3.1 Refactor `src/retrieval/retriever.py` to use a persistent module-level `ThreadPoolExecutor` instead of creating one per hybrid retrieval query
- [x] 3.2 Update `src/pipeline/inference.py` to run the synchronous SQLAlchemy commit in `CostTracker.log_query()` via `asyncio.to_thread()` or an executor, avoiding event loop blocking
- [x] 3.3 Replace deprecated `asyncio.get_event_loop()` with `asyncio.get_running_loop()` in `inference.py`
- [x] 3.4 Ensure `validate_query()` is only called once per query (remove redundant checks)
