## Context
The application relies heavily on NVIDIA NIM for its models layer. When this endpoint faces outages or rate limiting, the entire pipeline fails (`503 Service Unavailable`), leading to poor UX. Furthermore, budget checking fallback logic has bugs, where incorrect keys (`groq_models`) and non-existent fallback models (`llama_3_1_8b`) are requested. Performance regressions exist across the async data processing layers.

## Goals / Non-Goals
**Goals:**
- Fallback gracefully when NVIDIA API calls fail, leveraging Groq (primary fallback) and Google Gemini (secondary fallback).
- Ensure configuration correctly references models that exist in `models.yaml`.
- Ensure cost tracking actually functions correctly across models.
- Optimize thread pool creation and database access for retrieval.

**Non-Goals:**
- Do not migrate entirely away from NVIDIA NIM as the default provider.
- Do not rewrite the routing/classification ML model.

## Decisions

- **Multi-provider BaseLLM implementation:** A new `GroqModel` and `GeminiModel` (inheriting from `BaseLLM`) will be created similar to `NvidiaModel`. `ModelManager` will load instances and use a fallback loop if the primary provider raises `RateLimitError` or `APIError`.
- **Budget Fixes:** Fix the model key validation in `budget.py:147` from `groq_models` to `nvidia_models` (and extend to all defined model providers). Change the fallback in `inference.py:105` to the `default_strategy` simple model (e.g., `nemotron_nano_8b`).
- **Performance:** Define a module-level `ThreadPoolExecutor(max_workers=4)` inside `retriever.py` to reuse the thread pool across queries instead of redefining it inside `_retrieve_hybrid`.
- **Async DB updates:** Use `asyncio.to_thread` or an executor in `inference.py` around `tracker.log_query()` to prevent the synchronous SQLAlchemy logic from blocking the event loop.

## Risks / Trade-offs
- **Risk:** Latency spikes during failover.
  - **Mitigation:** Fallback providers will be selected based on fastest response time (Groq).
- **Risk:** Gemini uses different token counting heuristics.
  - **Mitigation:** Use a generic length-based heuristic (`len(text)//4`) consistently across providers unless exact counting is strictly needed.
