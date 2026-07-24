import os
from typing import AsyncIterator, Dict, List, Optional

from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.models.base import BaseLLM
from src.utils.circuit_breaker import AsyncCircuitBreaker
from src.utils.logger import logger


class OpenRouterModel(BaseLLM):
    """Async OpenRouter LLM — wraps AsyncOpenAI with retry logic and cost tracking."""

    def __init__(
        self,
        model_id: str,
        cost_per_1k_input: float = 0.0,
        cost_per_1k_output: float = 0.0,
        max_tokens: int = 4096,
        temperature: float = 0.5,
        max_retries: int = 3,
    ):
        self.model_id = model_id
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.warning(
                "OPENROUTER_API_KEY not found. Please provide an API key from https://openrouter.ai/"
            )

        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or "DUMMY_KEY_FOR_TESTS",
            timeout=10.0,
        )
        # Per-model circuit breaker — isolates failures to this model only
        self._breaker = AsyncCircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
        logger.info(f"OpenRouterModel initialized (async): {model_id}")

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((RateLimitError, APIError, APIConnectionError)),
        reraise=True,
    )
    async def _call_api(self, messages, max_tok, temp):
        self._breaker._update_state()
        if self._breaker.state == "OPEN":
            from src.utils.circuit_breaker import CircuitBreakerOpenException

            raise CircuitBreakerOpenException(f"Circuit breaker OPEN for {self.model_id}")
        try:
            result = await self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_tok,
                temperature=temp,
            )
            self._breaker.record_success()
            return result
        except Exception:
            self._breaker.record_failure()
            raise

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((RateLimitError, APIError, APIConnectionError)),
        reraise=True,
    )
    async def _call_api_stream(self, messages, max_tok, temp):
        self._breaker._update_state()
        if self._breaker.state == "OPEN":
            from src.utils.circuit_breaker import CircuitBreakerOpenException

            raise CircuitBreakerOpenException(f"Circuit breaker OPEN for {self.model_id}")
        try:
            result = await self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_tok,
                temperature=temp,
                stream=True,
                stream_options={"include_usage": True},
            )
            self._breaker.record_success()
            return result
        except Exception:
            self._breaker.record_failure()
            raise

    async def agenerate(
        self,
        messages: List[Dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict:
        """Non-streaming async generation with exponential backoff on rate limits."""
        max_tok = max_tokens or self.max_tokens
        temp = temperature if temperature is not None else self.temperature

        try:
            response = await self._call_api(messages, max_tok, temp)
            return {
                "text": response.choices[0].message.content.strip(),
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            }
        except Exception as e:
            logger.error(f"OpenRouter API unexpected error: {e}")
            raise

    async def astream(
        self,
        messages: List[Dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """Streaming async generation yielding tokens."""
        max_tok = max_tokens or self.max_tokens
        temp = temperature if temperature is not None else self.temperature

        try:
            stream = await self._call_api_stream(messages, max_tok, temp)
            async for chunk in stream:
                if (
                    chunk.choices
                    and len(chunk.choices) > 0
                    and chunk.choices[0].delta.content is not None
                ):
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenRouter API stream error: {e}")
            yield f"Error: {str(e)}"

    def count_tokens(self, text: str) -> int:
        """Estimate token count locally using heuristic to avoid tiktoken hangs."""
        return len(text) // 4

    def get_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate actual cost in USD from real token counts."""
        return (input_tokens / 1000) * self.cost_per_1k_input + (
            output_tokens / 1000
        ) * self.cost_per_1k_output
