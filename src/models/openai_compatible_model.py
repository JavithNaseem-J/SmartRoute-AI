from typing import AsyncGenerator, Dict, List, Optional

from openai import (
    APIConnectionError,
    AsyncOpenAI,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.models.base import BaseLLM
from src.utils.circuit_breaker import AsyncCircuitBreaker
from src.utils.logger import logger


class LLMProviderError(RuntimeError):
    """A provider request failed and must not be treated as answer text."""


class LLMAuthenticationError(LLMProviderError):
    """The active provider rejected or is missing its API key."""


class OpenAICompatibleModel(BaseLLM):
    """OpenAI-compatible model client for the active LLM provider."""

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str,
        model_id: str,
        cost_per_1k_input: float = 0.0,
        cost_per_1k_output: float = 0.0,
        max_tokens: int = 4096,
        temperature: float = 0.5,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        if not api_key:
            raise LLMAuthenticationError(
                f"LLM_API_KEY is not configured for provider '{provider}'."
            )

        self.provider = provider
        self.model_id = model_id
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers=default_headers,
            timeout=20.0,
        )
        self._breaker = AsyncCircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
        logger.info(f"LLM model initialized: provider={provider}, model={model_id}")

    async def _execute_with_breaker(self, api_func, **kwargs):
        self._breaker._update_state()
        if self._breaker.state == "OPEN":
            from src.utils.circuit_breaker import CircuitBreakerOpenException

            raise CircuitBreakerOpenException(f"Circuit breaker OPEN for {self.model_id}")
        try:
            result = await api_func(**kwargs)
            self._breaker.record_success()
            return result
        except AuthenticationError as error:
            logger.error(
                f"LLM authentication failed: provider={self.provider}, model={self.model_id}"
            )
            raise LLMAuthenticationError(
                f"The API key for provider '{self.provider}' is invalid or missing."
            ) from error
        except Exception:
            self._breaker.record_failure()
            raise

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, InternalServerError)),
        reraise=True,
    )
    async def _call_api(self, messages, max_tokens, temperature):
        return await self._execute_with_breaker(
            self.client.chat.completions.create,
            model=self.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, InternalServerError)),
        reraise=True,
    )
    async def _call_api_stream(self, messages, max_tokens, temperature):
        return await self._execute_with_breaker(
            self.client.chat.completions.create,
            model=self.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )

    async def agenerate(
        self,
        messages: List[Dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict:
        max_tokens = max_tokens or self.max_tokens
        temperature = self.temperature if temperature is None else temperature
        try:
            response = await self._call_api(messages, max_tokens, temperature)
        except LLMProviderError:
            raise
        except Exception as error:
            logger.error(
                f"LLM generation failed: provider={self.provider}, model={self.model_id}: {error}"
            )
            raise LLMProviderError(f"Provider '{self.provider}' generation failed.") from error

        content = response.choices[0].message.content
        if not content:
            raise LLMProviderError(f"Provider '{self.provider}' returned an empty response.")
        return {
            "text": content.strip(),
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        }

    async def astream(
        self,
        messages: List[Dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        max_tokens = max_tokens or self.max_tokens
        temperature = self.temperature if temperature is None else temperature
        stream_started = False
        emitted_content = False
        try:
            stream = await self._call_api_stream(messages, max_tokens, temperature)
            stream_started = True
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    emitted_content = True
                    yield chunk.choices[0].delta.content
            if not emitted_content:
                self._breaker.record_failure()
                raise LLMProviderError(f"Provider '{self.provider}' returned an empty stream.")
        except LLMProviderError:
            raise
        except Exception as error:
            if stream_started:
                self._breaker.record_failure()
            logger.error(
                f"LLM stream failed: provider={self.provider}, model={self.model_id}: {error}"
            )
            raise LLMProviderError(f"Provider '{self.provider}' stream failed.") from error

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4) if text else 0

    def get_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1000) * self.cost_per_1k_input + (
            output_tokens / 1000
        ) * self.cost_per_1k_output
