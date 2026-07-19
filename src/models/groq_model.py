"""
Groq LLM provider — implements BaseLLM using the native AsyncGroq client.

Key upgrades over the previous synchronous version:
- Uses `AsyncGroq` so network I/O is non-blocking (no thread starvation).
- Inherits from `BaseLLM` so the pipeline is provider-agnostic.
- Exponential backoff on RateLimitError uses `asyncio.sleep` (non-blocking).
- Streaming is an async generator compatible with FastAPI's StreamingResponse.
"""

import asyncio
import os
from typing import AsyncIterator, Dict, List, Optional

from groq import AsyncGroq, RateLimitError

from src.models.base import BaseLLM
from src.utils.logger import logger


class GroqModel(BaseLLM):
    """Async Groq LLM — wraps AsyncGroq with retry logic and cost tracking."""

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

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Get your FREE key at: https://console.groq.com/keys"
            )

        # AsyncGroq — every API call is a coroutine; never blocks the event loop.
        self.client = AsyncGroq(api_key=api_key)
        logger.info(f"GroqModel initialized (async): {model_id}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        prompt: str,
        context: str,
        history: Optional[List[Dict]],
    ) -> List[Dict]:
        """Compose the messages list sent to the Groq API."""
        if context:
            system_msg = (
                "You are a helpful AI assistant. "
                "Use the provided context to answer the question accurately and concisely."
            )
            user_msg = f"Context:\n{context}\n\nQuestion: {prompt}"
        else:
            system_msg = (
                "You are a helpful AI assistant. Answer questions accurately and concisely."
            )
            user_msg = prompt

        return [
            {"role": "system", "content": system_msg},
            *(history or []),
            {"role": "user", "content": user_msg},
        ]

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    async def agenerate(
        self,
        prompt: str,
        context: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        history: Optional[List[Dict]] = None,
    ) -> Dict:
        """Non-streaming async generation with exponential backoff on rate limits."""
        from src.utils.guardrails import validate_query

        validate_query(prompt)

        messages = self._build_messages(prompt, context, history)
        max_tok = max_tokens or self.max_tokens
        temp = temperature if temperature is not None else self.temperature

        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=max_tok,
                    temperature=temp,
                )
                return {
                    "text": response.choices[0].message.content.strip(),
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                }

            except RateLimitError:
                if attempt < self.max_retries - 1:
                    wait = 2**attempt  # 1s, 2s, 4s
                    logger.warning(
                        f"Groq rate-limited — retrying in {wait}s (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(wait)  # non-blocking sleep
                else:
                    logger.error("Groq rate limit exceeded after all retries")
                    raise

            except Exception as e:
                logger.error(f"Groq API error: {e}")
                raise

        raise RuntimeError("max_retries must be > 0")

    async def astream(
        self,
        prompt: str,
        context: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        history: Optional[List[Dict]] = None,
    ) -> AsyncIterator[str]:
        """Async token streaming — yields chunks as they arrive from Groq."""
        from src.utils.guardrails import validate_query

        validate_query(prompt)

        messages = self._build_messages(prompt, context, history)
        max_tok = max_tokens or self.max_tokens
        temp = temperature if temperature is not None else self.temperature

        from groq import AsyncStream
        from groq.types.chat import ChatCompletionChunk

        stream: AsyncStream[ChatCompletionChunk] = await self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tok,
            temperature=temp,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def count_tokens(self, text: str) -> int:
        """Estimate token count locally using tiktoken (no API call)."""
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            logger.warning("tiktoken not installed — using character heuristic")
            return len(text) // 4

    def get_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate actual cost in USD from real token counts."""
        return (input_tokens / 1000) * self.cost_per_1k_input + (
            output_tokens / 1000
        ) * self.cost_per_1k_output

    def get_info(self) -> Dict:
        return {
            "model_id": self.model_id,
            "provider": "groq",
            "cost_per_1k_input": self.cost_per_1k_input,
            "cost_per_1k_output": self.cost_per_1k_output,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
