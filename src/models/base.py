"""
Abstract base class for all LLM providers.

Every LLM (Groq, OpenAI, Gemini, local vLLM) must implement this interface.
The InferencePipeline only depends on BaseLLM — swapping providers requires
zero changes to the pipeline or any other module.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, List, Optional


class BaseLLM(ABC):
    """Provider-agnostic LLM interface.

    Subclasses must implement:
        - async agenerate(...)  → standard non-streaming call
        - async astream(...)    → token-by-token streaming
        - count_tokens(...)     → local token counting (no API call)
        - get_cost(...)         → cost in USD for a token pair
        - get_info()            → provider metadata dict
    """

    @abstractmethod
    async def agenerate(
        self,
        prompt: str,
        context: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        history: Optional[List[Dict]] = None,
    ) -> Dict:
        """Generate a response asynchronously.

        Returns:
            {
                "text": str,            # the model's response
                "input_tokens": int,    # prompt tokens used
                "output_tokens": int,   # completion tokens generated
            }
        """

    @abstractmethod
    def astream(
        self,
        prompt: str,
        context: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        history: Optional[List[Dict]] = None,
    ) -> AsyncIterator[str]:
        """Stream response tokens asynchronously (Server-Sent Events)."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Estimate token count locally without an API call."""

    @abstractmethod
    def get_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Return cost in USD for the given token pair."""

    @abstractmethod
    def get_info(self) -> Dict:
        """Return provider metadata (model_id, provider, cost rates, etc.)."""
