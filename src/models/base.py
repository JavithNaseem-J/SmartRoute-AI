from abc import ABC, abstractmethod
from typing import AsyncGenerator, AsyncIterator, Dict, List, Optional


class BaseLLM(ABC):
    """Provider-agnostic LLM interface.

    Subclasses must implement:
        - async agenerate(...)  → standard non-streaming call
        - astream(...)          → token-by-token streaming (async generator)
        - count_tokens(...)     → local token counting (no API call)
        - get_cost(...)         → cost in USD for a token pair
        - get_info()            → provider metadata dict
    """

    @abstractmethod
    async def agenerate(
        self,
        messages: List[Dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
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
        messages: List[Dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate response token-by-token (async generator)."""


    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Estimate token count locally without an API call."""

    @abstractmethod
    def get_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Return cost in USD for the given token pair."""
