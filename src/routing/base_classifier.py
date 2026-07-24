from abc import ABC, abstractmethod
from typing import Tuple


class BaseClassifier(ABC):
    """Abstract base class for query complexity classifiers."""

    @abstractmethod
    async def predict(self, query: str) -> Tuple[str, float]:
        """
        Classify the complexity of a query.

        Args:
            query: The user's input string.

        Returns:
            A tuple of (complexity_level, confidence).
            complexity_level should be "simple", "medium", or "complex".
            confidence should be a float between 0.0 and 1.0.
        """
        pass
