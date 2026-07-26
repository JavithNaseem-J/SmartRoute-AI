from typing import Tuple

from src.routing.base_classifier import BaseClassifier
from src.routing.keywords import COMPLEX_KEYWORDS, MEDIUM_KEYWORDS
from src.utils.logger import logger


class HeuristicClassifier(BaseClassifier):
    """
    Extremely fast, zero-dependency heuristic router.
    Uses regex, length, and keyword matching instead of ML.
    Latency: ~0.1ms
    """

    def __init__(self):
        self.complex_keywords = COMPLEX_KEYWORDS
        self.medium_keywords = MEDIUM_KEYWORDS
        logger.info("HeuristicClassifier initialized.")

    async def predict(self, query: str) -> Tuple[str, float]:
        """Classify complexity using fast heuristic rules."""
        query_lower = query.lower()
        word_count = len(query.split())

        import re

        # 1. Complex Triggers (Code, deep reasoning, long context, acronym density)
        has_code = "```" in query or "def " in query or "class " in query
        acronyms = len(re.findall(r"\b[A-Z]{3,}\b", query))

        if (
            has_code
            or word_count > 45
            or acronyms >= 2
            or any(k in query_lower for k in self.complex_keywords)
        ):
            return "complex", 1.0

        # 2. Medium Triggers (Multi-part questions, moderate length)
        is_multipart = query.count("?") > 1 or " and " in query_lower

        if is_multipart or word_count > 15 or any(k in query_lower for k in self.medium_keywords):
            return "medium", 1.0

        # 3. Simple Triggers (Short, factual, greeting)
        return "simple", 1.0
