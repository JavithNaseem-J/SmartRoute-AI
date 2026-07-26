from typing import Tuple

from src.models.openrouter_model import OpenRouterModel
from src.routing.base_classifier import BaseClassifier
from src.utils.logger import logger


class LLMClassifier(BaseClassifier):
    """
    Uses a fast LLM (e.g. Llama-3-8B) to classify query complexity.
    Provides semantic understanding at the cost of ~200ms latency.
    """

    def __init__(self, model_id: str = "nvidia/nemotron-nano-9b-v2:free"):
        # We use a cheap, fast model for classification
        self.model = OpenRouterModel(
            model_id=model_id,
            max_tokens=10,
            temperature=0.0,
        )
        logger.info(f"LLMClassifier initialized with model: {model_id}")

    async def predict(self, query: str) -> Tuple[str, float]:
        """Classify complexity by prompting a fast LLM."""
        system_prompt = (
            "You are a routing agent for an API. "
            "Analyze the user's query and classify its complexity into exactly one of these three categories:\n"
            "- 'simple': Factual questions, definitions, basic greetings, short queries.\n"
            "- 'medium': "
            "How-to"
            " questions, comparisons, summaries, translations.\n"
            "- 'complex': Deep reasoning, architecture, writing code, debugging, analyzing data, math.\n\n"
            "Respond with EXACTLY ONE WORD (simple, medium, or complex) and nothing else. No punctuation."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        try:
            result = await self.model.agenerate(messages=messages, max_tokens=5, temperature=0.0)
            answer = result["text"].strip().lower()

            # Clean up the response in case the LLM was chatty
            if "complex" in answer:
                return "complex", 0.95
            elif "medium" in answer:
                return "medium", 0.95
            elif "simple" in answer:
                return "simple", 0.95
            else:
                logger.warning(
                    f"LLM Classifier returned unexpected output: '{answer}', defaulting to 'medium'"
                )
                return "medium", 0.50

        except Exception as e:
            logger.error(f"LLM Classifier failed: {e}, falling back to 'medium'")
            return "medium", 0.0
