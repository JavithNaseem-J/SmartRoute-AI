from pathlib import Path
from typing import Dict, Optional

import yaml  # type: ignore

from src.routing.classifier import ComplexityClassifier
from src.utils.guardrails import validate_query
from src.utils.logger import logger


class QueryRouter:
    """
    Intelligent query router
    Routes queries to cost-effective models based on complexity
    """

    def __init__(self, routing_config_path: Path, classifier_path: Optional[Path] = None):
        # Load routing configuration
        with open(routing_config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize classifier
        self.classifier = ComplexityClassifier(classifier_path)

        # Set default strategy
        self.default_strategy = self.config["default_strategy"]

        logger.info(f"Router initialized with strategy: {self.default_strategy}")

    def route(
        self,
        query: str,
        strategy: Optional[str] = None,
        user_context: Optional[Dict] = None,
    ) -> Dict:
        validate_query(query)

        # Use default strategy if not specified
        strategy = strategy or self.default_strategy

        if strategy not in self.config["strategies"]:
            logger.warning(f"Unknown strategy {strategy}, using {self.default_strategy}")
            strategy = self.default_strategy

        # Classify query complexity
        complexity, confidence = self.classifier.predict(query)

        logger.info(f"Query classified as {complexity} " f"(confidence: {confidence:.2f})")

        # Get routing rules for this strategy and complexity
        strategy_config = self.config["strategies"][strategy]

        if complexity not in strategy_config:
            # Fallback to medium if complexity not in strategy
            complexity = "medium"

        rules = strategy_config[complexity]

        # Select model based on rules
        model_id = rules["model"]
        fallback_model = rules.get("fallback", model_id)
        quality_threshold = rules.get("quality_threshold", 0.0)

        # Check if we need to escalate to fallback
        reason = "normal_routing"

        if confidence < quality_threshold:
            logger.info(
                f"Confidence {confidence:.2f} below threshold {quality_threshold}, "
                f"escalating to {fallback_model}"
            )
            model_id = fallback_model
            reason = "low_confidence_escalation"

        # Return routing decision
        return {
            "model_id": model_id,
            "complexity": complexity,
            "confidence": confidence,
            "fallback_model": fallback_model,
            "strategy": strategy,
            "reason": reason,
            "query_length": len(query.split()),
        }
