"""
Async InferencePipeline — orchestrates the full query pipeline.

Flow: Guardrails → Routing → Budget → Retrieval → Generation → Tracking
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.cost.budget import BudgetManager
from src.cost.tracker import CostTracker
from src.memory.conversation import ConversationMemory
from src.models.model_manager import ModelManager
from src.retrieval.retriever import DocumentRetriever
from src.routing.router import QueryRouter
from src.utils.guardrails import GuardrailViolation, validate_query
from src.utils.logger import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class InferencePipeline:
    """Async query processing pipeline."""

    def __init__(
        self,
        config_dir: Path = _PROJECT_ROOT / "config",
        classifier_path: Optional[Path] = None,
    ):
        logger.info("Initializing inference pipeline...")

        if classifier_path is None:
            classifier_path = (
                _PROJECT_ROOT / "models" / "classifiers" / "complexity_classifier.pkl"
            )

        self.router = QueryRouter(
            routing_config_path=config_dir / "routing.yaml",
            classifier_path=classifier_path if classifier_path.exists() else None,
        )
        self.model_manager = ModelManager(config_path=config_dir / "models.yaml")
        self.retriever = DocumentRetriever(
            persist_dir=_PROJECT_ROOT / "data" / "embeddings",
            top_k=5,
            max_distance=1.5,
        )
        self.tracker = CostTracker()
        self.budget_manager = BudgetManager(
            tracker=self.tracker,
            config_path=config_dir / "routing.yaml",
        )
        self.memory = ConversationMemory()
        logger.info("Pipeline ready (async)")

    async def run(
        self,
        query: str,
        strategy: Optional[str] = None,
        use_retrieval: bool = True,
        session_id: Optional[str] = None,
    ) -> Dict:
        """Process a single query through the full pipeline."""
        start_time = time.time()

        try:
            query = validate_query(query)
        except GuardrailViolation as e:
            logger.warning(f"Query blocked by guardrails: {e}")
            return self._error_response(
                str(e), "guardrail_violation", time.time() - start_time
            )

        try:
            # Route
            routing_decision = self.router.route(query, strategy)
            model_id = routing_decision["model_id"]
            complexity = routing_decision["complexity"]
            logger.info(
                f"Routed → {model_id} "
                f"(complexity={complexity}, confidence={routing_decision['confidence']:.2f})"
            )

            # Budget check
            estimated_cost = self.budget_manager.estimate_query_cost(
                model_id, len(query)
            )
            can_afford, reason = self.budget_manager.check_budget(estimated_cost)
            if not can_afford:
                logger.warning(
                    f"Budget exceeded ({reason}) — falling back to cheapest model"
                )
                model_id = "llama_3_1_8b"
                routing_decision["model_id"] = model_id
                routing_decision["reason"] = f"budget_{reason}"

            # Retrieval — sync call offloaded to executor to avoid blocking event loop
            context: str = ""
            sources: list = []
            if use_retrieval:
                loop = asyncio.get_event_loop()
                context, sources = await loop.run_in_executor(
                    None, self.retriever.retrieve, query
                )
                logger.info(f"Retrieved {len(sources)} sources")

            # Generate
            model = self.model_manager.load_model(model_id)
            full_prompt = f"{context}\n\n{query}" if context else query
            input_tokens = model.count_tokens(full_prompt)
            history = self.memory.get_history(session_id) if session_id else []

            result = await model.agenerate(
                prompt=query,
                context=context,
                max_tokens=1000,
                temperature=0.7,
                history=history,
            )

            answer = result["text"]
            output_tokens = result["output_tokens"]
            actual_cost = model.get_cost(input_tokens, output_tokens)
            latency = time.time() - start_time

            # Track cost
            self.tracker.log_query(
                query=query,
                model_id=model_id,
                complexity=complexity,
                strategy=routing_decision["strategy"],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=actual_cost,
                latency=latency,
                success=True,
            )

            if session_id:
                self.memory.add_turn(session_id, query, answer)

            logger.info(f"Query done: cost=${actual_cost:.4f}, latency={latency:.2f}s")

            return {
                "answer": answer,
                "model_used": model_id,
                "complexity": complexity,
                "confidence": routing_decision["confidence"],
                "cost": actual_cost,
                "latency": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "context": context,
                "sources": sources,
                "routing_info": routing_decision,
                "success": True,
                "error": None,
            }

        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            if "model_id" in locals() and "complexity" in locals():
                self.tracker.log_query(
                    query=query,
                    model_id=model_id,
                    complexity=complexity,
                    strategy=strategy or "unknown",
                    input_tokens=0,
                    output_tokens=0,
                    cost=0.0,
                    latency=latency,
                    success=False,
                )
            return self._error_response(str(e), str(e), latency)

    async def batch_run(
        self,
        queries: List[str],
        strategy: Optional[str] = None,
        use_retrieval: bool = True,
    ) -> List[Dict]:
        """Process multiple queries concurrently via asyncio.gather."""
        if not queries:
            return []
        logger.info(f"Batch processing {len(queries)} queries...")
        tasks = [
            self.run(query=q, strategy=strategy, use_retrieval=use_retrieval)
            for q in queries
        ]
        return list(await asyncio.gather(*tasks))

    def get_statistics(self, days: int = 1) -> Dict:
        return self.tracker.get_statistics(days)

    def get_savings(self, days: int = 1) -> Dict:
        return self.tracker.calculate_savings(days)

    @staticmethod
    def _error_response(answer: str, error: str, latency: float) -> Dict:
        return {
            "answer": answer,
            "model_used": None,
            "complexity": None,
            "confidence": 0.0,
            "cost": 0.0,
            "latency": latency,
            "input_tokens": 0,
            "output_tokens": 0,
            "context": "",
            "sources": [],
            "routing_info": {},
            "success": False,
            "error": error,
        }
