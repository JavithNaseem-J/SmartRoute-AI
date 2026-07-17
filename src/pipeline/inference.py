"""
Async InferencePipeline — enterprise-grade orchestration layer.

Enterprise upgrades:
1. Pipeline.run() is now `async def` — awaits the LLM directly instead of
   blocking a thread. FastAPI can serve thousands of concurrent queries on
   a single event loop without thread starvation.
2. batch_run() uses asyncio.gather() — true parallelism, no thread pool needed.
3. All hardcoded paths replaced with dynamic pathlib roots (Phase 4).
"""
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.routing.router import QueryRouter
from src.models.model_manager import ModelManager
from src.retrieval.retriever import DocumentRetriever
from src.cost.tracker import CostTracker
from src.cost.budget import BudgetManager
from src.memory.conversation import ConversationMemory
from src.utils.logger import logger
from src.utils.guardrails import validate_query, GuardrailViolation

# Resolve the project root dynamically — works from any working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class InferencePipeline:
    """
    Async query processing pipeline.
    Orchestrates: Guardrails → Routing → Budget → Retrieval → Generation → Tracking
    """

    def __init__(
        self,
        config_dir: Path = _PROJECT_ROOT / "config",
        classifier_path: Optional[Path] = None,
    ):
        logger.info("Initializing inference pipeline...")

        # Phase 4: dynamic path — no more hardcoded relative strings
        if classifier_path is None:
            classifier_path = _PROJECT_ROOT / "models" / "classifiers" / "complexity_classifier.pkl"

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

        logger.info("####### Pipeline ready (async) #######")

    async def run(
        self,
        query: str,
        strategy: Optional[str] = None,
        use_retrieval: bool = True,
        session_id: Optional[str] = None,
    ) -> Dict:
        """Process a single query asynchronously through the full pipeline."""
        start_time = time.time()

        # Step 0: Guardrails
        try:
            query = validate_query(query)
        except GuardrailViolation as e:
            logger.warning(f"Query blocked by guardrails: {e}")
            return self._error_response(str(e), "guardrail_violation", time.time() - start_time)

        try:
            # Step 1: Route
            routing_decision = self.router.route(query, strategy)
            model_id = routing_decision["model_id"]
            complexity = routing_decision["complexity"]
            logger.info(
                f"Routed → {model_id} "
                f"(complexity={complexity}, confidence={routing_decision['confidence']:.2f})"
            )

            # Step 2: Budget check
            estimated_cost = self.budget_manager.estimate_query_cost(model_id, len(query))
            can_afford, reason = self.budget_manager.check_budget(estimated_cost)
            if not can_afford:
                logger.warning(f"Budget exceeded ({reason}) — falling back to cheapest model")
                model_id = "llama_3_1_8b"
                routing_decision["model_id"] = model_id
                routing_decision["reason"] = f"budget_{reason}"

            # Step 3: Retrieval (async-friendly — runs in threadpool via asyncio)
            context, sources = "", []
            if use_retrieval:
                logger.info("Retrieving context...")
                # retriever.retrieve() is synchronous (ChromaDB) — run in executor
                # to avoid blocking the event loop during embedding/search.
                loop = asyncio.get_event_loop()
                context, sources = await loop.run_in_executor(
                    None, self.retriever.retrieve, query
                )
                logger.info(f"Retrieved {len(sources)} sources")

            # Step 4: Load model and generate — fully async, no thread needed
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

            # Step 5: Log cost (synchronous DB write — fast enough to not need async)
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

            # Step 6: Persist session turn
            if session_id:
                self.memory.add_turn(session_id, query, answer)

            logger.info(
                f"####### Query done: cost=${actual_cost:.4f}, latency={latency:.2f}s #######"
            )

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
        """Process multiple queries in true parallel using asyncio.gather.

        All queries are dispatched simultaneously. Total time ≈ max(individual latencies)
        instead of their sum. No thread pool needed.
        """
        if not queries:
            return []
        logger.info(f"Batch processing {len(queries)} queries concurrently...")
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