"""
Async InferencePipeline — orchestrates the full query pipeline.

Flow: Guardrails → Routing → Budget → Retrieval → Generation → Tracking
"""

import asyncio
import time
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

from langfuse.decorators import langfuse_context, observe

from src.cost.budget import BudgetManager
from src.cost.tracker import CostTracker
from src.memory.conversation import ConversationMemory
from src.models.model_manager import ModelManager
from src.retrieval.retriever import DocumentRetriever
from src.retrieval.semantic_cache import SemanticCache
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
        self.semantic_cache = SemanticCache()
        self.memory = ConversationMemory()
        logger.info("Pipeline ready (async)")

    @staticmethod
    def _build_messages(
        prompt: str,
        context: str,
        history: Optional[List[Dict]],
    ) -> List[Dict]:
        """Compose the messages list sent to the API."""
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

    async def _prepare_context(
        self,
        query: str,
        strategy: Optional[str],
        use_retrieval: bool,
        start_time: float,
    ) -> Dict:
        """Extract common pipeline setup steps."""
        # 0. Check Semantic Cache
        cached_result = await self.semantic_cache.get(query)
        if cached_result:
            cached_result["latency"] = time.time() - start_time
            await asyncio.to_thread(
                self.tracker.log_query,
                query=query,
                model_id="semantic_cache",
                complexity="cached",
                strategy="cache",
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                latency=cached_result["latency"],
                success=True,
            )
            return {"is_cached": True, "cached_result": cached_result}

        # Route
        routing_decision = await self.router.route(query, strategy)
        model_id = routing_decision["model_id"]
        complexity = routing_decision["complexity"]
        logger.info(
            f"Routed → {model_id} "
            f"(complexity={complexity}, confidence={routing_decision['confidence']:.2f})"
        )

        # Budget check
        estimated_cost = self.budget_manager.estimate_query_cost(model_id, len(query))
        can_afford, reason = await self.budget_manager.check_budget(estimated_cost)
        if not can_afford:
            logger.warning(f"Budget exceeded ({reason}) — falling back to cheapest model")
            model_id = "llama_3_1_8b"
            routing_decision["model_id"] = model_id
            routing_decision["reason"] = f"budget_{reason}"

        # Retrieval
        context = ""
        sources = []
        if use_retrieval:
            context, sources = await self.retriever.retrieve(query)
            logger.info(f"Retrieved {len(sources)} sources")

        return {
            "is_cached": False,
            "model_id": model_id,
            "complexity": complexity,
            "routing_decision": routing_decision,
            "context": context,
            "sources": sources,
        }

    @observe()
    async def run(
        self,
        query: str,
        strategy: Optional[str] = None,
        use_retrieval: bool = True,
        session_id: Optional[str] = None,
    ) -> Dict:
        """Process a single query through the full pipeline."""
        if session_id:
            langfuse_context.update_current_trace(session_id=session_id)
        start_time = time.time()

        try:
            query = validate_query(query)
        except GuardrailViolation as e:
            logger.warning(f"Query blocked by guardrails: {e}")
            return self._error_response(str(e), "guardrail_violation", time.time() - start_time)

        try:
            prep = await self._prepare_context(query, strategy, use_retrieval, start_time)
            if prep["is_cached"]:
                return prep["cached_result"]

            model_id = prep["model_id"]
            complexity = prep["complexity"]
            routing_decision = prep["routing_decision"]
            context = prep["context"]
            sources = prep["sources"]

            # Generate
            model = self.model_manager.load_model(model_id)
            history = await self.memory.get_history(session_id) if session_id else []
            messages = self._build_messages(query, context, history)

            # Rough token estimate (we might need a real tokenizer for messages eventually)
            full_prompt = f"{context}\n\n{query}" if context else query
            input_tokens = model.count_tokens(full_prompt)

            result = await model.agenerate(
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
            )

            answer = result["text"]
            output_tokens = result["output_tokens"]
            actual_cost = model.get_cost(input_tokens, output_tokens)
            latency = time.time() - start_time

            # Track cost
            await asyncio.to_thread(
                self.tracker.log_query,
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
                await self.memory.add_turn(session_id, query, answer)

            logger.info(f"Query done: cost=${actual_cost:.4f}, latency={latency:.2f}s")

            response_payload = {
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

            # Fire off background task to cache the response
            asyncio.create_task(self.semantic_cache.set(query, response_payload))

            return response_payload

        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            if "model_id" in locals() and "complexity" in locals():
                await asyncio.to_thread(
                    self.tracker.log_query,
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

    @observe()
    async def astream_run(
        self,
        query: str,
        strategy: Optional[str] = None,
        use_retrieval: bool = True,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[Dict]:
        """
        Process query and stream the response.
        Yields dictionaries:
          - {"type": "metadata", "data": {...}} at start (routing info, sources)
          - {"type": "chunk", "content": "..."} during generation
          - {"type": "done", "result": {...}} at the end with full payload and costs
        """
        if session_id:
            langfuse_context.update_current_trace(session_id=session_id)
        start_time = time.time()

        try:
            query = validate_query(query)
        except GuardrailViolation as e:
            logger.warning(f"Query blocked by guardrails: {e}")
            yield {
                "type": "done",
                "result": self._error_response(
                    str(e), "guardrail_violation", time.time() - start_time
                ),
            }
            return

        try:
            prep = await self._prepare_context(query, strategy, use_retrieval, start_time)
            if prep["is_cached"]:
                cached_result = prep["cached_result"]
                yield {
                    "type": "metadata",
                    "data": {
                        "routing_info": {"model_id": "semantic_cache", "complexity": "cached"},
                        "sources": cached_result.get("sources", []),
                    },
                }
                yield {"type": "chunk", "content": cached_result["answer"]}
                yield {"type": "done", "result": cached_result}
                return

            model_id = prep["model_id"]
            complexity = prep["complexity"]
            routing_decision = prep["routing_decision"]
            context = prep["context"]
            sources = prep["sources"]

            yield {
                "type": "metadata",
                "data": {"routing_info": routing_decision, "sources": sources},
            }

            # Generate (Stream)
            model = self.model_manager.load_model(model_id)
            history = await self.memory.get_history(session_id) if session_id else []
            messages = self._build_messages(query, context, history)

            full_prompt = f"{context}\n\n{query}" if context else query
            input_tokens = model.count_tokens(full_prompt)

            stream = model.astream(
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
            )

            full_answer = ""
            async for chunk in stream:
                full_answer += chunk
                yield {"type": "chunk", "content": chunk}

            # If fallback needed due to error
            if full_answer.startswith("Error:"):
                fallback_id = routing_decision.get("fallback", "openrouter/free")
                model_id = fallback_id
                routing_decision["strategy"] = "model_fallback"
                fallback_model = self.model_manager.load_model(model_id)
                input_tokens = fallback_model.count_tokens(full_prompt)

                stream = fallback_model.astream(messages=messages, max_tokens=1000, temperature=0.7)
                full_answer = ""
                async for chunk in stream:
                    full_answer += chunk
                    yield {"type": "chunk", "content": chunk}
                model = fallback_model

            output_tokens = model.count_tokens(full_answer)
            actual_cost = model.get_cost(input_tokens, output_tokens)
            latency = time.time() - start_time

            # Track cost
            await asyncio.to_thread(
                self.tracker.log_query,
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
                await self.memory.add_turn(session_id, query, full_answer)

            response_payload = {
                "answer": full_answer,
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

            asyncio.create_task(self.semantic_cache.set(query, response_payload))
            yield {"type": "done", "result": response_payload}

        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Pipeline stream failed: {e}", exc_info=True)
            yield {"type": "chunk", "content": f"\n\nError: {str(e)}"}
            yield {"type": "done", "result": self._error_response(str(e), str(e), latency)}

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
        tasks = [self.run(query=q, strategy=strategy, use_retrieval=use_retrieval) for q in queries]
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
