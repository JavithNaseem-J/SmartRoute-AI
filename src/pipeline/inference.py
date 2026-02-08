import time
from pathlib import Path
from typing import Dict, Optional

from src.routing.router import QueryRouter
from src.models.model_manager import ModelManager
from src.retrieval.retriever import DocumentRetriever
from src.cost.tracker import CostTracker
from src.cost.budget import BudgetManager
from src.utils.logger import logger


class InferencePipeline:
    """
    Main query processing pipeline
    Orchestrates: Routing → Retrieval → Generation → Cost Tracking
    """
    
    def __init__(
        self,
        config_dir: Path = Path("config"),
        classifier_path: Optional[Path] = None
    ):
        logger.info("Initializing inference pipeline...")
        
        # Set default classifier path
        if classifier_path is None:
            classifier_path = Path("models/classifiers/complexity_classifier.pkl")
        
        # Initialize router (YOUR CORE INNOVATION)
        self.router = QueryRouter(
            routing_config_path=config_dir / "routing.yaml",
            classifier_path=classifier_path if classifier_path.exists() else None
        )
        
        # Initialize model manager
        self.model_manager = ModelManager(
            config_path=config_dir / "models.yaml"
        )
        
        # Initialize retriever (RAG)
        self.retriever = DocumentRetriever(
            persist_dir=Path("data/embeddings"),
            top_k=5,
            max_distance=1.5  # L2 distance threshold - lower is more similar
        )
        
        # Initialize cost tracking (BUSINESS VALUE)
        self.tracker = CostTracker()
        self.budget_manager = BudgetManager(
            tracker=self.tracker,
            config_path=config_dir / "routing.yaml"
        )
        
        logger.info("####### Pipeline ready #######")
    
    def run(
        self,
        query: str,
        strategy: Optional[str] = None,
        use_retrieval: bool = True
    ) -> Dict:
        start_time = time.time()
        
        try:
            # Step 1: Route query to appropriate model
            logger.info(f"Processing query: {query[:50]}...")
            
            routing_decision = self.router.route(query, strategy)
            model_id = routing_decision['model_id']
            complexity = routing_decision['complexity']
            
            logger.info(
                f"Routed to {model_id} "
                f"(complexity: {complexity}, "
                f"confidence: {routing_decision['confidence']:.2f})"
            )
            
            # Step 2: Check budget
            estimated_cost = self.budget_manager.estimate_query_cost(
                model_id,
                len(query)
            )
            
            can_afford, reason = self.budget_manager.check_budget(estimated_cost)
            
            if not can_afford:
                logger.warning(
                    f"Budget exceeded ({reason}), "
                    f"falling back to cheapest model"
                )
                model_id = "llama_3_1_8b"  # Cheapest tier model
                routing_decision['model_id'] = model_id
                routing_decision['reason'] = f"budget_{reason}"
            
            # Step 3: Retrieve context (RAG)
            context = ""
            sources = []
            
            if use_retrieval:
                logger.info("Retrieving context...")
                context, sources = self.retriever.retrieve(query)
                logger.info(f"Retrieved {len(sources)} sources")
            
            # Step 4: Load model and generate response
            logger.info(f"Generating with {model_id}...")
            model = self.model_manager.load_model(model_id)
            
            # Count input tokens
            full_prompt = f"{context}\n\n{query}" if context else query
            input_tokens = model.count_tokens(full_prompt)
            
            # Generate
            result = model.generate(
                prompt=query,
                context=context,
                max_tokens=1000,
                temperature=0.7
            )
            
            answer = result['text']
            output_tokens = result['output_tokens']
            
            # Calculate actual cost
            actual_cost = model.get_cost(input_tokens, output_tokens)
            
            latency = time.time() - start_time
            
            # Step 5: Log everything for cost tracking
            self.tracker.log_query(
                query=query,
                model_id=model_id,
                complexity=complexity,
                strategy=routing_decision['strategy'],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=actual_cost,
                latency=latency,
                success=True
            )
            
            logger.info(
                f"####### Query completed: "
                f"cost=${actual_cost:.4f}, "
                f"latency={latency:.2f}s #######"
            )
            
            return {
                'answer': answer,
                'model_used': model_id,
                'complexity': complexity,
                'confidence': routing_decision['confidence'],
                'cost': actual_cost,
                'latency': latency,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'context': context,
                'sources': sources,
                'routing_info': routing_decision,
                'success': True,
                'error': None
            }
        
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            
            # Still log failures for tracking
            if 'model_id' in locals() and 'complexity' in locals():
                self.tracker.log_query(
                    query=query,
                    model_id=model_id,
                    complexity=complexity,
                    strategy=strategy or 'unknown',
                    input_tokens=0,
                    output_tokens=0,
                    cost=0.0,
                    latency=latency,
                    success=False
                )
            
            return {
                'answer': f"Error processing query: {str(e)}",
                'model_used': None,
                'complexity': None,
                'confidence': 0.0,
                'cost': 0.0,
                'latency': latency,
                'input_tokens': 0,
                'output_tokens': 0,
                'context': '',
                'sources': [],
                'routing_info': {},
                'success': False,
                'error': str(e)
            }
    
    def batch_run(
        self,
        queries: list,
        strategy: Optional[str] = None,
        use_retrieval: bool = True
    ) -> list:
        """Process multiple queries"""
        results = []
        for query in queries:
            result = self.run(query, strategy, use_retrieval)
            results.append(result)
        return results
    
    def get_statistics(self, days: int = 1) -> Dict:
        """Get pipeline statistics"""
        return self.tracker.get_statistics(days)
    
    def get_savings(self, days: int = 1) -> Dict:
        """Get cost savings vs baseline"""
        return self.tracker.calculate_savings(days)