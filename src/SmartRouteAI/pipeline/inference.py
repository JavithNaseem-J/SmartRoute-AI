import time
from typing import Dict
from src.models.model_manager import ModelManager
from src.routing.router import QueryRouter
from src.retrieval.retriever import DocumentRetriever
from src.cost.tracker import CostTracker
from src.cost.budget import BudgetManager
from src.utils.logging import logger


class HybridInferencePipeline:

    
    def __init__(self):
        self.retriever = DocumentRetriever()
        
        self.router = QueryRouter()
        self.model_manager = ModelManager()
        self.tracker = CostTracker()
        self.budget_manager = BudgetManager(self.tracker)
        
        logger.info("✓ Hybrid inference pipeline initialized")
    
    def run(self,query: str,strategy: str = None,use_retrieval: bool = True,collection_name: str = "documents") -> Dict:
        """
        Execute complete inference pipeline
        
        Returns dict with answer, cost, metadata
        """
        
        start_time = time.time()
        
        try:

            routing_decision = self.router.route(query, strategy)
            model_id = routing_decision['model_id']
            complexity = routing_decision['complexity']
            
            logger.info(
                f"[CUSTOM ROUTING] {complexity} → {model_id} "
                f"(confidence: {routing_decision['confidence']:.2f})"
            )
            

            estimated_cost = self.budget_manager.estimate_query_cost(model_id,len(query))
            
            can_afford, reason = self.budget_manager.check_budget(estimated_cost)
            
            if not can_afford:
                logger.warning(f"[BUDGET] Exceeded: {reason}, switching to free model")
                model_id = "llama_3_2_1b"
                routing_decision['model_id'] = model_id
                routing_decision['reason'] = f"budget_{reason}"

            
            context = ""
            sources = []
            
            if use_retrieval:
                retrieval_result = self.retriever.retrieve_with_context(
                    query,
                    collection_name=collection_name
                )
                context = retrieval_result['context']
                sources = retrieval_result['sources']
                
                logger.info(
                    f"[LANGCHAIN RETRIEVAL] Retrieved {retrieval_result['document_count']} docs"
                )
            

            
            model = self.model_manager.load_model(model_id)
            
            # Count input tokens
            full_prompt = f"{context}\n\n{query}" if context else query
            input_tokens = model.count_tokens(full_prompt)
            
            # Generate response
            if hasattr(model, 'generate') and hasattr(model.generate, '__call__'):
                # API model (returns dict)
                if context:
                    result = model.generate(prompt=query, context=context)
                else:
                    result = model.generate(prompt=query)
                
                # Extract response
                if isinstance(result, dict):
                    answer = result.get('text', result.get('answer', str(result)))
                    output_tokens = result.get('output_tokens', 0)
                else:
                    answer = str(result)
                    output_tokens = model.count_tokens(answer)
                
                actual_cost = model.get_cost(input_tokens, output_tokens)
            else:
                # Local model (returns string)
                answer = model.generate(prompt=query, context=context)
                output_tokens = model.count_tokens(answer)
                actual_cost = 0.0
            
            latency = time.time() - start_time
            

            
            self.tracker.log_query(
                query=query,model_id=model_id,complexity=complexity,
                strategy=routing_decision['strategy'],input_tokens=input_tokens,output_tokens=output_tokens,
                cost=actual_cost,latency=latency,success=True
            )
            
            logger.info(f" Query completed: cost=${actual_cost:.4f}, latency={latency:.2f}s")
            
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
            
            # Still track failures
            if 'model_id' in locals() and 'complexity' in locals():
                self.tracker.log_query(
                    query=query,model_id=model_id,complexity=complexity,
                    strategy=strategy or 'unknown',input_tokens=0,output_tokens=0,
                    cost=0.0,latency=latency,success=False
                )
            
            return {
                'answer': f"Error: {str(e)}",
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
    
    def batch_run(self, queries: list, **kwargs) -> list:
        """Run pipeline on multiple queries"""
        results = []
        for query in queries:
            result = self.run(query, **kwargs)
            results.append(result)
        return results