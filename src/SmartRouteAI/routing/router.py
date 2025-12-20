from typing import Dict, Tuple
from src.routing.classifier import ComplexityClassifier
from src.utils.config import load_routing_config
from src.utils.logging import logger


class QueryRouter:
    """
    Intelligent query router - YOUR CORE INNOVATION
    Routes queries to appropriate models based on complexity
    """
    
    def __init__(self, classifier_path: str = None):
        self.config = load_routing_config()
        self.classifier = ComplexityClassifier(classifier_path)
        self.default_strategy = self.config.get('default_strategy', 'cost_optimized')
        
        logger.info(f"Router initialized with strategy: {self.default_strategy}")
    
    def route(
        self,
        query: str,
        strategy: str = None,
        user_context: Dict = None
    ) -> Dict:
        """
        Route query to appropriate model
        
        Returns:
            {
                'model_id': str,
                'complexity': str,
                'confidence': float,
                'fallback_model': str,
                'strategy': str,
                'reason': str
            }
        """
        
        strategy = strategy or self.default_strategy
        
        # Step 1: Classify query complexity (YOUR ML MODEL)
        complexity, confidence = self.classifier.predict(query)
        
        logger.info(f"Query classified as {complexity} (confidence: {confidence:.2f})")
        
        # Step 2: Get routing rules for strategy
        if strategy not in self.config['strategies']:
            logger.warning(f"Unknown strategy {strategy}, using {self.default_strategy}")
            strategy = self.default_strategy
        
        strategy_rules = self.config['strategies'][strategy]
        
        # Step 3: Get model assignment based on complexity
        if complexity in strategy_rules:
            rules = strategy_rules[complexity]
            model_id = rules['model']
            fallback = rules.get('fallback', model_id)
            quality_threshold = rules.get('quality_threshold', 0.0)
            
            # Check if confidence meets threshold
            if confidence < quality_threshold and fallback:
                logger.info(
                    f"Confidence {confidence:.2f} below threshold {quality_threshold}, "
                    f"using fallback"
                )
                model_id = fallback
                reason = "low_confidence_escalation"
            else:
                reason = "normal_routing"
        else:
            # Fallback to medium complexity
            rules = strategy_rules.get('medium', strategy_rules['simple'])
            model_id = rules['model']
            fallback = rules.get('fallback', model_id)
            reason = "default_routing"
        
        return {
            'model_id': model_id,
            'complexity': complexity,
            'confidence': confidence,
            'fallback_model': fallback,
            'strategy': strategy,
            'reason': reason,
            'query_length': len(query.split())
        }
    
    def get_routing_stats(self, history: list) -> Dict:
        """Calculate routing statistics"""
        
        if not history:
            return {}
        
        complexity_counts = {'simple': 0, 'medium': 0, 'complex': 0}
        model_counts = {}
        
        for record in history:
            complexity_counts[record['complexity']] += 1
            model_id = record['model_id']
            model_counts[model_id] = model_counts.get(model_id, 0) + 1
        
        total = len(history)
        
        return {
            'total_queries': total,
            'complexity_distribution': {
                k: f"{(v/total)*100:.1f}%" 
                for k, v in complexity_counts.items()
            },
            'model_distribution': {
                k: f"{(v/total)*100:.1f}%" 
                for k, v in model_counts.items()
            },
            'avg_confidence': sum(r['confidence'] for r in history) / total
        }
    
    def explain_routing(self, routing_decision: Dict) -> str:
        """Generate human-readable explanation"""
        
        explanation = f"""
Query Routing Decision:
- Complexity: {routing_decision['complexity']} (confidence: {routing_decision['confidence']:.2%})
- Selected Model: {routing_decision['model_id']}
- Strategy: {routing_decision['strategy']}
- Fallback: {routing_decision['fallback_model']}
- Reason: {routing_decision['reason']}
        """.strip()
        
        return explanation
    
    def update_strategy(self, strategy: str):
        """Update default routing strategy"""
        if strategy in self.config['strategies']:
            self.default_strategy = strategy
            logger.info(f"Strategy updated to: {strategy}")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")