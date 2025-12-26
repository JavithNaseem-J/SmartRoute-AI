import yaml
from pathlib import Path
from typing import Dict, Optional

from .classifier import ComplexityClassifier
from ..utils.logger import logger


class QueryRouter:
    """
    Intelligent query router
    Routes queries to cost-effective models based on complexity
    """
    
    def __init__(
        self,
        routing_config_path: Path,
        classifier_path: Optional[Path] = None
    ):
        # Load routing configuration
        with open(routing_config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize classifier
        self.classifier = ComplexityClassifier(classifier_path)
        
        # Set default strategy
        self.default_strategy = self.config['default_strategy']
        
        logger.info(f"Router initialized with strategy: {self.default_strategy}")
    
    def route(
        self,
        query: str,
        strategy: Optional[str] = None,
        user_context: Optional[Dict] = None
    ) -> Dict:
        """
        Route query to appropriate model
        
        Args:
            query: User query string
            strategy: Routing strategy (cost_optimized, quality_first, balanced)
            user_context: Optional user context for personalization
        
        Returns:
            Dictionary with routing decision:
            {
                'model_id': str,
                'complexity': str,
                'confidence': float,
                'fallback_model': str,
                'strategy': str,
                'reason': str
            }
        """
        
        # Use default strategy if not specified
        strategy = strategy or self.default_strategy
        
        if strategy not in self.config['strategies']:
            logger.warning(f"Unknown strategy {strategy}, using {self.default_strategy}")
            strategy = self.default_strategy
        
        # Step 1: Classify query complexity
        complexity, confidence = self.classifier.predict(query)
        
        logger.info(
            f"Query classified as {complexity} "
            f"(confidence: {confidence:.2f})"
        )
        
        # Step 2: Get routing rules for this strategy and complexity
        strategy_config = self.config['strategies'][strategy]
        
        if complexity not in strategy_config:
            # Fallback to medium if complexity not in strategy
            complexity = 'medium'
        
        rules = strategy_config[complexity]
        
        # Step 3: Select model based on rules
        model_id = rules['model']
        fallback_model = rules.get('fallback', model_id)
        quality_threshold = rules.get('quality_threshold', 0.0)
        
        # Step 4: Check if we need to escalate to fallback
        reason = 'normal_routing'
        
        if confidence < quality_threshold:
            logger.info(
                f"Confidence {confidence:.2f} below threshold {quality_threshold}, "
                f"escalating to {fallback_model}"
            )
            model_id = fallback_model
            reason = 'low_confidence_escalation'
        
        # Step 5: Return routing decision
        return {
            'model_id': model_id,
            'complexity': complexity,
            'confidence': confidence,
            'fallback_model': fallback_model,
            'strategy': strategy,
            'reason': reason,
            'query_length': len(query.split())
        }
    
    def update_strategy(self, strategy: str):
        """Update default routing strategy"""
        if strategy in self.config['strategies']:
            self.default_strategy = strategy
            logger.info(f"Strategy updated to: {strategy}")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def get_strategy_info(self, strategy: Optional[str] = None) -> Dict:
        """Get information about a routing strategy"""
        strategy = strategy or self.default_strategy
        
        if strategy not in self.config['strategies']:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return {
            'strategy': strategy,
            'description': self.config['strategies'][strategy].get('description', ''),
            'rules': self.config['strategies'][strategy]
        }
    
    def explain_routing(self, routing_decision: Dict) -> str:
        """Generate human-readable explanation of routing decision"""
        
        explanation = f"""
Query Routing Decision:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Complexity:     {routing_decision['complexity']} 
Confidence:     {routing_decision['confidence']:.2%}
Selected Model: {routing_decision['model_id']}
Fallback Model: {routing_decision['fallback_model']}
Strategy:       {routing_decision['strategy']}
Reason:         {routing_decision['reason'].replace('_', ' ').title()}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Explanation:
The query was classified as '{routing_decision['complexity']}' with 
{routing_decision['confidence']:.0%} confidence. Based on the 
'{routing_decision['strategy']}' strategy, the query was routed to 
'{routing_decision['model_id']}'.
        """.strip()
        
        return explanation
    
    def get_routing_stats(self, routing_history: list) -> Dict:
        """Calculate statistics from routing history"""
        if not routing_history:
            return {}
        
        total = len(routing_history)
        
        # Count by complexity
        complexity_counts = {'simple': 0, 'medium': 0, 'complex': 0}
        for record in routing_history:
            complexity_counts[record['complexity']] += 1
        
        # Count by model
        model_counts = {}
        for record in routing_history:
            model_id = record['model_id']
            model_counts[model_id] = model_counts.get(model_id, 0) + 1
        
        # Calculate average confidence
        avg_confidence = sum(r['confidence'] for r in routing_history) / total
        
        # Escalation rate
        escalations = sum(1 for r in routing_history if r['reason'] == 'low_confidence_escalation')
        escalation_rate = escalations / total
        
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
            'avg_confidence': avg_confidence,
            'escalation_rate': escalation_rate,
            'escalations': escalations
        }