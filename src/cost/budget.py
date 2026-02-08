import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

from src.cost.tracker import CostTracker
from src.utils.logger import logger


class BudgetManager:
    """Enforce budget limits and prevent overspending with caching."""
    
    def __init__(
        self,
        tracker: CostTracker,
        config_path: Path = Path("config/routing.yaml"),
        cache_ttl_seconds: int = 60  
    ):
        self.tracker = tracker
        self._cache_ttl = cache_ttl_seconds
        self._stats_cache: Optional[Dict] = None
        self._cache_timestamp: Optional[datetime] = None
        
        # Load budget configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        budgets = config.get('budgets', {})
        self.limits = {
            'daily': budgets.get('daily', 10.0),
            'weekly': budgets.get('weekly', 50.0),
            'monthly': budgets.get('monthly', 200.0)
        }
        
        self.alert_threshold = budgets.get('alert_threshold', 0.8)
        self.emergency_stop = budgets.get('emergency_stop', True)
        
        logger.info(f"Budget manager initialized: Daily ${self.limits['daily']}")
    
    def _get_cached_stats(self) -> Dict:
        """Get budget statistics with caching to reduce DB queries."""
        now = datetime.utcnow()
        
        # Check if cache is valid
        if (self._stats_cache is not None and 
            self._cache_timestamp is not None and
            (now - self._cache_timestamp).total_seconds() < self._cache_ttl):
            return self._stats_cache
        
        # Refresh cache - single query for daily stats (most restrictive)
        # Weekly and monthly are only checked when daily passes
        self._stats_cache = {
            'daily': self.tracker.get_statistics(days=1)['total_cost']
        }
        self._cache_timestamp = now
        return self._stats_cache
    
    def _invalidate_cache(self):
        """Invalidate the stats cache (call after logging a query)."""
        self._stats_cache = None
        self._cache_timestamp = None
    
    def check_budget(self, estimated_cost: float) -> Tuple[bool, str]:
        """
        Check if we can afford this query (optimized with caching).
        
        Only checks daily budget by default (most restrictive).
        Weekly/monthly checked only on dashboard requests.
        
        Args:
            estimated_cost: Estimated cost of the query
        
        Returns:
            (can_afford, reason) tuple
        """
        # Get cached daily stats
        stats = self._get_cached_stats()
        daily_spent = stats['daily']
        daily_remaining = self.limits['daily'] - daily_spent
        
        if estimated_cost > daily_remaining:
            logger.warning(
                f"Daily budget exceeded: ${daily_spent:.4f} spent, "
                f"${estimated_cost:.4f} requested, "
                f"${self.limits['daily']:.2f} limit"
            )
            return False, "daily_limit_exceeded"
        
        return True, "within_budget"
    
    def check_budget_full(self, estimated_cost: float) -> Tuple[bool, str]:
        """
        Full budget check including weekly and monthly limits.
        Use this for dashboard/status checks, not per-query.
        """
        # Check daily budget first (cached)
        can_afford, reason = self.check_budget(estimated_cost)
        if not can_afford:
            return can_afford, reason
        
        # Check weekly budget
        weekly_spent = self.tracker.get_statistics(days=7)['total_cost']
        weekly_remaining = self.limits['weekly'] - weekly_spent
        
        if estimated_cost > weekly_remaining:
            logger.warning(
                f"Weekly budget exceeded: ${weekly_spent:.4f} spent, "
                f"${self.limits['weekly']:.2f} limit"
            )
            return False, "weekly_limit_exceeded"
        
        # Check monthly budget
        monthly_spent = self.tracker.get_statistics(days=30)['total_cost']
        monthly_remaining = self.limits['monthly'] - monthly_spent
        
        if estimated_cost > monthly_remaining:
            logger.warning(
                f"Monthly budget exceeded: ${monthly_spent:.4f} spent, "
                f"${self.limits['monthly']:.2f} limit"
            )
            return False, "monthly_limit_exceeded"
        
        return True, "within_budget"
    
    def get_budget_status(self) -> Dict:
        """Get current budget status for all periods"""
        
        daily_spent = self.tracker.get_statistics(days=1)['total_cost']
        weekly_spent = self.tracker.get_statistics(days=7)['total_cost']
        monthly_spent = self.tracker.get_statistics(days=30)['total_cost']
        
        def calculate_status(spent, limit):
            remaining = limit - spent
            percentage = (spent / limit * 100) if limit > 0 else 0
            alert = spent > (limit * self.alert_threshold)
            
            return {
                'spent': round(spent, 4),
                'limit': limit,
                'remaining': round(remaining, 4),
                'percentage': round(percentage, 2),
                'alert': alert
            }
        
        return {
            'daily': calculate_status(daily_spent, self.limits['daily']),
            'weekly': calculate_status(weekly_spent, self.limits['weekly']),
            'monthly': calculate_status(monthly_spent, self.limits['monthly']),
            'alert_threshold': self.alert_threshold,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def estimate_query_cost(
        self,
        model_id: str,
        query_length: int,
        model_config_path: Path = Path("config/models.yaml")
    ) -> float:
        """
        Estimate cost for a query before executing
        
        Args:
            model_id: ID of the model to use
            query_length: Length of query in characters
        
        Returns:
            Estimated cost in USD
        """
        
        # Load model costs
        with open(model_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check local models (free)
        if model_id in config.get('local_models', {}):
            return 0.0
        
        # Check API models
        if model_id in config.get('api_models', {}):
            model_config = config['api_models'][model_id]
            
            # Estimate tokens (rough: 4 chars per token)
            estimated_input_tokens = query_length // 4
            estimated_output_tokens = 500  # Average response
            
            cost_per_1k_input = model_config['cost_per_1k_input']
            cost_per_1k_output = model_config['cost_per_1k_output']
            
            input_cost = (estimated_input_tokens / 1000) * cost_per_1k_input
            output_cost = (estimated_output_tokens / 1000) * cost_per_1k_output
            
            return input_cost + output_cost
        
        # Unknown model, return conservative estimate
        return 0.05
    
    def should_alert(self) -> Dict:
        """Check if any budget alerts should be triggered"""
        
        status = self.get_budget_status()
        alerts = []
        
        for period, data in status.items():
            if period == 'alert_threshold' or period == 'timestamp':
                continue
            
            if data['alert']:
                alerts.append({
                    'period': period,
                    'spent': data['spent'],
                    'limit': data['limit'],
                    'percentage': data['percentage']
                })
        
        return {
            'should_alert': len(alerts) > 0,
            'alerts': alerts,
            'timestamp': status['timestamp']
        }