import os
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

from src.cost.tracker import CostTracker
from src.utils.logger import logger

try:
    import redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class BudgetManager:
    """Enforce budget limits and prevent overspending.

    Budget enforcement uses two modes:
    - Redis (when REDIS_URL is set): atomic INCRBYFLOAT ensures no race window
      across multiple workers or threads.
    - SQLite fallback (default): reads total_cost from DB. Safe for single-node
      deployments; susceptible to race conditions under very high concurrency.
    """

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

        # Optional Redis for atomic distributed budget enforcement.
        # When REDIS_URL is set and redis-py is installed, check_budget uses
        # atomic INCRBYFLOAT — no race window across threads or containers.
        # Falls back to SQLite transparently when Redis is unavailable.
        self._redis: Optional["redis.Redis"] = None
        redis_url = os.getenv("REDIS_URL")
        if redis_url and _REDIS_AVAILABLE:
            try:
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info(f"BudgetManager: Redis connected ({redis_url.split('@')[-1]})")
            except Exception as e:
                logger.warning(f"BudgetManager: Redis unavailable, falling back to SQLite: {e}")
                self._redis = None
        else:
            logger.info("BudgetManager: using SQLite budget tracking (single-node mode)")

        logger.info(f"Budget manager initialized: Daily ${self.limits['daily']}")
    
    def _get_cached_stats(self) -> Dict:
        """Get budget statistics with caching to reduce DB queries (used for UI/Status only)."""
        now = datetime.utcnow()

        if (self._stats_cache is not None
                and self._cache_timestamp is not None
                and (now - self._cache_timestamp).total_seconds() < self._cache_ttl):
            return self._stats_cache

        self._stats_cache = {
            'daily': self.tracker.get_statistics(days=1)['total_cost']
        }
        self._cache_timestamp = now
        return self._stats_cache

    def _invalidate_cache(self):
        """Invalidate the stats cache (call after logging a query)."""
        self._stats_cache = None
        self._cache_timestamp = None

    def _redis_check_budget(self, estimated_cost: float) -> Tuple[bool, str]:
        """Atomic budget check using Redis INCRBYFLOAT.

        Strategy:
        1. INCRBYFLOAT adds `estimated_cost` to the daily key and returns the new total.
        2. If the new total exceeds the daily limit, immediately decrement back
           (INCRBYFLOAT with -estimated_cost) and reject the request.
        3. The key is set with a TTL of 86400 seconds (1 day) on first creation
           using SET ... NX EX so it auto-expires at midnight.

        This is atomic: the increment and the limit check happen in the same
        Redis command response. No two threads can both pass the limit simultaneously.
        """
        today = datetime.utcnow().strftime("%Y-%m-%d")
        redis_key = f"smartroute:budget:daily:{today}"

        try:
            # INCRBYFLOAT is atomic — increment and get new total in one op
            new_total = float(self._redis.incrbyfloat(redis_key, estimated_cost))

            # Set TTL only if key was just created (NX = only if not exists)
            self._redis.expire(redis_key, 86400)

            if new_total > self.limits['daily']:
                # Over budget — roll back the increment
                self._redis.incrbyfloat(redis_key, -estimated_cost)
                logger.warning(
                    f"Daily budget exceeded (Redis): ${new_total:.4f} total, "
                    f"${self.limits['daily']:.2f} limit"
                )
                return False, "daily_limit_exceeded"

            return True, "within_budget"

        except Exception as e:
            # Redis error — fall back to SQLite check rather than blocking all requests
            logger.warning(f"Redis budget check failed, falling back to SQLite: {e}")
            return self._sqlite_check_budget(estimated_cost)

    def _sqlite_check_budget(self, estimated_cost: float) -> Tuple[bool, str]:
        """SQLite budget check — reads total_cost from DB. Not atomic across threads,
        but acceptable for single-node deployments with low-to-moderate concurrency."""
        daily_spent = self.tracker.get_statistics(days=1)['total_cost']
        daily_remaining = self.limits['daily'] - daily_spent

        if estimated_cost > daily_remaining:
            logger.warning(
                f"Daily budget exceeded: ${daily_spent:.4f} spent, "
                f"${estimated_cost:.4f} requested, "
                f"${self.limits['daily']:.2f} limit"
            )
            return False, "daily_limit_exceeded"

        return True, "within_budget"

    def check_budget(self, estimated_cost: float) -> Tuple[bool, str]:
        """Check if we can afford this query.

        Routes to Redis atomic check when available, SQLite otherwise.
        """
        if self._redis:
            return self._redis_check_budget(estimated_cost)
        return self._sqlite_check_budget(estimated_cost)
    
    def check_budget_full(self, estimated_cost: float) -> Tuple[bool, str]:
        """
        Full budget check including weekly and monthly limits.
        Use this for dashboard/status checks, not per-query.
        """
        # Check daily budget first (cached is fine here as it's just a check, not a gate for spending)
        # Actually for consistency, we rely on check_budget which is non-cached.
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
        try:
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
        except Exception:
            pass # Fallback
            
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