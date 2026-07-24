"""
BudgetManager — Upstash Redis only.

REDIS_URL is REQUIRED. The app will refuse to start without it.
No SQLite/in-memory fallback — this system runs in the cloud.

Get your free Upstash Redis URL at: https://upstash.com
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import yaml  # type: ignore

from src.core.dependencies import get_redis_client
from src.cost.tracker import CostTracker
from src.utils.logger import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class BudgetManager:
    """Atomic budget enforcement via Upstash Redis.

    REDIS_URL must be set. Raises RuntimeError on startup if missing.
    Uses Redis INCRBYFLOAT — no race window under any concurrency level.
    """

    def __init__(
        self,
        tracker: CostTracker,
        config_path: Path = _PROJECT_ROOT / "config" / "routing.yaml",
    ):
        self.tracker = tracker

        self._redis = get_redis_client()

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        budgets = config.get("budgets", {})
        self.limits = {
            "daily": budgets.get("daily", 10.0),
            "weekly": budgets.get("weekly", 50.0),
            "monthly": budgets.get("monthly", 200.0),
        }
        self.alert_threshold = budgets.get("alert_threshold", 0.8)
        logger.info(
            f"BudgetManager: daily=${self.limits['daily']}, weekly=${self.limits['weekly']}"
        )

    def _redis_key(self, period: str) -> str:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return f"smartroute:budget:{period}:{today}"

    async def check_budget(self, estimated_cost: float) -> Tuple[bool, str]:
        """Atomic budget check using Redis INCRBYFLOAT.

        INCRBYFLOAT increments the key and returns the new total atomically.
        If over budget, immediately decrements back and rejects the request.
        No two concurrent requests can both pass the limit simultaneously.
        """
        key = self._redis_key("daily")
        try:
            new_total = float(await self._redis.incrbyfloat(key, estimated_cost))
            await self._redis.expire(key, 86400)  # auto-expire after 24h

            if new_total > self.limits["daily"]:
                await self._redis.incrbyfloat(key, -estimated_cost)  # roll back
                logger.warning(f"Daily budget exceeded: ${new_total:.4f} / ${self.limits['daily']}")
                import asyncio

                from src.utils.alerting import send_alert

                asyncio.create_task(
                    send_alert(
                        "Budget Exceeded",
                        f"Daily budget limit reached! Spent: ${new_total:.4f} / ${self.limits['daily']}",
                        "critical",
                    )
                )
                return False, "daily_limit_exceeded"

            return True, "within_budget"

        except Exception as e:
            logger.error(f"Redis budget check failed: {e}")
            raise

    def get_budget_status(self) -> Dict:
        daily_spent = self.tracker.get_statistics(days=1)["total_cost"]
        weekly_spent = self.tracker.get_statistics(days=7)["total_cost"]
        monthly_spent = self.tracker.get_statistics(days=30)["total_cost"]

        def status(spent, limit):
            return {
                "spent": round(spent, 4),
                "limit": limit,
                "remaining": round(limit - spent, 4),
                "percentage": round((spent / limit * 100) if limit > 0 else 0, 2),
                "alert": spent > (limit * self.alert_threshold),
            }

        return {
            "daily": status(daily_spent, self.limits["daily"]),
            "weekly": status(weekly_spent, self.limits["weekly"]),
            "monthly": status(monthly_spent, self.limits["monthly"]),
            "alert_threshold": self.alert_threshold,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def estimate_query_cost(
        self,
        model_id: str,
        query_length: int,
        model_config_path: Path = _PROJECT_ROOT / "config" / "models.yaml",
    ) -> float:
        try:
            import yaml  # type: ignore

            with open(model_config_path, "r") as f:
                config = yaml.safe_load(f)
            if model_id in config.get("openrouter_models", {}):
                cfg = config["openrouter_models"][model_id]
                estimated_input = query_length // 4
                estimated_output = 500
                return float(
                    (estimated_input / 1000) * cfg.get("cost_per_1k_input", 0.001)
                    + (estimated_output / 1000) * cfg.get("cost_per_1k_output", 0.002)
                )
        except Exception:
            pass
        return 0.05
