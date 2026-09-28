"""Tenant-scoped, fail-closed budget enforcement backed by Redis."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml  # type: ignore
from redis.asyncio import Redis

from src.core.dependencies import get_redis_client
from src.cost.tracker import CostTracker
from src.models.provider_config import load_provider_settings
from src.utils.alerting import send_alert
from src.utils.logger import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class BudgetUnavailableError(RuntimeError):
    """Budget state cannot be checked, so paid inference must not proceed."""


class BudgetExceededError(RuntimeError):
    """The tenant has reached the configured hard spending limit."""


class BudgetManager:
    """Enforce hard, tenant-scoped budgets with atomic Redis counters."""

    def __init__(
        self,
        tracker: CostTracker,
        config_path: Path = _PROJECT_ROOT / "config" / "routing.yaml",
    ):
        self.tracker = tracker
        self._redis: Optional[Redis[Any]] = None
        self._enforcement_status = "unavailable"

        try:
            self._redis = get_redis_client()
            self._enforcement_status = "unknown"
        except Exception as exc:
            logger.warning(f"BudgetManager: Redis unavailable ({exc}).")

        with open(config_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        budgets = config.get("budgets", {})
        self.limits = {
            "daily": budgets.get("daily", 10.0),
            "weekly": budgets.get("weekly", 50.0),
            "monthly": budgets.get("monthly", 200.0),
        }
        self.alert_threshold = budgets.get("alert_threshold", 0.8)

        try:
            settings = load_provider_settings(_PROJECT_ROOT / "config" / "models.yaml")
            self.model_config = settings.models
        except Exception as exc:
            logger.warning(f"Could not load models.yaml for cost estimation: {exc}")
            self.model_config = {}

        logger.info(
            f"BudgetManager: daily=${self.limits['daily']}, weekly=${self.limits['weekly']}"
        )

    def _redis_key(self, period: str, user_id: str) -> str:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return f"smartroute:budget:{user_id}:{period}:{today}"

    async def check_budget(
        self, estimated_cost: float, user_id: str | None = None
    ) -> Tuple[bool, str]:
        """Reserve estimated cost atomically and reject when enforcement is unavailable."""
        if self._redis is None:
            self._enforcement_status = "unavailable"
            raise BudgetUnavailableError("Redis budget enforcement is unavailable.")

        key = self._redis_key("daily", user_id or "anonymous")
        try:
            new_total = float(await self._redis.incrbyfloat(key, estimated_cost))
            await self._redis.expire(key, 86400)
            self._enforcement_status = "active"

            if new_total > self.limits["daily"]:
                await self._redis.incrbyfloat(key, -estimated_cost)
                logger.warning(f"Daily budget exceeded: ${new_total:.4f} / ${self.limits['daily']}")
                asyncio.create_task(
                    send_alert(
                        "Budget Exceeded",
                        f"Daily budget limit reached: ${new_total:.4f} / ${self.limits['daily']}",
                        "critical",
                    )
                )
                raise BudgetExceededError("Daily budget limit exceeded.")

            return True, "within_budget"
        except BudgetExceededError:
            raise
        except Exception as exc:
            self._enforcement_status = "unavailable"
            logger.error(f"Redis budget check failed: {exc}")
            raise BudgetUnavailableError("Redis budget enforcement is unavailable.") from exc

    async def check_health(self) -> bool:
        if self._redis is None:
            self._enforcement_status = "unavailable"
            return False
        try:
            healthy = bool(await self._redis.ping())
        except Exception:
            healthy = False
        self._enforcement_status = "active" if healthy else "unavailable"
        return healthy

    def get_budget_status(self, user_id: str | None = None) -> Dict:
        daily_spent = self.tracker.get_statistics(days=1, user_id=user_id)["total_cost"]
        weekly_spent = self.tracker.get_statistics(days=7, user_id=user_id)["total_cost"]
        monthly_spent = self.tracker.get_statistics(days=30, user_id=user_id)["total_cost"]

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
            "enforcement": self._enforcement_status,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def estimate_query_cost(self, model_id: str, query_length: int) -> float:
        if model_id in self.model_config:
            cfg = self.model_config[model_id]
            estimated_input = query_length // 4
            estimated_output = 1000
            return float(
                (estimated_input / 1000) * cfg.get("cost_per_1k_input", 0.001)
                + (estimated_output / 1000) * cfg.get("cost_per_1k_output", 0.002)
            )
        return 0.05
