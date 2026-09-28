from unittest.mock import AsyncMock, MagicMock

import pytest

from src.cost.budget import BudgetExceededError, BudgetManager, BudgetUnavailableError


def make_manager(redis):
    manager = BudgetManager.__new__(BudgetManager)
    manager._redis = redis
    manager._enforcement_status = "unknown"
    manager.limits = {"daily": 1.0, "weekly": 5.0, "monthly": 10.0}
    manager.alert_threshold = 0.8
    manager.tracker = MagicMock()
    manager.model_config = {}
    return manager


@pytest.mark.asyncio
async def test_budget_counter_is_tenant_scoped():
    redis = MagicMock()
    redis.incrbyfloat = AsyncMock(return_value=0.25)
    redis.expire = AsyncMock()
    manager = make_manager(redis)

    assert await manager.check_budget(0.25, user_id="tenant-a") == (
        True,
        "within_budget",
    )

    key = redis.incrbyfloat.call_args.args[0]
    assert key.startswith("smartroute:budget:tenant-a:daily:")
    assert manager._enforcement_status == "active"


@pytest.mark.asyncio
async def test_budget_fails_closed_when_redis_is_unavailable():
    manager = make_manager(None)

    with pytest.raises(BudgetUnavailableError):
        await manager.check_budget(0.25, user_id="tenant-a")

    assert manager._enforcement_status == "unavailable"


@pytest.mark.asyncio
async def test_budget_rolls_back_and_rejects_over_limit():
    redis = MagicMock()
    redis.incrbyfloat = AsyncMock(side_effect=[1.25, 1.0])
    redis.expire = AsyncMock()
    manager = make_manager(redis)

    with pytest.raises(BudgetExceededError):
        await manager.check_budget(0.25, user_id="tenant-a")

    assert redis.incrbyfloat.await_count == 2
