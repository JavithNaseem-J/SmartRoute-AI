import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))
from src.cost.tracker import CostTracker


@pytest.fixture
def tracker():
    """Create CostTracker which will automatically use the dummy sqlite:///:memory: from conftest."""
    t = CostTracker()
    yield t
    t.close()


def test_tracker_logs_query(tracker):
    """Test logging a query."""
    tracker.log_query(
        query="What is AI?",
        model_id="llama_3_1_8b",
        complexity="simple",
        strategy="cost_optimized",
        input_tokens=10,
        output_tokens=50,
        cost=0.0,
        latency=0.5,
        success=True,
    )

    stats = tracker.get_statistics(days=1)
    assert stats["total_queries"] == 1


def test_tracker_statistics(tracker):
    """Test statistics calculation."""
    # Log 3 queries
    for model in ["llama_3_1_8b", "llama_3_1_8b", "llama_3_3_70b"]:
        tracker.log_query(
            query="Test",
            model_id=model,
            complexity="simple",
            strategy="cost_optimized",
            input_tokens=10,
            output_tokens=50,
            cost=0.01,
            latency=1.0,
            success=True,
        )

    stats = tracker.get_statistics(days=1)

    assert stats["total_queries"] == 3
    assert stats["total_cost"] == 0.03
    assert stats["by_model"]["llama_3_1_8b"]["count"] == 2


def test_tracker_savings(tracker):
    """Test savings calculation."""
    for _ in range(5):
        tracker.log_query(
            query="Test",
            model_id="llama_3_1_8b",
            complexity="simple",
            strategy="cost_optimized",
            input_tokens=10,
            output_tokens=50,
            cost=0.0,  # Free with Groq
            latency=1.0,
            success=True,
        )

    savings = tracker.calculate_savings(days=1, baseline_cost_per_query=0.10)

    assert savings["baseline_cost"] == 0.50
    assert savings["actual_cost"] == 0.0
    assert savings["percentage"] == 100


def test_tracker_statistics_are_scoped_by_user(tracker):
    for user_id in ("user-a", "user-b"):
        tracker.log_query(
            query="Tenant query",
            model_id="economy",
            complexity="simple",
            strategy="cost_optimized",
            input_tokens=10,
            output_tokens=10,
            cost=0.01,
            latency=0.2,
            user_id=user_id,
        )

    user_a = tracker.get_statistics(days=1, user_id="user-a")
    user_b = tracker.get_statistics(days=1, user_id="user-b")
    unknown = tracker.get_statistics(days=1, user_id="user-c")

    assert user_a["total_queries"] == 1
    assert user_b["total_queries"] == 1
    assert unknown["total_queries"] == 0
