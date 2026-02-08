import pytest
import tempfile
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.cost.tracker import CostTracker


@pytest.fixture
def tracker():
    """Create tracker with temp database."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    t = CostTracker(db_path=temp_file.name)
    yield t
    t.session.close()
    t.engine.dispose()  
    try:
        Path(temp_file.name).unlink(missing_ok=True)
    except PermissionError:
        pass  


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
        success=True
    )
    
    stats = tracker.get_statistics(days=1)
    assert stats['total_queries'] == 1


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
            success=True
        )
    
    stats = tracker.get_statistics(days=1)
    
    assert stats['total_queries'] == 3
    assert stats['total_cost'] == 0.03
    assert stats['by_model']['llama_3_1_8b']['count'] == 2


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
            success=True
        )
    
    savings = tracker.calculate_savings(days=1, baseline_cost_per_query=0.10)
    
    assert savings['baseline_cost'] == 0.50
    assert savings['actual_cost'] == 0.0
    assert savings['percentage'] == 100 
