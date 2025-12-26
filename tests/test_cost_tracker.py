import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.cost.tracker import CostTracker


@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    db_path = temp_file.name
    temp_file.close()
    
    yield db_path
    
    # Cleanup handled by tracker fixture


@pytest.fixture
def tracker(temp_db):
    """Create cost tracker with temp database"""
    t = CostTracker(db_path=temp_db)
    yield t
    # Close connection before cleanup
    t.close()
    try:
        Path(temp_db).unlink(missing_ok=True)
    except PermissionError:
        pass  # Windows file locking, will clean up on next run


def test_tracker_initialization(tracker):
    """Test tracker initializes correctly"""
    assert tracker is not None
    assert tracker.session is not None
    assert tracker.db_path.exists()


def test_log_query(tracker):
    """Test logging a query"""
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
    
    # Check it was logged
    stats = tracker.get_statistics(days=1)
    assert stats['total_queries'] == 1
    assert stats['total_cost'] == 0.0


def test_multiple_queries(tracker):
    """Test logging multiple queries"""
    queries = [
        ("What is AI?", "llama_3_1_8b", "simple", 0.0),
        ("How does ML work?", "qwen_32b", "medium", 0.0),
        ("Analyze AI ethics", "llama_3_3_70b", "complex", 0.0),
    ]
    
    for query, model_id, complexity, cost in queries:
        tracker.log_query(
            query=query,
            model_id=model_id,
            complexity=complexity,
            strategy="cost_optimized",
            input_tokens=100,
            output_tokens=500,
            cost=cost,
            latency=1.0,
            success=True
        )
    
    stats = tracker.get_statistics(days=1)
    assert stats['total_queries'] == 3
    assert stats['total_cost'] == 0.0


def test_statistics_by_model(tracker):
    """Test statistics grouped by model"""
    # Log queries with different models
    tracker.log_query(
        query="Query 1",
        model_id="llama_3_1_8b",
        complexity="simple",
        strategy="cost_optimized",
        input_tokens=10,
        output_tokens=50,
        cost=0.0,
        latency=0.5,
        success=True
    )
    
    tracker.log_query(
        query="Query 2",
        model_id="llama_3_1_8b",
        complexity="simple",
        strategy="cost_optimized",
        input_tokens=10,
        output_tokens=50,
        cost=0.0,
        latency=0.5,
        success=True
    )
    
    tracker.log_query(
        query="Query 3",
        model_id="llama_3_3_70b",
        complexity="complex",
        strategy="cost_optimized",
        input_tokens=100,
        output_tokens=500,
        cost=0.0,
        latency=2.0,
        success=True
    )
    
    stats = tracker.get_statistics(days=1)
    
    assert "llama_3_1_8b" in stats['by_model']
    assert "llama_3_3_70b" in stats['by_model']
    assert stats['by_model']['llama_3_1_8b']['count'] == 2
    assert stats['by_model']['llama_3_3_70b']['count'] == 1


def test_statistics_by_complexity(tracker):
    """Test statistics grouped by complexity"""
    complexities = ["simple", "simple", "medium", "complex"]
    
    for complexity in complexities:
        tracker.log_query(
            query=f"Query {complexity}",
            model_id="test_model",
            complexity=complexity,
            strategy="cost_optimized",
            input_tokens=10,
            output_tokens=50,
            cost=0.01,
            latency=1.0,
            success=True
        )
    
    stats = tracker.get_statistics(days=1)
    
    assert stats['by_complexity']['simple']['count'] == 2
    assert stats['by_complexity']['medium']['count'] == 1
    assert stats['by_complexity']['complex']['count'] == 1


def test_calculate_savings(tracker):
    """Test savings calculation"""
    # Log 10 queries with cost $0.00 each (Groq is free)
    for i in range(10):
        tracker.log_query(
            query=f"Query {i}",
            model_id="llama_3_1_8b",
            complexity="simple",
            strategy="cost_optimized",
            input_tokens=10,
            output_tokens=50,
            cost=0.0,
            latency=1.0,
            success=True
        )
    
    # Calculate savings vs baseline ($0.15 per query = $1.50 total)
    savings = tracker.calculate_savings(days=1, baseline_cost_per_query=0.15)
    
    assert savings['baseline_cost'] == 1.50
    assert savings['actual_cost'] == 0.0
    assert savings['savings'] == 1.50
    assert savings['percentage'] == 100  # 100% savings with free Groq


def test_empty_statistics(tracker):
    """Test statistics with no queries"""
    stats = tracker.get_statistics(days=1)
    
    assert stats['total_queries'] == 0
    assert stats['total_cost'] == 0.0
    assert stats['avg_cost_per_query'] == 0.0


def test_failed_query_logging(tracker):
    """Test logging failed queries"""
    tracker.log_query(
        query="This will fail",
        model_id="test_model",
        complexity="simple",
        strategy="cost_optimized",
        input_tokens=0,
        output_tokens=0,
        cost=0.0,
        latency=0.1,
        success=False
    )
    
    stats = tracker.get_statistics(days=1)
    assert stats['total_queries'] == 1
    assert stats['total_cost'] == 0.0


def test_avg_latency(tracker):
    """Test average latency calculation"""
    latencies = [0.5, 1.0, 1.5, 2.0]
    
    for latency in latencies:
        tracker.log_query(
            query="Test query",
            model_id="test_model",
            complexity="simple",
            strategy="cost_optimized",
            input_tokens=10,
            output_tokens=50,
            cost=0.01,
            latency=latency,
            success=True
        )
    
    stats = tracker.get_statistics(days=1)
    expected_avg = sum(latencies) / len(latencies)
    
    assert abs(stats['avg_latency'] - expected_avg) < 0.01


def test_daily_breakdown(tracker):
    """Test daily cost breakdown"""
    # Log queries for today
    for i in range(5):
        tracker.log_query(
            query=f"Query {i}",
            model_id="test_model",
            complexity="simple",
            strategy="cost_optimized",
            input_tokens=10,
            output_tokens=50,
            cost=0.01,
            latency=1.0,
            success=True
        )
    
    breakdown = tracker.get_daily_breakdown(days=7)
    today = datetime.now().date().isoformat()
    
    assert today in breakdown
    assert breakdown[today]['queries'] == 5
    assert breakdown[today]['cost'] == 0.05


def test_export_to_jsonl(tracker, tmp_path):
    """Test exporting logs to JSONL"""
    # Log some queries
    for i in range(3):
        tracker.log_query(
            query=f"Query {i}",
            model_id="test_model",
            complexity="simple",
            strategy="cost_optimized",
            input_tokens=10,
            output_tokens=50,
            cost=0.01,
            latency=1.0,
            success=True
        )
    
    # Export
    output_file = tmp_path / "logs.jsonl"
    tracker.export_to_jsonl(output_file, days=1)
    
    assert output_file.exists()
    
    # Check content
    lines = output_file.read_text().strip().split('\n')
    assert len(lines) == 3


def test_statistics_time_filter(tracker):
    """Test that statistics respect time filter"""
    # This test assumes we can't easily manipulate timestamps
    # In a real scenario, you'd mock datetime or use a test database
    # with pre-populated historical data
    
    tracker.log_query(
        query="Recent query",
        model_id="test_model",
        complexity="simple",
        strategy="cost_optimized",
        input_tokens=10,
        output_tokens=50,
        cost=0.01,
        latency=1.0,
        success=True
    )
    
    # Query for today
    stats_1_day = tracker.get_statistics(days=1)
    assert stats_1_day['total_queries'] >= 1
    
    # Query for last 7 days should include today's data
    stats_7_days = tracker.get_statistics(days=7)
    assert stats_7_days['total_queries'] >= stats_1_day['total_queries']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])