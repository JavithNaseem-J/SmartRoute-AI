"""Simple tests for the routing module."""
import pytest
import tempfile
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.routing.router import QueryRouter


@pytest.fixture
def router():
    """Create router with real config."""
    classifier_path = Path(__file__).parent.parent / "models/classifiers/complexity_classifier.pkl"
    return QueryRouter(
        routing_config_path=Path("config/routing.yaml"),
        classifier_path=classifier_path if classifier_path.exists() else None
    )


def test_router_routes_simple_query(router):
    """Test simple query gets routed to small model."""
    decision = router.route("What is AI?")
    
    assert decision['complexity'] == 'simple'
    assert decision['model_id'] == 'llama_3_1_8b'
    assert 0 <= decision['confidence'] <= 1


def test_router_routes_complex_query(router):
    """Test complex query gets routed to large model."""
    decision = router.route(
        "Analyze the ethical implications of AI in healthcare, "
        "evaluate regulatory approaches, and synthesize recommendations."
    )
    
    assert decision['complexity'] == 'complex'
    assert decision['model_id'] == 'llama_3_3_70b'


def test_router_strategy_changes_model(router):
    """Test different strategies route to different models."""
    query = "What is AI?"
    
    cost_decision = router.route(query, strategy='cost_optimized')
    quality_decision = router.route(query, strategy='quality_first')
    
    # Quality-first should use bigger model even for simple queries
    assert quality_decision['model_id'] == 'llama_3_3_70b'


def test_router_returns_required_fields(router):
    """Test routing decision has all required fields."""
    decision = router.route("Test query")
    
    required = ['model_id', 'complexity', 'confidence', 'fallback_model', 'strategy']
    for key in required:
        assert key in decision
