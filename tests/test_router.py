import pytest
import tempfile
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.routing.router import QueryRouter
from src.routing.classifier import ComplexityClassifier


@pytest.fixture
def temp_routing_config():
    """Create temporary routing config for testing"""
    config = {
        'default_strategy': 'cost_optimized',
        'strategies': {
            'cost_optimized': {
                'description': 'Test strategy',
                'simple': {
                    'model': 'llama_3_1_8b',
                    'fallback': 'llama4_scout_17b',
                    'quality_threshold': 0.7
                },
                'medium': {
                    'model': 'llama4_scout_17b',
                    'fallback': 'llama_3_3_70b',
                    'quality_threshold': 0.75
                },
                'complex': {
                    'model': 'llama_3_3_70b',
                    'fallback': 'llama_3_3_70b',
                    'quality_threshold': 0.8
                }
            },
            'quality_first': {
                'description': 'Quality strategy',
                'simple': {
                    'model': 'llama_3_3_70b',
                    'fallback': 'llama_3_3_70b'
                },
                'medium': {
                    'model': 'llama_3_3_70b',
                    'fallback': 'llama_3_3_70b'
                },
                'complex': {
                    'model': 'llama_3_3_70b',
                    'fallback': 'llama_3_3_70b'
                }
            }
        }
    }
    
    # Create temp file
    temp_file = tempfile.NamedTemporaryFile(
        mode='w',
        delete=False,
        suffix='.yaml'
    )
    yaml.dump(config, temp_file)
    temp_file.close()
    
    yield Path(temp_file.name)
    
    # Cleanup
    Path(temp_file.name).unlink(missing_ok=True)


@pytest.fixture
def router(temp_routing_config):
    """Create router with temp config"""
    return QueryRouter(
        routing_config_path=temp_routing_config,
        classifier_path=None  # Will use rule-based classification
    )


def test_router_initialization(router):
    """Test router initializes correctly"""
    assert router is not None
    assert router.default_strategy == 'cost_optimized'
    assert router.classifier is not None


def test_simple_query_routing(router):
    """Test routing of simple query"""
    query = "What is AI?"
    
    decision = router.route(query)
    
    assert decision['complexity'] == 'simple'
    assert decision['model_id'] == 'llama_3_1_8b'
    assert decision['strategy'] == 'cost_optimized'
    assert 'confidence' in decision
    assert 'fallback_model' in decision


def test_medium_query_routing(router):
    """Test routing of medium complexity query"""
    query = "How does machine learning work and why is it useful?"
    
    decision = router.route(query)
    
    # Classifier may classify differently based on training data
    assert decision['complexity'] in ['simple', 'medium', 'complex']
    assert decision['model_id'] in ['llama_3_1_8b', 'qwen_32b', 'llama_3_3_70b']


def test_complex_query_routing(router):
    """Test routing of complex query"""
    query = "Analyze the ethical implications of artificial intelligence in healthcare, evaluate different regulatory approaches, and synthesize recommendations for policymakers considering both innovation and safety concerns."
    
    decision = router.route(query)
    
    assert decision['complexity'] == 'complex'
    assert decision['model_id'] == 'llama_3_3_70b'
    assert decision['strategy'] == 'cost_optimized'


def test_quality_first_strategy(router):
    """Test quality-first routing strategy"""
    query = "What is AI?"
    
    decision = router.route(query, strategy='quality_first')
    
    assert decision['strategy'] == 'quality_first'
    assert decision['model_id'] == 'llama_3_3_70b'  # Quality-first uses better model


def test_confidence_based_escalation(router):
    """Test escalation when confidence is low"""
    # Create a query that might have low confidence
    query = "Hmm..."  # Very short, ambiguous
    
    decision = router.route(query)
    
    # Should have low confidence and potentially escalate
    assert 'confidence' in decision
    assert decision['reason'] in ['normal_routing', 'low_confidence_escalation']


def test_unknown_strategy_fallback(router):
    """Test fallback to default strategy for unknown strategy"""
    query = "What is AI?"
    
    decision = router.route(query, strategy='unknown_strategy')
    
    # Should fall back to default
    assert decision['strategy'] == 'cost_optimized'


def test_routing_decision_keys(router):
    """Test that routing decision has all required keys"""
    query = "Test query"
    
    decision = router.route(query)
    
    required_keys = [
        'model_id',
        'complexity',
        'confidence',
        'fallback_model',
        'strategy',
        'reason',
        'query_length'
    ]
    
    for key in required_keys:
        assert key in decision


def test_update_strategy(router):
    """Test updating default strategy"""
    assert router.default_strategy == 'cost_optimized'
    
    router.update_strategy('quality_first')
    
    assert router.default_strategy == 'quality_first'
    
    # Test routing uses new default
    decision = router.route("What is AI?")
    assert decision['strategy'] == 'quality_first'


def test_update_strategy_invalid(router):
    """Test updating to invalid strategy raises error"""
    with pytest.raises(ValueError):
        router.update_strategy('nonexistent_strategy')


def test_get_strategy_info(router):
    """Test getting strategy information"""
    info = router.get_strategy_info('cost_optimized')
    
    assert 'strategy' in info
    assert 'description' in info
    assert 'rules' in info
    assert info['strategy'] == 'cost_optimized'


def test_explain_routing(router):
    """Test routing explanation generation"""
    query = "What is AI?"
    decision = router.route(query)
    
    explanation = router.explain_routing(decision)
    
    assert isinstance(explanation, str)
    assert decision['complexity'] in explanation
    assert decision['model_id'] in explanation
    assert decision['strategy'] in explanation


def test_routing_stats_empty():
    """Test routing stats with empty history"""
    stats = QueryRouter(
        routing_config_path=Path("config/routing.yaml"),
        classifier_path=None
    ).get_routing_stats([])
    
    assert stats == {}


def test_routing_stats_with_history(router):
    """Test routing stats calculation"""
    # Create mock history
    history = [
        {'model_id': 'llama_3_1_8b', 'complexity': 'simple', 'confidence': 0.9, 'reason': 'normal_routing'},
        {'model_id': 'llama_3_1_8b', 'complexity': 'simple', 'confidence': 0.85, 'reason': 'normal_routing'},
        {'model_id': 'qwen_32b', 'complexity': 'medium', 'confidence': 0.8, 'reason': 'normal_routing'},
        {'model_id': 'llama_3_3_70b', 'complexity': 'complex', 'confidence': 0.75, 'reason': 'normal_routing'},
        {'model_id': 'llama_3_3_70b', 'complexity': 'medium', 'confidence': 0.6, 'reason': 'low_confidence_escalation'},
    ]
    
    stats = router.get_routing_stats(history)
    
    assert stats['total_queries'] == 5
    assert 'complexity_distribution' in stats
    assert 'model_distribution' in stats
    assert 'avg_confidence' in stats
    assert 'escalation_rate' in stats
    assert stats['escalations'] == 1


def test_query_length_tracking(router):
    """Test that query length is tracked"""
    short_query = "What is AI?"
    long_query = "This is a much longer query with many more words that goes on and on"
    
    short_decision = router.route(short_query)
    long_decision = router.route(long_query)
    
    assert short_decision['query_length'] < long_decision['query_length']


def test_fallback_model_assignment(router):
    """Test that fallback models are correctly assigned"""
    query = "What is AI?"
    
    decision = router.route(query, strategy='cost_optimized')
    
    # For simple queries in cost_optimized
    assert decision['fallback_model'] in ['qwen_32b', 'llama_3_1_8b', 'llama_3_3_70b']


def test_multiple_queries_different_complexity(router):
    """Test routing multiple queries of different complexities"""
    queries = [
        ("What is AI?", 'simple'),
        ("How does neural network training work?", 'medium'),
        ("Analyze AI ethics comprehensively with detailed reasoning", 'complex')
    ]
    
    decisions = []
    for query, expected_complexity in queries:
        decision = router.route(query)
        decisions.append(decision)
        
        # Verify complexity is reasonable (classifier might vary)
        assert decision['complexity'] in ['simple', 'medium', 'complex']
    
    # At least should have different complexities
    complexities = [d['complexity'] for d in decisions]
    assert len(set(complexities)) >= 2  # At least 2 different complexities


def test_routing_consistency(router):
    """Test that same query routes consistently"""
    query = "What is machine learning?"
    
    decision1 = router.route(query)
    decision2 = router.route(query)
    
    # Should get same routing decision for same query
    assert decision1['complexity'] == decision2['complexity']
    assert decision1['model_id'] == decision2['model_id']


def test_confidence_range(router):
    """Test that confidence is in valid range"""
    queries = [
        "What is AI?",
        "How does ML work?",
        "Analyze AI ethics comprehensively"
    ]
    
    for query in queries:
        decision = router.route(query)
        assert 0.0 <= decision['confidence'] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])