import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))
from src.routing.features import FeatureExtractor
from src.routing.router import QueryRouter


@pytest.fixture
def router():
    """Create router with real config."""
    classifier_path = Path(__file__).parent.parent / "models/classifiers/complexity_classifier.pkl"
    return QueryRouter(
        routing_config_path=Path("config/routing.yaml"),
        classifier_path=classifier_path if classifier_path.exists() else None,
    )


@pytest.mark.asyncio
async def test_router_routes_simple_query(router):
    """Test simple query gets routed to small model."""
    decision = await router.route("What is AI?")

    assert decision["complexity"] == "simple"
    assert decision["model_id"] == "nvidia/nemotron-nano-9b-v2:free"
    assert 0 <= decision["confidence"] <= 1


@pytest.mark.asyncio
async def test_router_routes_complex_query(router):
    """Test complex query gets routed to large model."""
    decision = await router.route(
        "Analyze the ethical implications of AI in healthcare, "
        "evaluate regulatory approaches, and synthesize recommendations."
    )

    assert decision["complexity"] == "complex"
    assert decision["model_id"] == "google/gemma-4-31b-it:free"


@pytest.mark.asyncio
async def test_router_strategy_changes_model(router):
    """Test different strategies route to different models."""
    query = "What is AI?"

    cost_decision = await router.route(query, strategy="cost_optimized")
    quality_decision = await router.route(query, strategy="quality_first")

    # Quality-first should use bigger model even for simple queries
    assert quality_decision["model_id"] == "google/gemma-4-31b-it:free"
    assert cost_decision["model_id"] != "google/gemma-4-31b-it:free"


@pytest.mark.asyncio
async def test_router_returns_required_fields(router):
    """Test routing decision has all required fields."""
    decision = await router.route("Test query")

    required = ["model_id", "complexity", "confidence", "fallback_model", "strategy"]
    for key in required:
        assert key in decision


@pytest.mark.asyncio
async def test_feature_extractor_upgraded_features():
    """Test the upgraded FeatureExtractor doesn't crash and returns the new features."""
    extractor = FeatureExtractor()
    query = "If I want to build an API, then how do I route =>?"
    features = await extractor.extract(query)

    # Check new linguistic features
    assert "logic_operator_count" in features
    assert "symbol_density" in features

    # Check new semantic features
    assert "medium_similarity" in features

    # Validate feature vector shape is now 16 (12 lexical + 4 semantic)
    vector = extractor.extract_vector(features)
    assert vector.shape == (16,)

    # Validate batch extraction
    batch_features = await extractor.batch_extract_vectors([query, "another query"])
    assert batch_features.shape == (2, 16)
