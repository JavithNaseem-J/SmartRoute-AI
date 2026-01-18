"""Simple tests for the API."""
import pytest
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)


@pytest.fixture
def client():
    """Create test client with mocked pipeline."""
    from fastapi.testclient import TestClient
    import api.main as api_module
    
    # Mock the pipeline
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = {
        "answer": "Test answer",
        "model_used": "llama_3_1_8b",
        "complexity": "simple",
        "confidence": 0.95,
        "cost": 0.0,
        "latency": 0.5,
        "sources": [],
        "success": True
    }
    mock_pipeline.tracker.get_statistics.return_value = {"total_queries": 10}
    mock_pipeline.budget_manager.get_budget_status.return_value = {"daily": {"spent": 0}}
    
    original = api_module.pipeline
    api_module.pipeline = mock_pipeline
    
    yield TestClient(api_module.app)
    
    api_module.pipeline = original


@pytest.fixture
def api_key():
    return os.getenv("SMARTROUTE_API_KEY", "dev-key-change-in-production")


def test_health_check(client):
    """Test health endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_query_requires_auth(client):
    """Test query endpoint requires API key."""
    response = client.post("/query", json={"query": "What is AI?"})
    assert response.status_code == 401


def test_query_with_auth(client, api_key):
    """Test query endpoint with valid API key."""
    response = client.post(
        "/query",
        json={"query": "What is AI?"},
        headers={"X-API-Key": api_key}
    )
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_stats_endpoint(client, api_key):
    """Test stats endpoint."""
    response = client.get("/stats", headers={"X-API-Key": api_key})
    assert response.status_code == 200
