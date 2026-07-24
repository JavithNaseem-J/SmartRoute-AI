import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)


@pytest.fixture
def client():
    """Create test client with mocked pipeline."""
    from fastapi.testclient import TestClient

    import api.main as api_module

    _MOCK_RESULT = {
        "answer": "Test answer",
        "model_used": "llama_3_1_8b",
        "complexity": "simple",
        "confidence": 0.95,
        "cost": 0.0,
        "latency": 0.5,
        "sources": [],
        "success": True,
        "error": None,
    }

    # pipeline.run is now async — must be AsyncMock so `await pipeline.run(...)` works.
    mock_pipeline = MagicMock()
    mock_pipeline.run = AsyncMock(return_value=_MOCK_RESULT)
    mock_pipeline.tracker.get_statistics.return_value = {"total_queries": 10}
    mock_pipeline.budget_manager.get_budget_status.return_value = {"daily": {"spent": 0}}

    original = api_module.pipeline
    api_module.pipeline = mock_pipeline

    yield TestClient(api_module.app)

    api_module.pipeline = original


import jwt


@pytest.fixture
def api_key():
    jwt_secret = os.getenv(
        "SUPABASE_JWT_SECRET", "super-secret-jwt-token-with-at-least-32-characters-long"
    )
    return jwt.encode({"sub": "test_user"}, jwt_secret, algorithm="HS256")


def test_health_check(client):
    """Test health endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_query_requires_auth(client):
    """Test query endpoint requires API key/JWT token."""
    response = client.post("/v1/query", json={"query": "What is AI?"})
    assert response.status_code == 403 or response.status_code == 401


def test_query_with_auth(client, api_key):
    """Test query endpoint with valid JWT."""
    response = client.post(
        "/v1/query", json={"query": "What is AI?"}, headers={"Authorization": f"Bearer {api_key}"}
    )
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_stats_endpoint(client, api_key):
    """Test stats endpoint."""
    response = client.get("/v1/stats", headers={"Authorization": f"Bearer {api_key}"})
    assert response.status_code == 200
