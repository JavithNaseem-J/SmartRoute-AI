import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "service" in data


def test_query_endpoint():
    """Test query endpoint"""
    response = client.post(
        "/query",
        json={
            "query": "What is machine learning?",
            "strategy": "cost_optimized",
            "use_retrieval": False
        }
    )
    # Might be 503 if models not loaded
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "answer" in data
        assert "cost" in data


def test_budget_endpoint():
    """Test budget endpoint"""
    response = client.get("/budget")
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "daily" in data
        assert "weekly" in data
        assert "monthly" in data


def test_stats_endpoint():
    """Test stats endpoint"""
    response = client.get("/stats?days=1")
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "total_queries" in data


def test_savings_endpoint():
    """Test savings calculation"""
    response = client.get("/savings?days=1")
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "savings" in data
        assert "percentage" in data


def test_models_endpoint():
    """Test models list"""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "local" in data
    assert "api" in data


def test_invalid_query():
    """Test validation"""
    response = client.post(
        "/query",
        json={
            "query": "",
            "strategy": "cost_optimized"
        }
    )
    assert response.status_code == 422