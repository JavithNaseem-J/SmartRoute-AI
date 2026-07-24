"""
Tests for SemanticCache.

Qdrant, Redis, and embeddings are mocked globally in conftest.py (autouse).
The mock_semantic_cache fixture overrides search/upsert/get/setex per test.
"""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from qdrant_client.models import ScoredPoint

sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.semantic_cache import SemanticCache


@pytest.fixture
def cache(mock_qdrant, mock_redis, mock_embeddings):
    """Return a SemanticCache wired to the conftest mocks."""
    # SemanticCache calls get_qdrant_client(), get_redis_client(), get_embeddings()
    # at __init__ — all three are already patched by conftest autouse fixtures.
    return SemanticCache(threshold=0.95)


async def test_semantic_cache_miss(cache, mock_qdrant):
    """Test when no similar vector is found in Qdrant."""
    mock_qdrant.search = AsyncMock(return_value=[])
    result = await cache.get("What is SmartRoute?")
    assert result is None
    mock_qdrant.search.assert_called_once()


async def test_semantic_cache_hit(cache, mock_qdrant, mock_redis):
    """Test when a highly similar query exists in the vector store."""
    mock_point = ScoredPoint(
        id="mock-uuid",
        version=1,
        score=0.99,
        payload={"query": "What is SmartRoute AI?"},
        vector=None,
    )
    mock_qdrant.search = AsyncMock(return_value=[mock_point])

    expected_payload = {"answer": "It is an enterprise AI routing system."}
    mock_redis.get = AsyncMock(return_value=json.dumps(expected_payload))

    result = await cache.get("What is SmartRoute?")

    assert result == expected_payload
    mock_qdrant.search.assert_called_once()
    mock_redis.get.assert_called_once_with("semantic_cache:mock-uuid")


async def test_semantic_cache_set(cache, mock_qdrant, mock_redis):
    """Test setting a new value in the cache writes to both Redis and Qdrant."""
    mock_redis.setex = AsyncMock(return_value=None)
    mock_qdrant.upsert = AsyncMock(return_value=None)

    payload = {"answer": "It is an enterprise AI routing system."}
    await cache.set("What is SmartRoute?", payload)

    mock_redis.setex.assert_called_once()
    mock_qdrant.upsert.assert_called_once()
