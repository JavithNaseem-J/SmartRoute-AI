import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import ScoredPoint
from qdrant_client import models

sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.semantic_cache import SemanticCache


@pytest.fixture
def cache(mock_qdrant, mock_redis, mock_embeddings):
    """Return a SemanticCache wired to the conftest mocks."""
    return SemanticCache(threshold=0.95)


async def test_semantic_cache_miss(cache, mock_qdrant):
    """Test when no similar vector is found in Qdrant."""
    mock_res = AsyncMock()
    mock_res.points = []
    mock_qdrant.query_points = AsyncMock(return_value=mock_res)
    result = await cache.get("What is SmartRoute?", user_id="user-1")
    assert result is None
    mock_qdrant.create_collection.assert_awaited_once()
    vector_config = mock_qdrant.create_collection.call_args.kwargs["vectors_config"]
    assert vector_config.size == 384
    assert vector_config.distance == models.Distance.COSINE
    assert {
        call.kwargs["field_name"] for call in mock_qdrant.create_payload_index.await_args_list
    } == {
        "user_id",
        "cache_scope",
    }
    mock_qdrant.query_points.assert_called_once()


async def test_semantic_cache_hit(cache, mock_qdrant, mock_redis):
    """Test when a highly similar query exists in the vector store."""
    mock_point = ScoredPoint(
        id="mock-uuid",
        version=1,
        score=0.99,
        payload={"query": "What is SmartRoute AI?"},
        vector=None,
    )
    mock_res = AsyncMock()
    mock_res.points = [mock_point]
    mock_qdrant.query_points = AsyncMock(return_value=mock_res)

    expected_payload = {"answer": "It is an enterprise AI routing system."}
    mock_redis.get = AsyncMock(return_value=json.dumps(expected_payload))

    result = await cache.get("What is SmartRoute?", user_id="user-1")

    assert result == expected_payload
    mock_qdrant.query_points.assert_called_once()
    call_kwargs = mock_qdrant.query_points.call_args.kwargs
    assert call_kwargs["query_filter"].must[0].key == "user_id"
    assert call_kwargs["query_filter"].must[0].match.value == "user-1"
    mock_redis.get.assert_called_once_with("semantic_cache:user-1:mock-uuid")


async def test_semantic_cache_set(cache, mock_qdrant, mock_redis):
    """Test setting a new value in the cache writes to both Redis and Qdrant."""
    mock_redis.setex = AsyncMock(return_value=None)
    mock_qdrant.upsert = AsyncMock(return_value=None)

    payload = {"answer": "It is an enterprise AI routing system."}
    await cache.set("What is SmartRoute?", payload, user_id="user-1")

    mock_redis.setex.assert_called_once()
    mock_qdrant.upsert.assert_called_once()
    point = mock_qdrant.upsert.call_args.kwargs["points"][0]
    assert point.payload["user_id"] == "user-1"


async def test_existing_cache_collection_is_reused(cache, mock_qdrant):
    mock_qdrant.collection_exists.return_value = True
    mock_qdrant.query_points = AsyncMock(return_value=AsyncMock(points=[]))

    await cache.get("first", user_id="user-1")
    await cache.get("second", user_id="user-1")

    mock_qdrant.create_collection.assert_not_awaited()
    assert mock_qdrant.create_payload_index.await_count == 2
    assert mock_qdrant.query_points.await_count == 2


async def test_concurrent_cache_reads_create_collection_once(cache, mock_qdrant):
    mock_qdrant.query_points = AsyncMock(return_value=AsyncMock(points=[]))

    await asyncio.gather(
        cache.get("first", user_id="user-1"),
        cache.get("second", user_id="user-1"),
    )

    mock_qdrant.create_collection.assert_awaited_once()
    assert mock_qdrant.create_payload_index.await_count == 2
    assert mock_qdrant.query_points.await_count == 2


async def test_cache_setup_failure_does_not_write_partial_entry(cache, mock_qdrant, mock_redis):
    mock_qdrant.create_collection.side_effect = RuntimeError("Qdrant unavailable")

    await cache.set("question", {"answer": "answer"}, user_id="user-1")

    assert await mock_redis.keys("semantic_cache:*") == []
    mock_qdrant.upsert.assert_not_awaited()


async def test_cache_round_trip_with_local_qdrant(cache, mock_embeddings):
    client = AsyncQdrantClient(location=":memory:")
    cache.qdrant = client
    mock_embeddings.aembed_query.return_value = [1.0] + [0.0] * 383
    try:
        assert await cache.get("question", user_id="user-1") is None
        await cache.set("question", {"answer": "answer"}, user_id="user-1")
        assert await cache.get("question", user_id="user-1") == {"answer": "answer"}
    finally:
        await client.close()


async def test_semantic_cache_skips_anonymous_user(cache, mock_qdrant, mock_redis):
    """Anonymous calls must not use a shared global semantic cache."""
    assert await cache.get("What is SmartRoute?", user_id=None) is None
    await cache.set("What is SmartRoute?", {"answer": "x"}, user_id=None)

    mock_qdrant.query_points.assert_not_called()
    mock_qdrant.upsert.assert_not_called()
    assert await mock_redis.keys("*") == []
