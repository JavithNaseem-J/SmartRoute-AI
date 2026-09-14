from unittest.mock import AsyncMock

import pytest

from src.retrieval.indexer import DocumentIndexer


@pytest.mark.asyncio
async def test_clear_all_documents_is_user_scoped(mock_qdrant, mock_redis, tmp_path):
    """Per-user clear must not delete the shared Qdrant collection or flush Redis."""
    indexer = DocumentIndexer()
    mock_qdrant.collection_exists = AsyncMock(return_value=True)
    mock_qdrant.delete = AsyncMock(return_value=None)
    mock_qdrant.delete_collection = AsyncMock(return_value=None)
    mock_redis.flushdb = AsyncMock(return_value=None)

    await mock_redis.set("semantic_cache:user-1:point", "cached")
    await mock_redis.set("semantic_cache:user-2:point", "cached")

    await indexer.aclear_all_documents(tmp_path, user_id="user-1")

    mock_qdrant.delete_collection.assert_not_called()
    mock_redis.flushdb.assert_not_called()
    assert mock_qdrant.delete.call_count == 2
    docs_filter = mock_qdrant.delete.call_args_list[0].kwargs["points_selector"]
    assert docs_filter.must[0].key == "metadata.user_id"
    assert docs_filter.must[0].match.value == "user-1"
    assert await mock_redis.get("semantic_cache:user-1:point") is None
    assert await mock_redis.get("semantic_cache:user-2:point") == "cached"
