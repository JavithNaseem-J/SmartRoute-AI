from unittest.mock import AsyncMock

import pytest
from langchain_core.documents import Document
from qdrant_client import models

from src.retrieval.indexer import DocumentIndexer


@pytest.mark.asyncio
async def test_clear_all_documents_is_user_scoped(mock_qdrant, mock_redis):
    """Per-user clear must not delete the shared Qdrant collection or flush Redis."""
    indexer = DocumentIndexer()
    mock_qdrant.collection_exists = AsyncMock(return_value=True)
    mock_qdrant.delete = AsyncMock(return_value=None)
    mock_qdrant.delete_collection = AsyncMock(return_value=None)
    mock_redis.flushdb = AsyncMock(return_value=None)

    await mock_redis.set("semantic_cache:user-1:point", "cached")
    await mock_redis.set("semantic_cache:user-2:point", "cached")

    await indexer.aclear_all_documents(user_id="user-1")

    mock_qdrant.delete_collection.assert_not_called()
    mock_redis.flushdb.assert_not_called()
    assert mock_qdrant.delete.call_count == 2
    docs_filter = mock_qdrant.delete.call_args_list[0].kwargs["points_selector"]
    assert docs_filter.must[0].key == "metadata.user_id"
    assert docs_filter.must[0].match.value == "user-1"
    assert await mock_redis.get("semantic_cache:user-1:point") is None
    assert await mock_redis.get("semantic_cache:user-2:point") == "cached"


@pytest.mark.asyncio
async def test_indexer_upsert_waits_for_vectors(mock_qdrant):
    """Uploads should wait until Qdrant has accepted points before returning."""
    indexer = DocumentIndexer()
    indexer.embeddings.aembed_documents.return_value = [[0.1] * 384]
    mock_qdrant.collection_exists.return_value = True

    await indexer.aindex_documents(
        [Document(page_content="cover letter text", metadata={"user_id": "user-1"})]
    )

    assert mock_qdrant.upsert.call_args.kwargs["wait"] is True


@pytest.mark.asyncio
async def test_count_indexed_chunks_filters_by_user_and_source(mock_qdrant):
    """Indexer verification counts the same user/source payload used during retrieval."""
    indexer = DocumentIndexer()
    mock_qdrant.collection_exists.return_value = True
    mock_qdrant.count.return_value = type("CountResult", (), {"count": 3})()

    count = await indexer.count_indexed_chunks(
        user_id="user-1",
        source="user-1/file.txt",
    )

    assert count == 3
    count_filter = mock_qdrant.count.call_args.kwargs["count_filter"]
    assert isinstance(count_filter, models.Filter)
    keys = [condition.key for condition in count_filter.must]
    assert keys == ["metadata.user_id", "metadata.source"]
