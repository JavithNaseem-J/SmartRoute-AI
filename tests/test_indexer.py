from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from langchain_core.documents import Document
from qdrant_client import models

import src.core.dependencies as deps
from src.retrieval.indexer import DocumentIndexer, NoIndexableTextError


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


@pytest.mark.asyncio
async def test_indexer_reports_embedding_generation_failure(mock_qdrant):
    """Embedding provider failures should produce a clear upload error."""
    indexer = DocumentIndexer()
    indexer.embeddings.aembed_documents.side_effect = RuntimeError("hf unavailable")

    with pytest.raises(RuntimeError, match="Embedding generation failed"):
        await indexer.aindex_documents([Document(page_content="cover letter text")])

    mock_qdrant.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_indexer_rejects_blank_documents_before_embedding(mock_qdrant):
    """Blank document pages must not be sent to FastEmbed as an empty batch."""
    indexer = DocumentIndexer()

    with pytest.raises(NoIndexableTextError, match="No readable text"):
        await indexer.aindex_documents([Document(page_content="   ")])

    indexer.embeddings.aembed_documents.assert_not_called()
    mock_qdrant.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_indexer_reports_empty_fastembed_vector_response(mock_qdrant):
    """An empty provider response must identify FastEmbed and the affected chunk count."""
    indexer = DocumentIndexer()
    indexer.embeddings.aembed_documents.side_effect = None
    indexer.embeddings.aembed_documents.return_value = []

    with pytest.raises(RuntimeError, match="FastEmbed returned no vectors for 1 text chunks"):
        await indexer.aindex_documents([Document(page_content="cover letter text")])

    mock_qdrant.upsert.assert_not_called()


def test_load_file_ignores_blank_text(tmp_path, mock_qdrant):
    """Text loaders should reject files whose extracted content is blank."""
    file_path = tmp_path / "blank.txt"
    file_path.write_text("  \n\t", encoding="utf-8")

    assert DocumentIndexer().load_file(file_path, source="user-1/blank.txt") == []


@pytest.mark.asyncio
async def test_indexer_reports_fastembed_generation_failure(mock_qdrant):
    """Local FastEmbed failures should identify the local embedding runtime."""
    indexer = DocumentIndexer()
    indexer.embeddings.aembed_documents.side_effect = RuntimeError("model download failed")

    with pytest.raises(RuntimeError, match="FastEmbed model download"):
        await indexer.aindex_documents([Document(page_content="cover letter text")])

    mock_qdrant.upsert.assert_not_called()


def test_fastembed_embeddings_do_not_require_hf_token(monkeypatch):
    """Dense embeddings should initialize without Hugging Face credentials."""
    previous_embeddings = deps._embeddings
    monkeypatch.delenv("HF_TOKEN", raising=False)
    deps._embeddings = None

    try:
        embeddings = deps.get_embeddings()
        assert isinstance(embeddings, deps.FastEmbedEmbeddings)
    finally:
        deps._embeddings = previous_embeddings


@pytest.mark.asyncio
async def test_fastembed_adapter_generates_document_and_query_vectors(monkeypatch):
    """The local adapter must use FastEmbed's passage and query encoders."""
    embeddings = deps.FastEmbedEmbeddings()
    model = MagicMock()
    model.passage_embed.return_value = iter([np.array([0.1, 0.2])])
    model.query_embed.return_value = iter([np.array([0.3, 0.4])])
    monkeypatch.setattr(embeddings, "_get_model", lambda: model)

    assert await embeddings.aembed_documents(["cover letter"]) == [[0.1, 0.2]]
    assert await embeddings.aembed_query("what is this about?") == [0.3, 0.4]


@pytest.mark.asyncio
async def test_indexer_reports_qdrant_upsert_failure(mock_qdrant):
    """Qdrant write failures should be distinct from embedding failures."""
    indexer = DocumentIndexer()
    indexer.embeddings.aembed_documents.return_value = [[0.1] * 384]
    mock_qdrant.collection_exists.return_value = True
    mock_qdrant.upsert.side_effect = RuntimeError("schema mismatch")

    with pytest.raises(RuntimeError, match="Vector database upsert failed"):
        await indexer.aindex_documents([Document(page_content="cover letter text")])
