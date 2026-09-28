import pytest
from langchain_core.documents import Document

from src.retrieval.reranker import DocumentReranker


@pytest.mark.asyncio
async def test_local_reranker_scores_and_limits_documents(monkeypatch):
    monkeypatch.setenv("RERANKER_MODE", "local")
    reranker = DocumentReranker()
    docs = [
        Document(page_content="London is in the United Kingdom"),
        Document(page_content="Paris is the capital of France"),
        Document(page_content="France is in Europe"),
    ]

    result = await reranker.rerank("capital of France", docs, top_k=1)

    assert result == [docs[1]]
    assert reranker.last_mode == "local"


@pytest.mark.asyncio
async def test_local_reranker_handles_empty_and_short_lists(monkeypatch):
    monkeypatch.setenv("RERANKER_MODE", "local")
    reranker = DocumentReranker()
    doc = Document(page_content="Only document")

    assert await reranker.rerank("query", []) == []
    assert await reranker.rerank("query", [doc], top_k=5) == [doc]


@pytest.mark.asyncio
async def test_disabled_reranker_preserves_qdrant_order(monkeypatch):
    monkeypatch.setenv("RERANKER_MODE", "disabled")
    reranker = DocumentReranker()
    docs = [Document(page_content="first"), Document(page_content="second")]

    assert await reranker.rerank("second", docs, top_k=1) == [docs[0]]
    assert reranker.last_mode == "disabled"


def test_external_reranker_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("RERANKER_MODE", "huggingface")

    with pytest.raises(ValueError, match="local, disabled"):
        DocumentReranker()
