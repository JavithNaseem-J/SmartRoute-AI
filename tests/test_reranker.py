from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document

from src.retrieval.reranker import DocumentReranker


@pytest.fixture(scope="module")
def reranker():
    return DocumentReranker()


def test_reranker_initialization(reranker):
    """Reranker either loads the model or fails gracefully."""
    assert isinstance(reranker.is_ready, bool)


@pytest.mark.asyncio
async def test_reranker_returns_top_k(reranker):
    """Regardless of readiness, exactly top_k documents are returned."""
    docs = [Document(page_content=f"Document {i}") for i in range(10)]

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_response = AsyncMock()
        mock_response.json.return_value = [0.5] * 10
        mock_post.return_value.__aenter__.return_value = mock_response

        result = await reranker.rerank("test query", docs, top_k=3)
        assert len(result) == 3


@pytest.mark.asyncio
async def test_reranker_handles_empty_list(reranker):
    """Empty document list returns empty list — no crash."""
    result = await reranker.rerank("query", [], top_k=5)
    assert result == []


@pytest.mark.asyncio
async def test_reranker_handles_fewer_docs_than_top_k(reranker):
    """If fewer documents than top_k exist, all documents are returned."""
    docs = [Document(page_content="Only doc")]

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_response = AsyncMock()
        mock_response.json.return_value = [0.9]
        mock_post.return_value.__aenter__.return_value = mock_response

        result = await reranker.rerank("query", docs, top_k=5)
        assert len(result) == 1


@pytest.mark.asyncio
async def test_reranker_semantic_ranking():
    """Semantically correct document is ranked first based on mock API scores."""
    reranker = DocumentReranker()
    query = "What is the capital of France?"

    doc_unrelated = Document(
        page_content="London is the capital of the UK.", metadata={"src": "unrelated"}
    )
    doc_correct = Document(
        page_content="Paris is the capital of France.", metadata={"src": "correct"}
    )
    doc_partial = Document(
        page_content="France is a country in Western Europe.", metadata={"src": "partial"}
    )

    docs = [doc_unrelated, doc_partial, doc_correct]

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_response = AsyncMock()
        # Mock scores matching the input docs order: unrelated(0.1), partial(0.5), correct(0.9)
        mock_response.json.return_value = [0.1, 0.5, 0.9]
        mock_post.return_value.__aenter__.return_value = mock_response

        result = await reranker.rerank(query, docs, top_k=2)

        assert (
            result[0].metadata["src"] == "correct"
        ), f"Expected 'correct' to be top-ranked, got '{result[0].metadata['src']}'"
