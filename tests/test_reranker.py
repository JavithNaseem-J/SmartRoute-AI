"""
Tests for DocumentReranker.

Strategy: If sentence-transformers is installed, verify semantic ranking.
If not, verify graceful fallback (returns documents unchanged).
"""

import pytest
from langchain_core.documents import Document

from src.retrieval.reranker import DocumentReranker


@pytest.fixture(scope="module")
def reranker():
    return DocumentReranker(device="cpu")


def test_reranker_initialization(reranker):
    """Reranker either loads the model or fails gracefully."""
    # Either it's ready (model loaded) or not (sentence-transformers missing)
    # Both states are valid — is_ready tells us which
    assert isinstance(reranker.is_ready, bool)


def test_reranker_returns_top_k(reranker):
    """Regardless of readiness, exactly top_k documents are returned."""
    docs = [Document(page_content=f"Document {i}") for i in range(10)]
    result = reranker.rerank("test query", docs, top_k=3)
    assert len(result) == 3


def test_reranker_handles_empty_list(reranker):
    """Empty document list returns empty list — no crash."""
    result = reranker.rerank("query", [], top_k=5)
    assert result == []


def test_reranker_handles_fewer_docs_than_top_k(reranker):
    """If fewer documents than top_k exist, all documents are returned."""
    docs = [Document(page_content="Only doc")]
    result = reranker.rerank("query", docs, top_k=5)
    assert len(result) == 1


@pytest.mark.skipif(
    not DocumentReranker().is_ready,
    reason="sentence-transformers not installed; semantic test skipped",
)
def test_reranker_semantic_ranking():
    """Semantically correct document is ranked first."""
    reranker = DocumentReranker(device="cpu")
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

    # Pass them in wrong order intentionally
    docs = [doc_unrelated, doc_partial, doc_correct]
    result = reranker.rerank(query, docs, top_k=2)

    assert (
        result[0].metadata["src"] == "correct"
    ), f"Expected 'correct' to be top-ranked, got '{result[0].metadata['src']}'"
