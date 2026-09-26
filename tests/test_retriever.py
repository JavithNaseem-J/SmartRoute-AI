from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from src.retrieval.reranker import DocumentReranker
from src.retrieval.chunking import DocumentChunker
from src.retrieval.retriever import DocumentRetriever


@pytest.mark.asyncio
async def test_document_retriever_requires_user_id(mock_qdrant):
    """Document retrieval must not use a shared global corpus without a user."""
    retriever = DocumentRetriever()

    context, sources = await retriever.retrieve("What is SmartRoute?")

    assert context == ""
    assert sources == []
    mock_qdrant.query_points.assert_not_called()


@pytest.mark.asyncio
async def test_document_retriever_filters_qdrant_by_user_id(mock_qdrant, monkeypatch):
    """Qdrant searches include metadata.user_id so tenants cannot see each other's chunks."""
    retriever = DocumentRetriever()
    retriever.dense_ready = True
    mock_qdrant.collection_exists = AsyncMock(return_value=True)
    mock_qdrant.count = AsyncMock(return_value=MagicMock(count=0))
    mock_qdrant.query_points = AsyncMock(return_value=MagicMock(points=[]))
    monkeypatch.setattr("src.retrieval.retriever.get_sparse_vector", lambda *_: None)

    await retriever.retrieve("What is SmartRoute?", user_id="user-1")

    query_filter = mock_qdrant.query_points.call_args.kwargs["query_filter"]
    assert query_filter.must[0].key == "metadata.user_id"
    assert query_filter.must[0].match.value == "user-1"


@pytest.mark.asyncio
async def test_document_retriever_diagnoses_user_with_no_chunks(mock_qdrant, monkeypatch):
    """No-source retrieval should reveal when the current user has no indexed chunks."""
    retriever = DocumentRetriever()
    retriever.dense_ready = True
    mock_qdrant.collection_exists = AsyncMock(return_value=True)
    mock_qdrant.count = AsyncMock(return_value=MagicMock(count=0))
    mock_qdrant.query_points = AsyncMock(return_value=MagicMock(points=[]))
    monkeypatch.setattr("src.retrieval.retriever.get_sparse_vector", lambda *_: None)

    context, sources = await retriever.retrieve(
        "what is this cover letter about?", user_id="user-1"
    )

    assert context == ""
    assert sources == []
    assert retriever.last_diagnostics["reason"] == "no_user_chunks"
    assert retriever.last_diagnostics["user_chunk_count"] == 0


@pytest.mark.asyncio
async def test_document_retriever_diagnoses_no_matching_chunks(mock_qdrant, monkeypatch):
    """No-source retrieval should distinguish indexed chunks from no semantic match."""
    retriever = DocumentRetriever()
    retriever.dense_ready = True
    mock_qdrant.collection_exists = AsyncMock(return_value=True)
    mock_qdrant.count = AsyncMock(return_value=MagicMock(count=4))
    mock_qdrant.query_points = AsyncMock(return_value=MagicMock(points=[]))
    monkeypatch.setattr("src.retrieval.retriever.get_sparse_vector", lambda *_: None)

    context, sources = await retriever.retrieve(
        "what is this cover letter about?", user_id="user-1"
    )

    assert context == ""
    assert sources == []
    assert retriever.last_diagnostics["reason"] == "no_matching_chunks"
    assert retriever.last_diagnostics["user_chunk_count"] == 4


@pytest.fixture
def reranker_no_token(monkeypatch):
    """Reranker with no HF_TOKEN to exercise local keyword fallback path."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return DocumentReranker()


@pytest.fixture
def fill_normalize_docs():
    """Simulate chunks from the 'beautiful websites.pdf' document."""
    return [
        Document(
            page_content=(
                "Fill\n"
                "Now review the entire website again and populate every section with realistic, "
                "polished content based on our research. Keep the design clean — no walls of text "
                "or clutter. The copy should be concise and impactful, consistent with the brand "
                "identity and positioning we've established."
            ),
            metadata={"source": "beautiful_websites.pdf", "section": "Fill"},
        ),
        Document(
            page_content=(
                "Normalize\n"
                "I like the current design of the page overall. However, since we've pulled in "
                "sections from different websites, the styling isn't fully consistent — things like "
                "fonts, spacing, colors, and component sizes don't always match up cleanly across "
                "sections."
            ),
            metadata={"source": "beautiful_websites.pdf", "section": "Normalize"},
        ),
        Document(
            page_content=(
                "Research\n"
                "First, review the existing website structure — go through every component and "
                "identify what content and information each section needs."
            ),
            metadata={"source": "beautiful_websites.pdf", "section": "Research"},
        ),
        Document(
            page_content="Some unrelated paragraph about machine learning algorithms.",
            metadata={"source": "other.pdf", "section": "intro"},
        ),
    ]


# ---------------------------------------------------------------------------
# Local reranker keyword scoring tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_reranker_keyword_fill(reranker_no_token, fill_normalize_docs):
    """Single-word query 'fill' should rank the Fill section chunk first."""
    results = await reranker_no_token.rerank("fill", fill_normalize_docs, top_k=2)
    assert len(results) >= 1
    top_doc = results[0]
    assert "Fill" in top_doc.page_content or "fill" in top_doc.page_content.lower(), (
        f"Expected Fill section chunk at rank 1, got: {top_doc.page_content[:80]}"
    )


@pytest.mark.asyncio
async def test_local_reranker_keyword_normalize(reranker_no_token, fill_normalize_docs):
    """Single-word query 'normalize' should rank the Normalize section chunk first."""
    results = await reranker_no_token.rerank("normalize", fill_normalize_docs, top_k=2)
    assert len(results) >= 1
    top_doc = results[0]
    assert "Normalize" in top_doc.page_content or "normalize" in top_doc.page_content.lower(), (
        f"Expected Normalize section chunk at rank 1, got: {top_doc.page_content[:80]}"
    )


@pytest.mark.asyncio
async def test_local_reranker_phrase_query(reranker_no_token, fill_normalize_docs):
    """Natural-language phrase 'what is fill' should also surface the Fill section."""
    results = await reranker_no_token.rerank("what is fill", fill_normalize_docs, top_k=2)
    assert len(results) >= 1
    top_doc = results[0]
    assert "fill" in top_doc.page_content.lower(), (
        f"Expected Fill section chunk at rank 1, got: {top_doc.page_content[:80]}"
    )


@pytest.mark.asyncio
async def test_local_reranker_empty_docs(reranker_no_token):
    """Empty document list should return empty list without error."""
    results = await reranker_no_token.rerank("fill", [], top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_local_reranker_respects_top_k(reranker_no_token, fill_normalize_docs):
    """Local reranker should return exactly top_k results when enough docs exist."""
    results = await reranker_no_token.rerank("fill", fill_normalize_docs, top_k=2)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Chunker header-awareness tests
# ---------------------------------------------------------------------------


def test_chunker_header_separators_include_markdown():
    """DocumentChunker should have markdown header separators in its config."""
    chunker = DocumentChunker()
    assert "\n# " in chunker.separators, "Chunker must split on '\\n# ' (h1 header)"
    assert "\n## " in chunker.separators, "Chunker must split on '\\n## ' (h2 header)"
    assert "\n### " in chunker.separators, "Chunker must split on '\\n### ' (h3 header)"


def test_chunker_header_separators_ordered_before_newline():
    """Markdown header separators must appear BEFORE '\\n\\n' in the list."""
    chunker = DocumentChunker()
    seps = chunker.separators
    h1_idx = seps.index("\n# ")
    nn_idx = seps.index("\n\n")
    assert h1_idx < nn_idx, "Header separator '\\n# ' must be listed before '\\n\\n' separator"


def test_chunker_preserves_section_header_with_body():
    """Section header line should appear in same chunk as its body paragraph."""
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    text = (
        "# Fill\n"
        "Now review the entire website again and populate every section with realistic, "
        "polished content based on our research. Keep the design clean.\n\n"
        "# Normalize\n"
        "Ensure styling is consistent across all sections."
    )
    docs = [Document(page_content=text)]
    chunks = chunker.chunk_documents(docs)

    fill_chunks = [c for c in chunks if "Fill" in c.page_content]
    assert fill_chunks, "At least one chunk must contain the 'Fill' header text"

    fill_chunk = fill_chunks[0]
    assert "populate" in fill_chunk.page_content.lower(), (
        "The 'Fill' section header chunk should include its body content"
    )
