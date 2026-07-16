"""
Tests for RetrievalCache — in-memory LRU backend only.
Redis backend is integration-tested separately when REDIS_URL is set.
"""
import pytest
from src.retrieval.cache import RetrievalCache


@pytest.fixture
def cache():
    return RetrievalCache(max_size=3)


def test_miss_on_empty_cache(cache):
    """A fresh cache returns None for any query."""
    assert cache.get("anything") is None


def test_set_and_get(cache):
    """A cached result is returned correctly."""
    cache.set("What is RAG?", "RAG means retrieval-augmented generation.", ["doc1.pdf"])
    result = cache.get("What is RAG?")

    assert result is not None
    context, sources = result
    assert context == "RAG means retrieval-augmented generation."
    assert sources == ["doc1.pdf"]


def test_normalization_case_insensitive(cache):
    """Queries are normalized (lowercased + stripped) before hashing."""
    cache.set("what is rag?", "RAG context.", ["s1"])
    # Uppercase + whitespace variant should still hit
    result = cache.get("  WHAT IS RAG?  ")
    assert result is not None, "Cache miss: normalization failed"


def test_lru_eviction_at_max_size(cache):
    """When max_size=3 is exceeded, the least-recently-used entry is evicted."""
    cache.set("Q1", "A1", [])
    cache.set("Q2", "A2", [])
    cache.set("Q3", "A3", [])

    # Access Q1 to mark it as recently used
    cache.get("Q1")

    # Adding Q4 should evict Q2 (the LRU entry, since Q1 was just accessed)
    cache.set("Q4", "A4", [])

    assert cache.get("Q1") is not None, "Q1 should still be cached (was recently accessed)"
    assert cache.get("Q2") is None, "Q2 should have been evicted (LRU)"
    assert cache.get("Q3") is not None
    assert cache.get("Q4") is not None


def test_clear_empties_cache(cache):
    """clear() removes all entries."""
    cache.set("Q1", "A1", [])
    cache.clear()
    assert cache.get("Q1") is None


def test_cache_returns_empty_sources(cache):
    """Cache handles empty source lists correctly."""
    cache.set("Q?", "Some answer.", [])
    ctx, sources = cache.get("Q?")
    assert sources == []
