"""
Tests for RetrievalCache — Upstash Redis backend (mocked).
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import fakeredis
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

from src.retrieval.cache import RetrievalCache  # noqa: E402


@pytest.fixture
def cache():
    """Return a RetrievalCache with a mocked Redis connection."""
    fake_redis = fakeredis.FakeStrictRedis(decode_responses=True)
    with patch("src.retrieval.cache.sync_redis.from_url", return_value=fake_redis):
        return RetrievalCache(ttl_seconds=3600)


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
