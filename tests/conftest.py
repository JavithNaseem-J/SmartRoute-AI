"""
conftest.py — Global test fixtures.

Sets dummy cloud env vars before any module is imported so that
CostTracker, BudgetManager, ConversationMemory, and VectorStore
don't raise RuntimeError during test collection.

The real cloud connections are mocked globally using autouse fixtures.
"""

import os
from unittest.mock import MagicMock, patch

import fakeredis
import pytest

# ── Inject fake cloud env vars before any import tries to connect ─────────────
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "rediss://default:test@test.upstash.io:6379")
os.environ.setdefault("QDRANT_URL", "https://test.qdrant.io")
os.environ.setdefault("QDRANT_API_KEY", "test-key")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("SMARTROUTE_API_KEY", "dev-key-change-in-production")


@pytest.fixture(autouse=True, scope="session")
def mock_qdrant():
    """Globally mock QdrantClient so tests never hit the real Qdrant Cloud."""
    with patch("src.retrieval.vector_store.QdrantClient", autospec=True) as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.get_collections.return_value = MagicMock(collections=[])
        yield mock_client


@pytest.fixture(autouse=True, scope="session")
def mock_redis():
    """Globally mock Upstash Redis with fakeredis."""
    fake_redis = fakeredis.FakeStrictRedis(decode_responses=True)
    with patch("src.memory.conversation.sync_redis.from_url", return_value=fake_redis):
        with patch("src.cost.budget.sync_redis.from_url", return_value=fake_redis):
            yield fake_redis


@pytest.fixture(autouse=True, scope="session")
def setup_test_db():
    """Create the SQLAlchemy schema in the SQLite in-memory database."""
    from src.cost.tracker import Base, engine

    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
