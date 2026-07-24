"""
conftest.py — Global test fixtures.

Sets dummy cloud env vars before any module is imported so that
CostTracker, BudgetManager, ConversationMemory, and the Qdrant client
don't raise RuntimeError during test collection.

The real cloud connections are mocked globally using autouse fixtures.
"""

import os
from unittest.mock import AsyncMock, MagicMock

import fakeredis
import pytest

import src.core.dependencies as deps

# ── Inject fake cloud env vars before any import tries to connect ─────────────
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "rediss://default:test@test.upstash.io:6379")
os.environ.setdefault("QDRANT_URL", "https://test.qdrant.io")
os.environ.setdefault("QDRANT_API_KEY", "test-key")
os.environ.setdefault("NVIDIA_API_KEY", "test-nvidia-key")
os.environ.setdefault("HF_TOKEN", "dummy-hf-token-for-testing")
os.environ.setdefault("SMARTROUTE_API_KEY", "dev-key-change-in-production")


@pytest.fixture(autouse=True)
async def mock_qdrant():
    """Globally mock AsyncQdrantClient so tests never hit the real Qdrant Cloud."""
    mock_instance = MagicMock()
    mock_instance.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
    mock_instance.search = AsyncMock(return_value=[])
    mock_instance.upsert = AsyncMock(return_value=None)
    mock_instance.collection_exists = AsyncMock(return_value=False)
    mock_instance.count = AsyncMock(return_value=MagicMock(count=0))

    # Directly inject into the singleton cache to bypass `from ... import` binding issues
    old = deps._qdrant_client
    deps._qdrant_client = mock_instance
    yield mock_instance
    deps._qdrant_client = old


@pytest.fixture(autouse=True)
async def mock_redis():
    """Globally mock Upstash Redis with fakeredis."""
    fake_async = fakeredis.FakeAsyncRedis(decode_responses=True)

    # Directly inject into the singleton cache
    old = deps._redis_client
    deps._redis_client = fake_async
    yield fake_async
    deps._redis_client = old


@pytest.fixture(autouse=True)
async def mock_embeddings():
    """Globally mock HuggingFaceEndpointEmbeddings to avoid real API calls in tests."""
    mock_emb = MagicMock()
    # Return deterministic 384-dim zero vectors matching the length of the input
    mock_emb.embed_query = MagicMock(return_value=[0.0] * 384)
    mock_emb.aembed_query = AsyncMock(return_value=[0.0] * 384)

    def mock_embed_docs(texts):
        return [[0.0] * 384 for _ in texts]

    async def amock_embed_docs(texts):
        return mock_embed_docs(texts)

    mock_emb.embed_documents = MagicMock(side_effect=mock_embed_docs)
    mock_emb.aembed_documents = AsyncMock(side_effect=amock_embed_docs)

    # Directly inject into the singleton cache
    old = deps._embeddings
    deps._embeddings = mock_emb
    yield mock_emb
    deps._embeddings = old


@pytest.fixture(autouse=True)
def setup_test_db():
    """Create the SQLAlchemy schema in the SQLite in-memory database."""
    from src.cost.tracker import Base, CostTracker

    tracker = CostTracker()
    Base.metadata.create_all(bind=tracker.engine)
    yield
    Base.metadata.drop_all(bind=tracker.engine)
    tracker.close()
