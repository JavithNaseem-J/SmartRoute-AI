import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import jwt
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)


@pytest.fixture
def client():
    """Create test client with mocked pipeline."""
    from fastapi.testclient import TestClient

    import api.main as api_module

    _MOCK_RESULT = {
        "answer": "Test answer",
        "model_used": "llama_3_1_8b",
        "complexity": "simple",
        "confidence": 0.95,
        "cost": 0.0,
        "latency": 0.5,
        "sources": [],
        "success": True,
        "error": None,
    }

    # pipeline.run is now async — must be AsyncMock so `await pipeline.run(...)` works.
    mock_pipeline = MagicMock()
    mock_pipeline.run = AsyncMock(return_value=_MOCK_RESULT)
    mock_pipeline.tracker.get_statistics.return_value = {"total_queries": 10}
    mock_pipeline.budget_manager.get_budget_status.return_value = {"daily": {"spent": 0}}

    original = api_module.pipeline
    api_module.pipeline = mock_pipeline

    yield TestClient(api_module.app)

    api_module.pipeline = original


@pytest.fixture
def api_key():
    jwt_secret = os.getenv(
        "SUPABASE_JWT_SECRET", "super-secret-jwt-token-with-at-least-32-characters-long"
    )
    return jwt.encode({"sub": "test_user"}, jwt_secret, algorithm="HS256")


def test_health_check(client):
    """Test health endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_version_endpoint_reports_build_identity(client, monkeypatch):
    """Test unauthenticated deployment identity endpoint."""
    commit_sha = "a" * 40
    build_time = "2026-09-12T00:00:00Z"
    monkeypatch.setenv("SMARTROUTE_COMMIT_SHA", commit_sha)
    monkeypatch.setenv("SMARTROUTE_BUILD_TIME", build_time)

    response = client.get("/version")

    assert response.status_code == 200
    assert response.json() == {
        "commit_sha": commit_sha,
        "build_time": build_time,
    }


def test_query_requires_auth(client):
    """Test query endpoint requires API key/JWT token."""
    response = client.post("/v1/query", json={"query": "What is AI?"})
    assert response.status_code == 403 or response.status_code == 401


def test_query_with_auth(client, api_key):
    """Test query endpoint with valid JWT."""
    response = client.post(
        "/v1/query",
        json={"query": "What is AI?"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_stats_endpoint(client, api_key):
    """Test stats endpoint."""
    response = client.get("/v1/stats", headers={"Authorization": f"Bearer {api_key}"})
    assert response.status_code == 200


def test_list_documents_endpoint(client, api_key):
    """Test list documents endpoint."""
    response = client.get("/v1/documents", headers={"Authorization": f"Bearer {api_key}"})
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total" in data


def test_upload_documents_uses_cloud_storage(client, api_key, monkeypatch):
    """Test document upload stores, indexes, and records metadata without local persistence."""
    from langchain_core.documents import Document

    import api.main as api_module
    import src.retrieval.indexer as indexer_module

    uploaded = []

    class FakeStorage:
        bucket = "smartroute-documents"

        @classmethod
        def from_env(cls):
            return cls()

        def object_path(self, user_id, filename):
            return f"{user_id}/test-{filename}"

        async def upload(self, path, content, content_type):
            uploaded.append((path, content, content_type))

        async def delete(self, path):
            uploaded.append(("deleted", path, ""))

    class FakeIndexer:
        def load_file(self, file_path, *, source, metadata=None):
            return [Document(page_content="hello", metadata={"source": source, **(metadata or {})})]

        async def aindex_documents(self, documents):
            assert documents[0].metadata["storage_path"] == "test_user/test-note.txt"

        def get_stats(self):
            return {"chunker": {}}

    def fake_create_document_record(_tracker, **record):
        return record

    api_module.pipeline.retriever.reload = AsyncMock()
    monkeypatch.setattr(api_module, "SupabaseStorage", FakeStorage)
    monkeypatch.setattr(api_module, "create_document_record", fake_create_document_record)
    monkeypatch.setattr(indexer_module, "DocumentIndexer", FakeIndexer)

    response = client.post(
        "/v1/documents/upload",
        files={"files": ("note.txt", b"hello", "text/plain")},
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert response.status_code == 200
    assert uploaded == [("test_user/test-note.txt", b"hello", "text/plain")]
    assert response.json()["documents"][0]["storage_path"] == "test_user/test-note.txt"
