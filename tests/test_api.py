import os
import sys
import time
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
    jwt_secret = os.getenv("SUPABASE_JWT_SECRET", "test-supabase-jwt-secret-for-unit-tests")
    return jwt.encode(
        {"sub": "test_user", "exp": int(time.time()) + 3600},
        jwt_secret,
        algorithm="HS256",
    )


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


def test_demo_token_allows_portfolio_query(client):
    """The public portfolio can obtain a short-lived JWT without a sign-in screen."""
    token_response = client.post(
        "/v1/auth/demo-token",
        json={"session_id": "9924ac52-3a1c-4109-8433-d756e1dc52da"},
    )

    assert token_response.status_code == 200
    token = token_response.json()["access_token"]

    response = client.post(
        "/v1/query",
        json={"query": "What is AI?"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["success"] is True


def test_query_rejects_insecure_default_jwt_secret(client, monkeypatch):
    """The app must not accept tokens signed with the public sample secret."""
    from src.utils.security import _INSECURE_DEFAULT_SECRET

    monkeypatch.setenv("SUPABASE_JWT_SECRET", _INSECURE_DEFAULT_SECRET)
    token = jwt.encode(
        {"sub": "test_user", "exp": int(time.time()) + 3600},
        _INSECURE_DEFAULT_SECRET,
        algorithm="HS256",
    )

    response = client.post(
        "/v1/query",
        json={"query": "What is AI?"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 503


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
            return 1

        async def count_indexed_chunks(self, *, user_id=None, source=None, filename=None):
            assert user_id == "test_user"
            assert source == "test_user/test-note.txt"
            return 1

        def get_stats(self):
            return {"chunker": {}}

    def fake_create_document_record(_tracker, **record):
        return record

    api_module.pipeline.retriever.reload = AsyncMock()
    monkeypatch.setattr(api_module, "SupabaseStorage", FakeStorage)
    monkeypatch.setattr(api_module, "create_document_record", fake_create_document_record)
    monkeypatch.setattr(indexer_module, "DocumentIndexer", FakeIndexer)
    api_module.pipeline.semantic_cache.invalidate_user = AsyncMock()

    response = client.post(
        "/v1/documents/upload",
        files={"files": ("note.txt", b"hello", "text/plain")},
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert response.status_code == 200
    assert uploaded == [("test_user/test-note.txt", b"hello", "text/plain")]
    assert response.json()["documents"][0]["storage_path"] == "test_user/test-note.txt"
    assert response.json()["stats"]["indexed_chunks"] == 1
    assert response.json()["stats"]["verified_chunks"] == 1
    api_module.pipeline.semantic_cache.invalidate_user.assert_awaited_once_with("test_user")


def test_upload_rolls_back_storage_when_indexing_verification_fails(client, api_key, monkeypatch):
    """If Qdrant cannot verify searchable chunks, no document record is saved."""
    from langchain_core.documents import Document

    import api.main as api_module
    import src.retrieval.indexer as indexer_module

    operations = []
    records = []

    class FakeStorage:
        bucket = "smartroute-documents"

        @classmethod
        def from_env(cls):
            return cls()

        def object_path(self, user_id, filename):
            return f"{user_id}/test-{filename}"

        async def upload(self, path, content, content_type):
            operations.append(("upload", path, content_type))

        async def delete(self, path):
            operations.append(("delete", path, ""))

    class FakeIndexer:
        def load_file(self, file_path, *, source, metadata=None):
            return [Document(page_content="hello", metadata={"source": source, **(metadata or {})})]

        async def aindex_documents(self, documents):
            return 2

        async def count_indexed_chunks(self, *, user_id=None, source=None, filename=None):
            return 1

        def get_stats(self):
            return {"chunker": {}}

    def fake_create_document_record(_tracker, **record):
        records.append(record)
        return record

    monkeypatch.setattr(api_module, "SupabaseStorage", FakeStorage)
    monkeypatch.setattr(api_module, "create_document_record", fake_create_document_record)
    monkeypatch.setattr(indexer_module, "DocumentIndexer", FakeIndexer)

    response = client.post(
        "/v1/documents/upload",
        files={"files": ("note.txt", b"hello", "text/plain")},
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Vector indexing verification failed. Uploaded chunks are not searchable yet."
    )
    assert operations == [
        ("upload", "test_user/test-note.txt", "text/plain"),
        ("delete", "test_user/test-note.txt", ""),
    ]
    assert records == []


def test_upload_rejects_invalid_pdf_before_storage(client, api_key, monkeypatch):
    """A .pdf extension alone is not enough; content must look like a PDF."""
    import api.main as api_module

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

    monkeypatch.setattr(api_module, "SupabaseStorage", FakeStorage)

    response = client.post(
        "/v1/documents/upload",
        files={"files": ("fake.pdf", b"not a pdf", "application/pdf")},
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert response.status_code == 422
    assert uploaded == []


def test_upload_rejects_oversized_document_before_storage(client, api_key, monkeypatch):
    """Oversized documents are rejected before cloud upload."""
    import api.main as api_module

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

    monkeypatch.setattr(api_module, "SupabaseStorage", FakeStorage)
    monkeypatch.setattr(api_module, "MAX_DOCUMENT_UPLOAD_BYTES", 4)

    response = client.post(
        "/v1/documents/upload",
        files={"files": ("large.txt", b"hello", "text/plain")},
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert response.status_code == 413
    assert uploaded == []


def test_upload_rolls_back_storage_when_indexing_fails(client, api_key, monkeypatch):
    """If vector indexing fails, uploaded storage objects are removed and no record is written."""
    from langchain_core.documents import Document

    import api.main as api_module
    import src.retrieval.indexer as indexer_module

    operations = []
    records = []

    class FakeStorage:
        bucket = "smartroute-documents"

        @classmethod
        def from_env(cls):
            return cls()

        def object_path(self, user_id, filename):
            return f"{user_id}/test-{filename}"

        async def upload(self, path, content, content_type):
            operations.append(("upload", path, content_type))

        async def delete(self, path):
            operations.append(("delete", path, ""))

    class FakeIndexer:
        def load_file(self, file_path, *, source, metadata=None):
            return [Document(page_content="hello", metadata={"source": source, **(metadata or {})})]

        async def aindex_documents(self, documents):
            raise RuntimeError("Vector indexing failed")

        def get_stats(self):
            return {"chunker": {}}

    def fake_create_document_record(_tracker, **record):
        records.append(record)
        return record

    monkeypatch.setattr(api_module, "SupabaseStorage", FakeStorage)
    monkeypatch.setattr(api_module, "create_document_record", fake_create_document_record)
    monkeypatch.setattr(indexer_module, "DocumentIndexer", FakeIndexer)

    response = client.post(
        "/v1/documents/upload",
        files={"files": ("note.txt", b"hello", "text/plain")},
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Vector indexing failed. Check Qdrant and embedding provider configuration."
    )
    assert operations == [
        ("upload", "test_user/test-note.txt", "text/plain"),
        ("delete", "test_user/test-note.txt", ""),
    ]
    assert records == []
