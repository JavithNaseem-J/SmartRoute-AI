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
    mock_pipeline.tracker.calculate_savings.return_value = {
        "baseline_cost": 1.0,
        "actual_cost": 0.5,
        "savings": 0.5,
        "percentage": 50.0,
    }
    mock_pipeline.budget_manager.get_budget_status.return_value = {"daily": {"spent": 0}}
    mock_pipeline.budget_manager.check_health = AsyncMock(return_value=True)
    mock_pipeline.model_manager.configured = True
    mock_pipeline.model_manager.provider_ready = True
    mock_pipeline.model_manager.validate_provider = AsyncMock(return_value=None)

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
    assert response.json()["version"] == "2.1.0"


def test_readiness_checks_required_components(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["version"] == "2.1.0"
    assert all(value == "ok" for value in response.json()["components"].values())


def test_startup_validation_allows_local_qdrant_without_api_key(monkeypatch):
    import api.main as api_module

    for name, _hint in api_module._REQUIRED_ENV_VARS:
        monkeypatch.setenv(name, "configured")
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    api_module.validate_env()


def test_startup_validation_requires_active_provider_key(monkeypatch):
    import api.main as api_module

    for name, _hint in api_module._REQUIRED_ENV_VARS:
        monkeypatch.setenv(name, "configured")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        api_module.validate_env()


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


def test_query_provider_failure_returns_bad_gateway(client, api_key, monkeypatch):
    import api.main as api_module

    failed = {
        **api_module.pipeline.run.return_value,
        "answer": "Request failed. Please try again.",
        "success": False,
        "error": "pipeline_error",
    }
    api_module.pipeline.run = AsyncMock(return_value=failed)

    response = client.post(
        "/v1/query",
        json={"query": "What is AI?"},
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert response.status_code == 502
    assert response.json()["error"] == "pipeline_error"


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
    import api.main as api_module

    api_module.pipeline.tracker.get_statistics.assert_called_with(1, "test_user")


def test_savings_and_budget_endpoints_are_tenant_scoped(client, api_key):
    import api.main as api_module

    savings = client.get("/v1/savings", headers={"Authorization": f"Bearer {api_key}"})
    budget = client.get("/v1/budget", headers={"Authorization": f"Bearer {api_key}"})

    assert savings.status_code == 200
    assert budget.status_code == 200
    api_module.pipeline.tracker.calculate_savings.assert_called_with(1, 0.15, "test_user")
    api_module.pipeline.budget_manager.get_budget_status.assert_called_with("test_user")


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


def test_upload_normalizes_utf16_text_before_storage_and_indexing(client, api_key, monkeypatch):
    """Windows UTF-16 text uploads are stored as UTF-8 for the text loader."""
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
            return f"{user_id}/{filename}"

        async def upload(self, path, content, content_type):
            uploaded.append((path, content, content_type))

        async def delete(self, path):
            return None

    class FakeIndexer:
        def load_file(self, file_path, *, source, metadata=None):
            assert (
                file_path.read_text(encoding="utf-8")
                == "Cover letter for a software engineer role."
            )
            return [
                Document(
                    page_content="cover letter", metadata={"source": source, **(metadata or {})}
                )
            ]

        async def aindex_documents(self, documents):
            return len(documents)

        async def count_indexed_chunks(self, **_kwargs):
            return 1

        def get_stats(self):
            return {"chunker": {}}

    api_module.pipeline.retriever.reload = AsyncMock()
    monkeypatch.setattr(api_module, "SupabaseStorage", FakeStorage)
    monkeypatch.setattr(indexer_module, "DocumentIndexer", FakeIndexer)

    response = client.post(
        "/v1/documents/upload",
        files={
            "files": (
                "ATS Prompt.txt",
                "Cover letter for a software engineer role.".encode("utf-16"),
                "text/plain",
            )
        },
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert response.status_code == 200
    assert uploaded == [
        (
            "test_user/ATS Prompt.txt",
            b"Cover letter for a software engineer role.",
            "text/plain",
        )
    ]


def test_upload_rejects_malformed_utf16_text(client, api_key):
    response = client.post(
        "/v1/documents/upload",
        files={"files": ("broken.txt", b"\xff\xfe\x00", "text/plain")},
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert response.status_code == 422
    assert response.json()["detail"].startswith("Unsupported text encoding:")


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


def test_upload_returns_safe_specific_indexing_error(client, api_key, monkeypatch):
    """Safe RuntimeError messages from the indexer should reach the upload response."""
    from langchain_core.documents import Document

    import api.main as api_module
    import src.retrieval.indexer as indexer_module

    class FakeStorage:
        bucket = "smartroute-documents"

        @classmethod
        def from_env(cls):
            return cls()

        def object_path(self, user_id, filename):
            return f"{user_id}/test-{filename}"

        async def upload(self, path, content, content_type):
            return None

        async def delete(self, path):
            return None

    class FakeIndexer:
        def load_file(self, file_path, *, source, metadata=None):
            return [Document(page_content="hello", metadata={"source": source, **(metadata or {})})]

        async def aindex_documents(self, documents):
            raise RuntimeError(
                "Embedding generation failed. Check the FastEmbed model download and local runtime access."
            )

        def get_stats(self):
            return {"chunker": {}}

    monkeypatch.setattr(api_module, "SupabaseStorage", FakeStorage)
    monkeypatch.setattr(indexer_module, "DocumentIndexer", FakeIndexer)

    response = client.post(
        "/v1/documents/upload",
        files={"files": ("note.txt", b"hello", "text/plain")},
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Embedding generation failed. Check the FastEmbed model download and local runtime access."
    )


def test_delete_keeps_document_active_when_vector_removal_fails(client, api_key, monkeypatch):
    import api.main as api_module
    import src.retrieval.indexer as indexer_module

    operations = []

    class FakeStorage:
        @classmethod
        def from_env(cls):
            return cls()

        async def delete(self, path):
            operations.append(("storage-delete", path))

    class FakeIndexer:
        async def adelete_document(self, filename, source=None, user_id=None):
            raise RuntimeError("Qdrant delete rejected")

    monkeypatch.setattr(api_module, "SupabaseStorage", FakeStorage)
    monkeypatch.setattr(
        api_module,
        "get_active_document",
        lambda *_args: {
            "filename": "old.txt",
            "storage_path": "test_user/old.txt",
        },
    )
    mark_deleted = MagicMock()
    monkeypatch.setattr(api_module, "mark_document_deleted", mark_deleted)
    monkeypatch.setattr(indexer_module, "DocumentIndexer", FakeIndexer)

    response = client.delete(
        "/v1/documents/old.txt",
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Document vector deletion failed. The document remains active."
    )
    assert operations == []
    mark_deleted.assert_not_called()


def test_clear_keeps_documents_active_when_vector_removal_fails(client, api_key, monkeypatch):
    import api.main as api_module
    import src.retrieval.indexer as indexer_module

    operations = []

    class FakeStorage:
        @classmethod
        def from_env(cls):
            return cls()

        async def delete(self, path):
            operations.append(("storage-delete", path))

    class FakeIndexer:
        async def aclear_all_documents(self, user_id=None):
            raise RuntimeError("Qdrant delete rejected")

    monkeypatch.setattr(api_module, "SupabaseStorage", FakeStorage)
    monkeypatch.setattr(
        api_module,
        "list_active_documents",
        lambda *_args: [{"filename": "old.txt", "storage_path": "test_user/old.txt"}],
    )
    mark_deleted = MagicMock()
    monkeypatch.setattr(api_module, "mark_document_deleted", mark_deleted)
    monkeypatch.setattr(indexer_module, "DocumentIndexer", FakeIndexer)

    response = client.delete(
        "/v1/documents",
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == ("Document vector clear failed. Documents remain active.")
    assert operations == []
    mark_deleted.assert_not_called()
