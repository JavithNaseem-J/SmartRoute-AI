import asyncio
import os
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, AsyncIterator, List, Optional, TypedDict

from dotenv import load_dotenv

load_dotenv()

import uvicorn  # noqa: E402
from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from slowapi import Limiter, _rate_limit_exceeded_handler  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402
from slowapi.util import get_remote_address  # noqa: E402

from src.pipeline.inference import InferencePipeline  # noqa: E402
from src.documents import (  # noqa: E402
    SupabaseStorage,
    create_document_record,
    get_active_document,
    list_active_documents,
    mark_document_deleted,
)
from src.documents.storage import guess_content_type  # noqa: E402
from src.utils.alerting import send_alert  # noqa: E402
from src.utils.logger import logger  # noqa: E402
from src.utils.security import create_demo_jwt, require_jwt  # noqa: E402
from src.utils.tracing import setup_tracing  # noqa: E402

#  validation

_REQUIRED_ENV_VARS = [
    ("DATABASE_URL", "Supabase PostgreSQL  -> https://supabase.com"),
    ("REDIS_URL", "Upstash Redis        -> https://upstash.com"),
    ("QDRANT_URL", "Qdrant Cloud         -> https://cloud.qdrant.io"),
    ("QDRANT_API_KEY", "Qdrant Cloud         -> https://cloud.qdrant.io"),
    ("HF_TOKEN", "HuggingFace API      -> https://huggingface.co/settings/tokens"),
    ("SUPABASE_URL", "Supabase Project URL -> https://supabase.com"),
    ("SUPABASE_SERVICE_ROLE_KEY", "Supabase service key -> Project Settings / API"),
    ("SUPABASE_STORAGE_BUCKET", "Supabase Storage bucket"),
    ("SUPABASE_JWT_SECRET", "Supabase JWT secret -> Project Settings / API"),
]


def validate_env() -> None:
    """Fail immediately on startup if any required credential is absent.

    This surfaces missing config before the server binds to a port,
    preventing confusing 500 errors at request time.
    """
    missing = [(var, hint) for var, hint in _REQUIRED_ENV_VARS if not os.getenv(var)]
    if not missing:
        return

    lines = [
        "\n" + "=" * 60,
        "STARTUP FAILED — missing required environment variables:",
        "=" * 60,
    ]
    for var, hint in missing:
        lines.append(f"  [MISSING]  {var}")
        lines.append(f"             Get it from: {hint}")
    lines += [
        "=" * 60,
        "Set these in your .env file or Render environment variables.\n",
    ]
    logger.error("\n".join(lines))
    sys.exit(1)


#  Application lifespan

pipeline: Optional[InferencePipeline] = None
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", "data/documents"))
ALLOWED_DOCUMENT_SUFFIXES = {".pdf", ".txt", ".md"}
MAX_DOCUMENT_UPLOAD_BYTES = int(os.getenv("MAX_DOCUMENT_UPLOAD_BYTES", str(10 * 1024 * 1024)))


def _validate_document_upload(filename: str, content: bytes, content_type: str) -> None:
    suffix = Path(filename).suffix.lower()
    if not filename or suffix not in ALLOWED_DOCUMENT_SUFFIXES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported document type for {filename or 'unnamed file'}",
        )
    if not content:
        raise HTTPException(status_code=422, detail=f"Uploaded document is empty: {filename}")
    if len(content) > MAX_DOCUMENT_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Document is too large: {filename}. "
                f"Maximum size is {MAX_DOCUMENT_UPLOAD_BYTES} bytes."
            ),
        )

    normalized_type = (content_type or "").split(";")[0].strip().lower()
    if suffix == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise HTTPException(status_code=422, detail=f"Invalid PDF content: {filename}")
        if normalized_type and normalized_type not in {
            "application/pdf",
            "application/octet-stream",
        }:
            raise HTTPException(status_code=422, detail=f"Invalid PDF content type: {filename}")
        return

    if b"\x00" in content[:4096]:
        raise HTTPException(status_code=422, detail=f"Invalid text document content: {filename}")
    try:
        content[:4096].decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=422, detail=f"Document must be UTF-8 text: {filename}")

    if normalized_type and not (
        normalized_type.startswith("text/")
        or normalized_type in {"application/octet-stream", "application/markdown"}
    ):
        raise HTTPException(status_code=422, detail=f"Invalid text content type: {filename}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Runs validate_env + pipeline init before serving; disposes on shutdown."""
    global pipeline
    validate_env()
    try:
        pipeline = InferencePipeline()
        logger.info("Pipeline initialised - all cloud services connected.")
    except Exception as exc:
        logger.error(f"Pipeline init failed: {exc}")
        sys.exit(1)
    yield
    logger.info("Shutting down SmartRoute-AI.")


# Rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="SmartRoute-AI API",
    description="Cost-optimised async RAG with intelligent LLM routing",
    version="2.0.0",
    lifespan=lifespan,  # replaces deprecated @app.on_event("startup")
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

ALLOWED_ORIGINS_RAW = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:8000",
)
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS_RAW.split(",") if origin.strip()]


def _accepts_html(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return "text/html" in accept


def _read_build_metadata_file(filename: str) -> str:
    try:
        return (Path("/app") / filename).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _deployment_commit_sha() -> str:
    metadata_sha = _read_build_metadata_file(".commit_sha")
    if metadata_sha and metadata_sha != "unknown":
        return metadata_sha

    for env_name in ("SMARTROUTE_COMMIT_SHA", "RENDER_GIT_COMMIT"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return metadata_sha or "unknown"


def _deployment_build_time() -> str:
    value = os.getenv("SMARTROUTE_BUILD_TIME", "").strip()
    if value:
        return value
    return _read_build_metadata_file(".build_time") or "unknown"


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "HEAD"],
    allow_headers=["*"],
)

setup_tracing(app)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    error_details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    await send_alert(
        title="Unhandled API Exception (500)",
        message=f"Path: {request.url.path}\nError: {str(exc)}\n\n```python\n{error_details[:1500]}\n```",
        level="critical",
    )
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


@app.get("/health", tags=["system"])
async def health_check():
    """Cheap liveness probe for Docker, Render, and load balancers."""
    return {
        "status": "healthy" if pipeline else "starting",
        "version": "2.0.0",
    }


@app.get("/version", tags=["system"])
async def version():
    """Unauthenticated deployment identity used by CI/CD verification."""
    return {
        "commit_sha": _deployment_commit_sha(),
        "build_time": _deployment_build_time(),
    }


async def _component_status() -> dict:
    if not pipeline:
        return {"pipeline": "initializing"}

    components = {
        "router": "ok" if pipeline.router else "error",
        "model_manager": "ok" if pipeline.model_manager else "error",
        "retriever": "ok" if pipeline.retriever else "error",
        "cost_tracker": "ok" if pipeline.tracker else "error",
        "redis": "error",
        "qdrant": "error",
        "postgres": "error",
    }

    try:
        from src.core.dependencies import get_redis_client

        redis_client = get_redis_client()
        if await redis_client.ping():
            components["redis"] = "ok"
    except Exception as e:
        logger.warning(f"Redis readiness check failed: {e}")
        components["redis"] = "error"

    try:
        from src.core.dependencies import get_qdrant_client

        qdrant_client = get_qdrant_client()
        await qdrant_client.get_collections()
        components["qdrant"] = "ok"
    except Exception as e:
        logger.warning(f"Qdrant readiness check failed: {e}")
        components["qdrant"] = "error"

    try:
        if pipeline.tracker and pipeline.tracker.engine:
            import asyncio

            from sqlalchemy import text

            def ping_db():
                with pipeline.tracker.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))

            await asyncio.to_thread(ping_db)
            components["postgres"] = "ok"
    except Exception as e:
        logger.warning(f"Postgres readiness check failed: {e}")
        components["postgres"] = "error"

    return components


@app.get("/ready", tags=["system"])
async def readiness_check():
    """Readiness probe that checks runtime dependencies."""
    components = await _component_status()
    ready = bool(pipeline) and all(value == "ok" for value in components.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "version": "2.0.0",
            "components": components,
        },
    )


# ── Authentication ────────────────────────────────────────────────────────────


def require_api_key(payload: dict = Depends(require_jwt)) -> str:
    """JWT validation facade. Returns the user ID (sub) from the token."""
    return str(payload["sub"])


# ── Request / Response models ─────────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User query")
    strategy: Optional[str] = Field(
        None, description="Routing strategy: cost_optimized | quality_first | balanced"
    )
    use_retrieval: bool = Field(True, description="Enable RAG retrieval")
    session_id: Optional[str] = Field(None, description="Session ID for multi-turn conversation")


class QueryResponse(BaseModel):
    answer: str
    model_used: Optional[str]
    complexity: Optional[str]
    confidence: float
    cost: float
    latency: float
    sources: List[str]
    success: bool
    error: Optional[str] = None


class DemoTokenRequest(BaseModel):
    session_id: Optional[str] = Field(
        None, description="Client-generated browser session ID for demo isolation"
    )


class DemoTokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_at: int
    session_id: str


class PendingDocumentRecord(TypedDict):
    user_id: str
    filename: str
    content_type: str
    size_bytes: int
    storage_bucket: str
    storage_path: str


# ── Public endpoints ──────────────────────────────────────────────────────────


@app.get("/")
@app.head("/")
async def root(request: Request):
    index_path = FRONTEND_DIST / "index.html"
    if index_path.exists() and _accepts_html(request):
        return FileResponse(index_path)

    return {
        "status": "healthy" if pipeline else "degraded",
        "service": "SmartRoute-AI",
        "version": "2.0.0",
        "endpoints": {
            "query": "/v1/query",
            "batch": "/v1/query/batch",
            "stream": "/v1/query/stream",
            "stats": "/v1/stats",
            "savings": "/v1/savings",
            "budget": "/v1/budget",
            "models": "/v1/models",
            "health": "/health",
            "ready": "/ready",
            "docs": "/docs",
        },
    }


# ── Versioned router ──────────────────────────────────────────────────────
#
# All business endpoints go on v1_router so we can introduce /v2 later
# without touching existing client code. The prefix is injected at mount time.

v1 = APIRouter(prefix="/v1", tags=["v1"])


@v1.post("/auth/demo-token", response_model=DemoTokenResponse)
@limiter.limit("20/hour")
async def demo_token(request: Request, token_request: DemoTokenRequest):
    """Issue a short-lived server-signed JWT for the frictionless portfolio demo."""
    return create_demo_jwt(token_request.session_id)


@v1.post("/query", response_model=QueryResponse)
@limiter.limit("30/minute")
async def query(
    request: Request,
    query_request: QueryRequest,
    user_id: str = Depends(require_api_key),
):
    """Process a query — fully async, no thread pool required."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        result = await pipeline.run(
            query=query_request.query,
            strategy=query_request.strategy,
            use_retrieval=query_request.use_retrieval,
            session_id=query_request.session_id,
            user_id=user_id,
        )
        if result.get("latency", 0) > 10.0:
            from src.utils.alerting import send_alert

            asyncio.create_task(
                send_alert(
                    "High API Latency",
                    f"Query took {result['latency']:.2f}s to process.\nModel: {result.get('model_used')}",
                    "warning",
                )
            )
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Query failed")


@v1.post("/query/batch")
@limiter.limit("10/minute")
async def query_batch(
    request: Request,
    queries: List[str],
    strategy: Optional[str] = None,
    use_retrieval: bool = True,
    user_id: str = Depends(require_api_key),
):
    """Process multiple queries concurrently. Max 10 queries per call."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    if not queries:
        raise HTTPException(status_code=422, detail="queries list cannot be empty")
    if len(queries) > 10:
        raise HTTPException(status_code=422, detail="Maximum 10 queries per batch request")
    results = await pipeline.batch_run(
        queries=queries, strategy=strategy, use_retrieval=use_retrieval, user_id=user_id
    )
    return results


@v1.post("/query/stream")
@limiter.limit("30/minute")
async def query_stream(
    request: Request,
    query_request: QueryRequest,
    user_id: str = Depends(require_api_key),
):
    """Stream LLM tokens as Server-Sent Events (async generator)."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")

    async def _token_generator() -> AsyncIterator[str]:
        import json

        try:
            async for item in pipeline.astream_run(
                query=query_request.query,
                strategy=query_request.strategy,
                use_retrieval=query_request.use_retrieval,
                session_id=query_request.session_id,
                user_id=user_id,
            ):
                yield f"data: {json.dumps(item)}\n\n"
        except Exception as e:
            logger.error(f"Stream failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': 'Stream failed'})}\n\n"

    return StreamingResponse(
        _token_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@v1.get("/stats")
@limiter.limit("60/minute")
async def get_stats(
    request: Request,
    days: int = 1,
    _: str = Depends(require_api_key),
):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    return await asyncio.to_thread(pipeline.tracker.get_statistics, days)


@v1.get("/savings")
@limiter.limit("60/minute")
async def get_savings(
    request: Request,
    days: int = 1,
    _: str = Depends(require_api_key),
):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    return await asyncio.to_thread(pipeline.tracker.calculate_savings, days)


@v1.get("/budget")
@limiter.limit("60/minute")
async def get_budget(
    request: Request,
    _: str = Depends(require_api_key),
):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    return await asyncio.to_thread(pipeline.budget_manager.get_budget_status)


@v1.get("/models")
async def list_models(_: str = Depends(require_api_key)):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    available = list(pipeline.model_manager.config.get("openrouter_models", {}).keys())
    loaded = list(pipeline.model_manager.loaded_models.keys())
    return {"available": available, "loaded": loaded}


@v1.delete("/memory/{session_id}")
async def clear_memory(session_id: str, user_id: str = Depends(require_api_key)):
    """Clear conversation history for a session."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    await pipeline.memory.clear(user_id, session_id)
    return {"status": "cleared", "session_id": session_id}


@v1.post("/index")
async def index_documents(_: str = Depends(require_api_key)):
    """Trigger indexing of documents in the data/documents directory."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    from src.retrieval.indexer import DocumentIndexer

    try:
        indexer = DocumentIndexer()
        # Async indexing on the active event loop
        await indexer.aindex_directory(DOCUMENTS_DIR)
        # Reload the retriever to pick up new documents
        if hasattr(pipeline.retriever, "reload"):
            await pipeline.retriever.reload()
        stats = indexer.get_stats()
        return {"status": "success", "stats": stats}
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail="Indexing failed")


@v1.post("/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    user_id: str = Depends(require_api_key),
):
    """Upload PDF, TXT, or MD documents to Supabase Storage and index them."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    if not files:
        raise HTTPException(status_code=422, detail="No files uploaded")

    from src.retrieval.indexer import DocumentIndexer

    try:
        storage = SupabaseStorage.from_env()
    except RuntimeError as e:
        logger.error(f"Supabase Storage configuration failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Supabase Storage is misconfigured. Check bucket and service role key.",
        )

    saved_files: List[dict[str, Any]] = []
    uploaded_paths: List[str] = []

    async def cleanup_uploaded_paths() -> None:
        for storage_path in uploaded_paths:
            try:
                await storage.delete(storage_path)
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to clean up uploaded object {storage_path}: {cleanup_error}"
                )

    try:
        try:
            indexer = DocumentIndexer()
        except Exception as e:
            logger.error(f"Document indexer initialization failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=503,
                detail="Document indexing is unavailable. Check embedding and vector database configuration.",
            )

        documents_to_index = []
        pending_records: List[PendingDocumentRecord] = []
        indexed_chunks = 0

        with TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            for upload in files:
                filename = Path(upload.filename or "").name
                content = await upload.read()
                content_type = upload.content_type or guess_content_type(filename)
                _validate_document_upload(filename, content, content_type)
                storage_path = storage.object_path(user_id, filename)
                try:
                    await storage.upload(storage_path, content, content_type)
                except Exception as e:
                    logger.error(
                        f"Supabase Storage upload failed for {filename}: {e}", exc_info=True
                    )
                    raise HTTPException(
                        status_code=502,
                        detail="Supabase Storage upload failed. Check bucket and service role key.",
                    )
                uploaded_paths.append(storage_path)

                temp_path = temp_root / filename
                temp_path.write_bytes(content)
                try:
                    loaded_docs = await asyncio.to_thread(
                        indexer.load_file,
                        temp_path,
                        source=storage_path,
                        metadata={
                            "storage_bucket": storage.bucket,
                            "storage_path": storage_path,
                            "user_id": user_id,
                        },
                    )
                except Exception as e:
                    logger.error(f"Document loading failed for {filename}: {e}", exc_info=True)
                    raise HTTPException(
                        status_code=422,
                        detail="Document loading failed. Check that the file is a valid PDF, TXT, or MD document.",
                    )
                if not loaded_docs:
                    raise HTTPException(
                        status_code=422,
                        detail="No readable text found in the uploaded document.",
                    )
                documents_to_index.extend(loaded_docs)
                pending_records.append(
                    {
                        "user_id": user_id,
                        "filename": filename,
                        "content_type": content_type,
                        "size_bytes": len(content),
                        "storage_bucket": storage.bucket,
                        "storage_path": storage_path,
                    }
                )

            try:
                indexed_chunks = await indexer.aindex_documents(documents_to_index) or 0
            except Exception as e:
                logger.error(f"Vector indexing failed: {e}", exc_info=True)
                raise HTTPException(
                    status_code=502,
                    detail="Vector indexing failed. Check Qdrant and embedding provider configuration.",
                )

        try:
            for record in pending_records:
                saved_files.append(
                    await asyncio.to_thread(create_document_record, pipeline.tracker, **record)
                )
        except Exception as e:
            logger.error(f"Document record save failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Document record save failed. Check DATABASE_URL and migrations.",
            )
        if hasattr(pipeline.retriever, "reload"):
            try:
                await pipeline.retriever.reload()
            except Exception as e:
                logger.warning(f"Retriever reload failed after document upload: {e}")

        return {
            "status": "success",
            "documents": saved_files,
            "total": len(saved_files),
            "stats": {**indexer.get_stats(), "indexed_chunks": indexed_chunks},
        }
    except HTTPException:
        await cleanup_uploaded_paths()
        raise
    except Exception as e:
        await cleanup_uploaded_paths()
        logger.error(f"Document upload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Document upload failed. Check storage, embeddings, vector database, and database configuration.",
        )


@v1.get("/documents")
async def list_documents(user_id: str = Depends(require_api_key)):
    """List active document metadata from Supabase-backed Postgres."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")

    docs = await asyncio.to_thread(list_active_documents, pipeline.tracker, user_id)
    return {"documents": docs, "total": len(docs)}


@v1.delete("/documents/{filename}")
async def delete_document(filename: str, user_id: str = Depends(require_api_key)):
    """Delete a stored document, purge vector points from Qdrant, and flush cache."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    from src.retrieval.indexer import DocumentIndexer

    try:
        document = await asyncio.to_thread(get_active_document, pipeline.tracker, filename, user_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document not found: {filename}")

        storage = SupabaseStorage.from_env()
        await storage.delete(document["storage_path"])

        indexer = DocumentIndexer()
        deleted = await indexer.adelete_document(
            filename,
            DOCUMENTS_DIR,
            source=document["storage_path"],
            user_id=user_id,
        )
        await asyncio.to_thread(mark_document_deleted, pipeline.tracker, document["storage_path"])
        if hasattr(pipeline.retriever, "reload"):
            await pipeline.retriever.reload()
        return {
            "status": "success",
            "filename": filename,
            "storage_path": document["storage_path"],
            "deleted": deleted,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deleting document {filename} failed: {e}")
        raise HTTPException(status_code=500, detail="Document deletion failed")


@v1.delete("/documents")
async def clear_all_documents(user_id: str = Depends(require_api_key)):
    """Clear stored documents, reset Qdrant collection, and flush cache."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    from src.retrieval.indexer import DocumentIndexer

    try:
        storage = SupabaseStorage.from_env()
        documents = await asyncio.to_thread(list_active_documents, pipeline.tracker, user_id)
        for document in documents:
            await storage.delete(document["storage_path"])
            await asyncio.to_thread(
                mark_document_deleted, pipeline.tracker, document["storage_path"]
            )

        indexer = DocumentIndexer()
        await indexer.aclear_all_documents(DOCUMENTS_DIR, user_id=user_id)
        if hasattr(pipeline.retriever, "reload"):
            await pipeline.retriever.reload()
        return {"status": "success", "message": "All documents cleared"}
    except Exception as e:
        logger.error(f"Clearing all documents failed: {e}")
        raise HTTPException(status_code=500, detail="Document clear failed")


# Mount versioned router — all /v1/* routes are now live
app.include_router(v1)

if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets")


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend(full_path: str):
    """Serve the compiled React app for client-side routes."""
    if full_path.startswith(("v1/", "docs", "openapi.json", "redoc", "health", "ready")):
        raise HTTPException(status_code=404, detail="Not found")

    requested_path = FRONTEND_DIST / full_path
    if requested_path.exists() and requested_path.is_file():
        return FileResponse(requested_path)

    index_path = FRONTEND_DIST / "index.html"
    if index_path.exists():
        return FileResponse(index_path)

    raise HTTPException(status_code=404, detail="Frontend build not found")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
