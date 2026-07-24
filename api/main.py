import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional

from dotenv import load_dotenv

load_dotenv()

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.pipeline.inference import InferencePipeline
from src.utils.logger import logger
from src.utils.tracing import setup_tracing

#  validation

_REQUIRED_ENV_VARS = [
    ("DATABASE_URL", "Supabase PostgreSQL  -> https://supabase.com"),
    ("REDIS_URL", "Upstash Redis        -> https://upstash.com"),
    ("QDRANT_URL", "Qdrant Cloud         -> https://cloud.qdrant.io"),
    ("QDRANT_API_KEY", "Qdrant Cloud         -> https://cloud.qdrant.io"),
    ("HF_TOKEN", "HuggingFace API      -> https://huggingface.co/settings/tokens"),
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
        sys.exit(1)  # crash loudly; Render will restart and show the error
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

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8501,http://localhost:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

setup_tracing(app)


@app.get("/health", tags=["system"])
async def health_check():
    """Unauthenticated health probe for Docker, Render, and load balancers.

    Always returns 200 so Docker probes don't restart the container during
    the 60-second pipeline startup. Returns component detail once ready.
    """
    if not pipeline:
        return {"status": "starting", "pipeline": "initializing"}

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
        components["redis"] = f"error: {str(e)}"

    try:
        from src.core.dependencies import get_qdrant_client

        qdrant_client = get_qdrant_client()
        await qdrant_client.get_collections()
        components["qdrant"] = "ok"
    except Exception as e:
        components["qdrant"] = f"error: {str(e)}"

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
        components["postgres"] = f"error: {str(e)}"

    return {
        "status": "healthy",
        "version": "2.0.0",
        "components": components,
    }


# ── Authentication ────────────────────────────────────────────────────────────

from src.utils.security import require_jwt


def require_api_key(payload: dict = Depends(require_jwt)) -> str:
    """JWT validation facade. Returns the user ID (sub) from the token."""
    return payload.get("sub", "unknown_user")


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


# ── Public endpoints ──────────────────────────────────────────────────────────


@app.get("/")
async def root():
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
            "docs": "/docs",
        },
    }


# ── Versioned router ──────────────────────────────────────────────────────
#
# All business endpoints go on v1_router so we can introduce /v2 later
# without touching existing client code. The prefix is injected at mount time.

v1 = APIRouter(prefix="/v1", tags=["v1"])


@v1.post("/query", response_model=QueryResponse)
@limiter.limit("30/minute")
async def query(
    request: Request,
    query_request: QueryRequest,
    _: str = Depends(require_api_key),
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
        raise HTTPException(status_code=500, detail=str(e))


@v1.post("/query/batch")
@limiter.limit("10/minute")
async def query_batch(
    request: Request,
    queries: List[str],
    strategy: Optional[str] = None,
    use_retrieval: bool = True,
    _: str = Depends(require_api_key),
):
    """Process multiple queries concurrently. Max 10 queries per call."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    if not queries:
        raise HTTPException(status_code=422, detail="queries list cannot be empty")
    if len(queries) > 10:
        raise HTTPException(status_code=422, detail="Maximum 10 queries per batch request")
    results = await pipeline.batch_run(
        queries=queries, strategy=strategy, use_retrieval=use_retrieval
    )
    return results


@v1.post("/query/stream")
@limiter.limit("30/minute")
async def query_stream(
    request: Request,
    query_request: QueryRequest,
    _: str = Depends(require_api_key),
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
            ):
                yield f"data: {json.dumps(item)}\n\n"
        except Exception as e:
            logger.error(f"Stream failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

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
async def clear_memory(session_id: str, _: str = Depends(require_api_key)):
    """Clear conversation history for a session."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    await pipeline.memory.clear(session_id)
    return {"status": "cleared", "session_id": session_id}


@v1.post("/index")
async def index_documents(_: str = Depends(require_api_key)):
    """Trigger indexing of documents in the data/documents directory."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    from src.retrieval.indexer import DocumentIndexer

    try:
        indexer = DocumentIndexer()
        # Run synchronous indexing in a thread
        await asyncio.to_thread(indexer.index_directory, "data/documents")
        # Reload the retriever to pick up new documents
        if hasattr(pipeline.retriever, "reload"):
            await asyncio.to_thread(pipeline.retriever.reload)
        stats = indexer.get_stats()
        return {"status": "success", "stats": stats}
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Mount versioned router — all /v1/* routes are now live
app.include_router(v1)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
