"""
SmartRoute-AI — FastAPI entry point.

Startup order:
  1. validate_env()       — fail fast if any required env var is missing
  2. InferencePipeline()  — connect to Qdrant, Supabase, Upstash
  3. Serve traffic        — all endpoints are async and non-blocking
"""

import os
import secrets
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.pipeline.inference import InferencePipeline
from src.utils.guardrails import GuardrailViolation, validate_query
from src.utils.logger import logger
from src.utils.tracing import setup_tracing

# ── Startup: env-var validation ───────────────────────────────────────────────

_REQUIRED_ENV_VARS = [
    ("DATABASE_URL", "Supabase PostgreSQL  → https://supabase.com"),
    ("REDIS_URL", "Upstash Redis        → https://upstash.com"),
    ("QDRANT_URL", "Qdrant Cloud         → https://cloud.qdrant.io"),
    ("QDRANT_API_KEY", "Qdrant Cloud         → https://cloud.qdrant.io"),
    ("GROQ_API_KEY", "Groq LLM             → https://console.groq.com"),
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
        lines.append(f"  ❌  {var}")
        lines.append(f"       Get it from: {hint}")
    lines += [
        "=" * 60,
        "Set these in your .env file or Render environment variables.\n",
    ]
    logger.error("\n".join(lines))
    sys.exit(1)


# ── Application lifespan ──────────────────────────────────────────────────────

pipeline: Optional[InferencePipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Runs validate_env + pipeline init before serving; disposes on shutdown."""
    global pipeline
    validate_env()
    try:
        pipeline = InferencePipeline()
        logger.info("Pipeline initialised — all cloud services connected.")
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
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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

# ── Authentication ────────────────────────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: Optional[str] = Depends(_api_key_header)) -> str:
    """Constant-time API key validation to prevent timing attacks."""
    valid_key = os.getenv("SMARTROUTE_API_KEY", "dev-key-change-in-production")
    if not api_key or not secrets.compare_digest(api_key, valid_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# ── Request / Response models ─────────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User query")
    strategy: Optional[str] = Field(
        None, description="Routing strategy: cost_optimized | quality_first | balanced"
    )
    use_retrieval: bool = Field(True, description="Enable RAG retrieval")
    session_id: Optional[str] = Field(
        None, description="Session ID for multi-turn conversation"
    )


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
            "query": "/query",
            "stream": "/query/stream",
            "stats": "/stats",
            "savings": "/savings",
            "budget": "/budget",
            "health": "/health",
        },
    }


@app.get("/health")
async def health():
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    return {
        "status": "healthy",
        "components": {
            "router": "ok" if pipeline.router else "error",
            "model_manager": "ok" if pipeline.model_manager else "error",
            "retriever": "ok" if pipeline.retriever else "error",
            "cost_tracker": "ok" if pipeline.tracker else "error",
        },
    }


# ── Protected endpoints ───────────────────────────────────────────────────────


@app.post("/query", response_model=QueryResponse)
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
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
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
        try:
            # Guardrails
            try:
                query = validate_query(query_request.query)
            except GuardrailViolation as e:
                yield f"data: [ERROR] {e}\n\n"
                return

            # Route
            routing_decision = pipeline.router.route(query, query_request.strategy)
            model_id = routing_decision["model_id"]

            # Retrieve context (run blocking call in executor)
            import asyncio

            context = ""
            if query_request.use_retrieval:
                loop = asyncio.get_event_loop()
                context, _ = await loop.run_in_executor(
                    None, pipeline.retriever.retrieve, query
                )

            model = pipeline.model_manager.load_model(model_id)
            history = (
                pipeline.memory.get_history(query_request.session_id)
                if query_request.session_id
                else []
            )

            # Stream tokens from the async generator
            async for chunk in model.astream(
                prompt=query,
                context=context,
                max_tokens=1000,
                temperature=0.7,
                history=history,
            ):
                yield f"data: {chunk}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Stream failed: {e}")
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(
        _token_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/stats")
@limiter.limit("60/minute")
async def get_stats(
    request: Request,
    days: int = 1,
    _: str = Depends(require_api_key),
):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    return pipeline.tracker.get_statistics(days=days)


@app.get("/savings")
@limiter.limit("60/minute")
async def get_savings(
    request: Request,
    days: int = 1,
    _: str = Depends(require_api_key),
):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    return pipeline.tracker.calculate_savings(days=days)


@app.get("/budget")
@limiter.limit("60/minute")
async def get_budget(
    request: Request,
    _: str = Depends(require_api_key),
):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    return pipeline.budget_manager.get_budget_status()


@app.get("/models")
async def list_models(_: str = Depends(require_api_key)):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {
        "available": pipeline.model_manager.list_available_models(),
        "loaded": pipeline.model_manager.list_loaded_models(),
    }


@app.delete("/memory/{session_id}")
async def clear_memory(session_id: str, _: str = Depends(require_api_key)):
    """Clear conversation history for a session."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    pipeline.memory.clear(session_id)
    return {"status": "cleared", "session_id": session_id}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
