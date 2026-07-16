import sys
import os
import secrets
import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Optional, List, Iterator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

sys.path.append(str(Path(__file__).parent.parent))
from src.pipeline.inference import InferencePipeline
from src.utils.logger import logger

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI
app = FastAPI(
    title="Cost-Optimized RAG API",
    description="Smart routing for cost-effective LLM inference",
    version="1.0.0"
)

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", 
    "http://localhost:8501,http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

# Initialize pipeline
try:
    pipeline = InferencePipeline()
    logger.info("####### Pipeline initialized #######")
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {e}")
    pipeline = None

# Optional OpenTelemetry tracing — activates when OTEL_EXPORTER_OTLP_ENDPOINT is set
from src.utils.tracing import setup_tracing
setup_tracing(app)


# --- Authentication ---
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: Optional[str] = Depends(_api_key_header)) -> str:
    """FastAPI dependency — validates X-API-Key header on protected endpoints.

    Uses secrets.compare_digest instead of == to prevent timing attacks:
    == short-circuits on first mismatch, leaking information about how many
    leading characters were correct. compare_digest always runs in constant time.

    Public endpoints (/, /health) do NOT include this dependency so that
    load balancer readiness probes work without credentials.
    """
    valid_key = os.getenv("SMARTROUTE_API_KEY", "dev-key-change-in-production")
    if not api_key or not secrets.compare_digest(api_key, valid_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# Request/Response models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User query")
    strategy: Optional[str] = Field(None, description="Routing strategy (cost_optimized, quality_first, balanced)")
    use_retrieval: bool = Field(True, description="Use RAG retrieval")
    session_id: Optional[str] = Field(
        None,
        description="Session ID for multi-turn conversation. Omit for stateless single-turn queries."
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


@app.get("/")
async def root():
    """Health check and API info"""
    return {
        "status": "healthy" if pipeline else "degraded",
        "service": "Cost-Optimized RAG API",
        "version": "1.0.0",
        "endpoints": {
            "query": "/query",
            "stats": "/stats",
            "savings": "/savings",
            "budget": "/budget",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Check components
        router_ok = pipeline.router is not None
        model_manager_ok = pipeline.model_manager is not None
        retriever_ok = pipeline.retriever is not None
        tracker_ok = pipeline.tracker is not None
        
        return {
            "status": "healthy",
            "components": {
                "router": "ok" if router_ok else "error",
                "model_manager": "ok" if model_manager_ok else "error",
                "retriever": "ok" if retriever_ok else "error",
                "cost_tracker": "ok" if tracker_ok else "error"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
@limiter.limit("30/minute")
def query(
    request: Request,
    query_request: QueryRequest,
    _: str = Depends(require_api_key)
):
    """Process a query through the pipeline.

    Uses a plain `def` (not async) so FastAPI dispatches this to a
    threadpool via run_in_executor. This keeps the event loop free to
    accept new connections while the blocking pipeline.run() executes.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        logger.info(f"Processing query: {query_request.query[:50]}...")

        result = pipeline.run(
            query=query_request.query,
            strategy=query_request.strategy,
            use_retrieval=query_request.use_retrieval,
            session_id=query_request.session_id,
        )

        return QueryResponse(**result)

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
@limiter.limit("60/minute")
def get_stats(
    request: Request,
    days: int = 1,
    _: str = Depends(require_api_key)
):
    """Get cost statistics"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        stats = pipeline.tracker.get_statistics(days=days)
        return stats

    except Exception as e:
        logger.error(f"Stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/savings")
@limiter.limit("60/minute")
def get_savings(
    request: Request,
    days: int = 1,
    _: str = Depends(require_api_key)
):
    """Calculate cost savings vs baseline"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        savings = pipeline.tracker.calculate_savings(days=days)
        return savings

    except Exception as e:
        logger.error(f"Savings calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/budget")
@limiter.limit("60/minute")
def get_budget(
    request: Request,
    _: str = Depends(require_api_key)
):
    """Get current budget status"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        status = pipeline.budget_manager.get_budget_status()
        return status

    except Exception as e:
        logger.error(f"Budget check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
def list_models(
    _: str = Depends(require_api_key)
):
    """List available models"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        available = pipeline.model_manager.list_available_models()
        loaded = pipeline.model_manager.list_loaded_models()

        return {
            "available": available,
            "loaded": loaded
        }

    except Exception as e:
        logger.error(f"Models list failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/strategy")
def update_strategy(
    request: Request,
    strategy: str,
    _: str = Depends(require_api_key)
):
    """Update default routing strategy"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        pipeline.router.update_strategy(strategy)
        return {"status": "success", "strategy": strategy}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Strategy update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/{session_id}")
def clear_memory(
    session_id: str,
    _: str = Depends(require_api_key)
):
    """Clear conversation history for a session.

    Call this when the user ends a conversation to free memory immediately.
    Sessions also expire automatically after 30 minutes of inactivity.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    pipeline.memory.clear(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.post("/query/stream")
@limiter.limit("30/minute")
def query_stream(
    request: Request,
    query_request: QueryRequest,
    _: str = Depends(require_api_key)
):
    """Stream the LLM response as Server-Sent Events (SSE).

    Returns tokens incrementally as `data: <chunk>\n\n` events.
    The client sees the first token within ~200ms instead of waiting
    ~1.5s for the full response.

    SSE format per W3C spec:
        data: <text chunk>\n\n
        data: [DONE]\n\n  (signals end of stream)
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")

    def _token_generator() -> Iterator[str]:
        try:
            # Run routing and retrieval first (blocking, in this thread)
            from src.utils.guardrails import validate_query, GuardrailViolation
            try:
                query = validate_query(query_request.query)
            except GuardrailViolation as e:
                yield f"data: [ERROR] {e}\n\n"
                return

            routing_decision = pipeline.router.route(query, query_request.strategy)
            model_id = routing_decision['model_id']

            context = ""
            if query_request.use_retrieval:
                context, _ = pipeline.retriever.retrieve(query)

            model = pipeline.model_manager.load_model(model_id)

            # Stream tokens
            for chunk in model.generate_stream(
                prompt=query,
                context=context,
                max_tokens=1000,
                temperature=0.7
            ):
                yield f"data: {chunk}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Stream failed: {e}")
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(
        _token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"  # Disable nginx buffering for true streaming
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)