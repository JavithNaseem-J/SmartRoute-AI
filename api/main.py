import sys
import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
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

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize pipeline
try:
    pipeline = InferencePipeline()
    logger.info("✓ Pipeline initialized")
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {e}")
    pipeline = None


# Request/Response models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User query")
    strategy: Optional[str] = Field(None, description="Routing strategy (cost_optimized, quality_first, balanced)")
    use_retrieval: bool = Field(True, description="Use RAG retrieval")


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
async def query(request: Request, query_request: QueryRequest):
    """Process a query through the pipeline"""
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        logger.info(f"Processing query: {query_request.query[:50]}...")
        
        result = pipeline.run(
            query=query_request.query,
            strategy=query_request.strategy,
            use_retrieval=query_request.use_retrieval
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
@limiter.limit("60/minute")
async def get_stats(request: Request, days: int = 1):
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
async def get_savings(request: Request, days: int = 1):
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
async def get_budget(request: Request):
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
async def list_models():
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
async def update_strategy(strategy: str):
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


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)