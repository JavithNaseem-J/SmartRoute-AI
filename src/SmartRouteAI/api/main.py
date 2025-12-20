import time
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from src.pipeline.inference import HybridInferencePipeline
from src.cost.budget import BudgetManager
from src.cost.reporter import CostReporter
from src.utils.logging import logger

# Initialize FastAPI
app = FastAPI(
    title="Cost-Optimized RAG API",
    description="HYBRID: LangChain + Custom routing for cost optimization",
    version="1.0.0"
)

# CORS
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
try:
    pipeline = HybridInferencePipeline()
    budget_manager = BudgetManager()
    reporter = CostReporter()
    logger.info("✓ API components initialized")
except Exception as e:
    logger.error(f"Failed to initialize: {e}")
    pipeline = None
    budget_manager = None
    reporter = None


# Request/Response models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    strategy: Optional[str] = None
    use_retrieval: bool = True
    collection_name: str = "documents"


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
    """Health check"""
    return {
        "status": "healthy" if pipeline else "degraded",
        "service": "Cost-Optimized RAG API",
        "version": "1.0.0",
        "approach": "HYBRID (LangChain + Custom)"
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query through the RAG pipeline"""
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        result = pipeline.run(
            query=request.query,
            strategy=request.strategy,
            use_retrieval=request.use_retrieval,
            collection_name=request.collection_name
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/budget")
async def get_budget():
    """Get current budget status"""
    
    if not budget_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        status = budget_manager.get_budget_status()
        return status
        
    except Exception as e:
        logger.error(f"Budget check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats(days: int = 1):
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
async def get_savings(days: int = 1):
    """Get cost savings vs baseline"""
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        savings = pipeline.tracker.calculate_savings(days=days)
        return savings
        
    except Exception as e:
        logger.error(f"Savings calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/report/daily")
async def daily_report():
    """Get daily cost report"""
    
    if not reporter:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        report = reporter.generate_daily_report()
        return {"report": report}
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/report/weekly")
async def weekly_report():
    """Get weekly cost report"""
    
    if not reporter:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        report = reporter.generate_weekly_report()
        return {"report": report}
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List all available models"""
    
    try:
        from src.utils.config import load_model_config
        config = load_model_config()
        
        models = {
            "local": list(config.get("local_models", {}).keys()),
            "api": list(config.get("api_models", {}).keys())
        }
        
        return models
        
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
    uvicorn.run(app, host="0.0.0.0", port=8000)