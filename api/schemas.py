from typing import List, Optional, TypedDict

from pydantic import BaseModel, Field


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
