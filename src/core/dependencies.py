import os

import redis.asyncio as redis
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from qdrant_client import AsyncQdrantClient

# Singletons
_redis_client = None
_qdrant_client = None
_embeddings = None


def get_redis_client() -> redis.Redis:
    """Get the shared async Redis client."""
    global _redis_client
    if _redis_client is None:
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            raise RuntimeError("REDIS_URL environment variable is required")
        _redis_client = redis.from_url(redis_url, decode_responses=True)
    return _redis_client


def get_qdrant_client() -> AsyncQdrantClient:
    """Get the shared async Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_key = os.getenv("QDRANT_API_KEY")
        if not qdrant_url or not qdrant_key:
            raise RuntimeError("QDRANT_URL and QDRANT_API_KEY environment variables are required")
        _qdrant_client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_key)
    return _qdrant_client


def get_embeddings(model_name: str = "BAAI/bge-small-en-v1.5") -> HuggingFaceEndpointEmbeddings:
    """Get the shared endpoint embeddings model."""
    global _embeddings
    if _embeddings is None:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("HF_TOKEN environment variable is required")
        _embeddings = HuggingFaceEndpointEmbeddings(
            model=model_name,
            huggingfacehub_api_token=hf_token,
        )
    return _embeddings
