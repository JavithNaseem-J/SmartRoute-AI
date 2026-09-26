import os
from asyncio import to_thread
from collections.abc import Iterable
from threading import Lock
from typing import SupportsFloat, cast

import redis.asyncio as redis
from fastembed import TextEmbedding
from qdrant_client import AsyncQdrantClient

_redis_client = None
_qdrant_client = None
_embeddings = None


class FastEmbedEmbeddings:
    """Async-compatible local dense embeddings backed by Qdrant FastEmbed."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model: TextEmbedding | None = None
        self._model_lock = Lock()

    def _get_model(self) -> TextEmbedding:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = TextEmbedding(
                        model_name=self.model_name,
                        cache_dir=os.getenv("FASTEMBED_CACHE_PATH") or None,
                        threads=1,
                        lazy_load=True,
                    )
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._get_model().passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        vectors = cast(Iterable[Iterable[SupportsFloat]], self._get_model().query_embed(text))
        return [float(value) for value in next(iter(vectors))]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await to_thread(self.embed_query, text)


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
        enable_sparse = os.getenv("ENABLE_SPARSE_EMBEDDINGS", "false").lower() == "true"
        if enable_sparse:
            try:
                _qdrant_client.set_sparse_model("prithivida/Splade_PP_en_v1", threads=1)
            except Exception as e:
                from src.utils.logger import logger

                logger.warning(f"Could not set sparse model, sparse vectors will be disabled: {e}")
    return _qdrant_client


def get_embeddings(
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> FastEmbedEmbeddings:
    """Get the shared local FastEmbed model."""
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings(model_name=model_name)
    return _embeddings


def get_sparse_vector(client: AsyncQdrantClient, text: str):
    """Safely extract a sparse query vector if the sparse embedding model is loaded on the Qdrant client."""
    if hasattr(client, "_sparse_embedding_model") and client._sparse_embedding_model is not None:  # type: ignore[attr-defined]
        try:
            return next(client._sparse_embedding_model.query_embed(text))  # type: ignore[attr-defined]
        except Exception:
            return None
    return None
