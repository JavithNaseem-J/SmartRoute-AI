from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from qdrant_client import models

from src.core.dependencies import get_embeddings, get_qdrant_client
from src.retrieval.reranker import DocumentReranker
from src.utils.logger import logger

_executor = ThreadPoolExecutor(max_workers=4)


class DocumentRetriever:
    """Handles document retrieval with Qdrant native hybrid search."""

    def __init__(
        self,
        persist_dir: Path = Path("data/embeddings"),
        collection_name: str = "smartroute_docs",
        top_k: int = 5,
        max_distance: float = 1.5,
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.top_k = top_k
        self.max_distance = max_distance

        # Initialize components
        self.embeddings = get_embeddings()
        self.qdrant = get_qdrant_client()

        try:
            self.qdrant.set_sparse_model("prithivida/Splade_PP_en_v1")
        except Exception as e:
            logger.warning(f"Could not set sparse model, sparse vectors will be disabled: {e}")

        # Re-ranker for post-retrieval relevance filtering
        self.reranker = DocumentReranker()

        logger.info("####### DocumentRetriever initialized #######")

    async def ensure_ready(self):
        self.dense_ready = await self.qdrant.collection_exists(self.collection_name)

    async def reload(self) -> None:
        """Reload all indexes (call after new documents are added)."""
        logger.info("Reloading retrieval indexes...")
        await self.ensure_ready()

    async def _search_qdrant(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Perform native hybrid search using Qdrant client (RRF)."""
        vector = await self.embeddings.aembed_query(query)

        try:
            # Check if fastembed sparse model is loaded
            sparse_supported = (
                hasattr(self.qdrant, "_sparse_embedding_model")
                and self.qdrant._sparse_embedding_model is not None
            )

            if sparse_supported:
                # Qdrant client native method for generating sparse queries
                sparse_vector = next(self.qdrant._sparse_embedding_model.query_embed(query))

                prefetch = [
                    models.Prefetch(
                        query=vector,
                        using="dense",
                        limit=k,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_vector.indices.tolist(),
                            values=sparse_vector.values.tolist(),
                        ),
                        using="sparse",
                        limit=k,
                    ),
                ]

                results = await self.qdrant.query_points(
                    collection_name=self.collection_name,
                    prefetch=prefetch,
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=k,
                    with_payload=True,
                )
                results = results.points
            else:
                results = await self.qdrant.search(
                    collection_name=self.collection_name,
                    query_vector=("dense", vector),
                    limit=k,
                    with_payload=True,
                )

            return [
                (
                    Document(
                        page_content=r.payload.get("page_content", ""),
                        metadata=r.payload.get("metadata", {}),
                    ),
                    r.score,
                )
                for r in results
            ]
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []

    async def retrieve(self, query: str) -> Tuple[str, List[str]]:
        if not hasattr(self, "dense_ready"):
            await self.ensure_ready()

        if not self.dense_ready:
            logger.warning("No vector store available")
            return "", []

        try:
            context, sources = await self._retrieve_hybrid(query)
            return context, sources
        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            return "", []

    async def _retrieve_hybrid(self, query: str) -> Tuple[str, List[str]]:
        """Retrieve using native Qdrant hybrid search with RRF Fusion."""
        logger.info("Using native Qdrant hybrid search")

        # Fetch top_k * 2 candidates from Qdrant
        results = await self._search_qdrant(query, self.top_k * 2)

        candidate_docs = [doc for doc, _ in results]

        # Re-rank candidates against the query
        top_docs = await self.reranker.rerank(query, candidate_docs, top_k=self.top_k)

        context_parts = []
        sources = []

        for i, doc in enumerate(top_docs):
            context_parts.append(f"[Source {i+1}]\n{doc.page_content}")
            source = doc.metadata.get("source", "Unknown")
            sources.append(f"Source {i+1}: {source}")

        logger.info(f"Hybrid search retrieved {len(top_docs)} documents")

        context = "\n\n".join(context_parts)
        return context, sources

    @property
    def retrieval_mode(self) -> str:
        """Get current retrieval mode."""
        dense = getattr(self, "dense_ready", False)
        return "native_hybrid" if dense else "unavailable"

    async def get_stats(self) -> dict:
        """Get retriever statistics."""
        dense = getattr(self, "dense_ready", False)
        qdrant_count = 0
        if dense:
            try:
                qdrant_count = (await self.qdrant.count(self.collection_name)).count
            except Exception:
                pass

        return {
            "status": "loaded" if dense else "not_loaded",
            "retrieval_mode": self.retrieval_mode,
            "top_k": self.top_k,
            "max_distance": self.max_distance,
            "vector_store": {"document_count": qdrant_count},
        }
