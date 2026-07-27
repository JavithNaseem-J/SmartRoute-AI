from typing import List, Optional, Tuple

from langchain_core.documents import Document
from qdrant_client import models

from src.core.dependencies import get_embeddings, get_qdrant_client, get_sparse_vector
from src.retrieval.reranker import DocumentReranker
from src.utils.logger import logger


class DocumentRetriever:
    """Handles document retrieval with Qdrant native hybrid search."""

    def __init__(
        self,
        collection_name: str = "smartroute_docs",
        top_k: int = 5,
    ):
        self.collection_name = collection_name
        self.top_k = top_k

        # Initialize components
        self.embeddings = get_embeddings()
        self.qdrant = get_qdrant_client()

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
            sparse_vector = get_sparse_vector(self.qdrant, query)

            if sparse_vector is not None:
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

                query_response = await self.qdrant.query_points(
                    collection_name=self.collection_name,
                    prefetch=prefetch,
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=k,
                    with_payload=True,
                )
                points = query_response.points
            else:
                query_response = await self.qdrant.query_points(
                    collection_name=self.collection_name,
                    query=vector,
                    using="dense",
                    limit=k,
                    with_payload=True,
                )
                points = query_response.points

            return [
                (
                    Document(
                        page_content=(r.payload.get("page_content", "") if r.payload else ""),
                        metadata=r.payload.get("metadata", {}) if r.payload else {},
                    ),
                    r.score,
                )
                for r in points
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

    async def _retrieve_hybrid(
        self, query: str, top_k: Optional[int] = None
    ) -> Tuple[str, List[str]]:
        """Retrieve using native Qdrant hybrid search with RRF Fusion."""
        logger.info("Using native Qdrant hybrid search")

        # Dynamically scale top_k for exhaustive/list queries ("all", "terms", "list", "what are")
        q_lower = query.lower()
        is_list_query = any(
            kw in q_lower
            for kw in ["all", "list", "terms", "components", "overview", "what are", "every"]
        )

        effective_k = top_k or (15 if is_list_query else self.top_k)

        # Fetch effective_k * 2 candidates from Qdrant
        results = await self._search_qdrant(query, effective_k * 2)

        candidate_docs = [doc for doc, _ in results]

        # Re-rank candidates against the query
        top_docs = await self.reranker.rerank(query, candidate_docs, top_k=effective_k)

        context_parts = []
        sources = []

        for i, doc in enumerate(top_docs):
            context_parts.append(f"[Source {i + 1}]\n{doc.page_content}")
            source = doc.metadata.get("source", "Unknown")
            sources.append(f"Source {i + 1}: {source}")

        logger.info(
            f"Hybrid search retrieved {len(top_docs)} documents (effective_k={effective_k})"
        )

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
            "vector_store": {"document_count": qdrant_count},
        }
