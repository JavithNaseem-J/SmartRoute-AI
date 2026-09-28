import os
import re
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from qdrant_client import models

from src.core.dependencies import get_embeddings, get_qdrant_client, get_sparse_vector
from src.retrieval.reranker import DocumentReranker
from src.retrieval.types import Citation
from src.utils.logger import logger


class DocumentRetriever:
    """Handle dense or explicitly enabled hybrid retrieval with Qdrant."""

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
        self.sparse_enabled = os.getenv("ENABLE_SPARSE_EMBEDDINGS", "false").lower() == "true"

        # Re-ranker for post-retrieval relevance filtering
        self.reranker = DocumentReranker()
        self.last_diagnostics: dict = {}
        self.last_citations: List[Citation] = []

        logger.info("####### DocumentRetriever initialized #######")

    async def ensure_ready(self):
        try:
            self.dense_ready = await self.qdrant.collection_exists(self.collection_name)
        except Exception as e:
            logger.warning(f"Qdrant collection check failed: {e}. Vector retrieval disabled.")
            self.dense_ready = False

    async def reload(self) -> None:
        """Reload all indexes (call after new documents are added)."""
        logger.info("Reloading retrieval indexes...")
        await self.ensure_ready()

    @staticmethod
    def _user_filter(
        user_id: Optional[str], active_sources: Optional[List[str]] = None
    ) -> Optional[models.Filter]:
        if not user_id:
            return None
        conditions: List[models.Condition] = [
            models.FieldCondition(
                key="metadata.user_id",
                match=models.MatchValue(value=user_id),
            )
        ]
        if active_sources is not None:
            conditions.append(
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchAny(any=active_sources),
                )
            )
        return models.Filter(must=conditions)

    @staticmethod
    def _citation_for_document(document: Document, index: int) -> Citation:
        metadata = document.metadata
        filename = str(metadata.get("filename") or metadata.get("source") or "Unknown")
        filename = filename.replace("\\", "/").rsplit("/", 1)[-1]

        page: Optional[int] = None
        if metadata.get("page") is not None:
            try:
                page = int(metadata["page"]) + 1
            except (TypeError, ValueError):
                page = None

        section_value = metadata.get("section")
        section = str(section_value).strip() if section_value else None
        excerpt = re.sub(r"\s+", " ", document.page_content).strip()
        if len(excerpt) > 280:
            excerpt = f"{excerpt[:277].rstrip()}..."

        return {
            "id": f"C{index}",
            "filename": filename,
            "page": page,
            "section": section,
            "excerpt": excerpt,
        }

    async def _search_qdrant(
        self,
        query: str,
        k: int,
        user_id: Optional[str] = None,
        active_sources: Optional[List[str]] = None,
    ) -> List[Tuple[Document, float]]:
        """Search Qdrant in dense mode, or fuse dense and sparse results with RRF."""
        vector = await self.embeddings.aembed_query(query)
        query_filter = self._user_filter(user_id, active_sources)

        try:
            sparse_vector = get_sparse_vector(self.qdrant, query) if self.sparse_enabled else None

            if self.sparse_enabled and sparse_vector is not None:
                self.last_diagnostics["retrieval_mode"] = "hybrid"
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
                    query_filter=query_filter,
                    limit=k,
                    with_payload=True,
                )
                points = query_response.points
            else:
                self.last_diagnostics["retrieval_mode"] = "dense"
                if self.sparse_enabled:
                    self.last_diagnostics["hybrid_fallback_reason"] = "sparse_vector_unavailable"
                query_response = await self.qdrant.query_points(
                    collection_name=self.collection_name,
                    query=vector,
                    using="dense",
                    query_filter=query_filter,
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
            self.last_diagnostics["search_error"] = str(e)
            return []

    async def count_user_chunks(
        self, user_id: str, active_sources: Optional[List[str]] = None
    ) -> int:
        """Count indexed document chunks for a user."""
        if not await self.qdrant.collection_exists(self.collection_name):
            return 0
        result = await self.qdrant.count(
            collection_name=self.collection_name,
            count_filter=self._user_filter(user_id, active_sources),
            exact=True,
        )
        return int(result.count)

    async def retrieve(
        self,
        query: str,
        user_id: Optional[str] = None,
        active_sources: Optional[List[str]] = None,
    ) -> Tuple[str, List[str]]:
        try:
            self.last_citations = []
            self.last_diagnostics = {
                "reason": None,
                "collection_ready": False,
                "user_chunk_count": None,
                "retrieved_source_count": 0,
                "retrieval_mode": self.retrieval_mode,
            }
            if not user_id:
                logger.warning("No user ID provided; document retrieval disabled")
                self.last_diagnostics["reason"] = "missing_user_id"
                return "", []

            if active_sources == []:
                logger.info("No active documents are available for this user")
                self.last_diagnostics["reason"] = "no_active_documents"
                self.last_diagnostics["user_chunk_count"] = 0
                return "", []

            if not hasattr(self, "dense_ready"):
                await self.ensure_ready()

            self.last_diagnostics["collection_ready"] = bool(self.dense_ready)
            if not self.dense_ready:
                logger.warning("No vector store available")
                self.last_diagnostics["reason"] = "collection_unavailable"
                return "", []

            try:
                self.last_diagnostics["user_chunk_count"] = await self.count_user_chunks(
                    user_id, active_sources
                )
            except Exception as e:
                logger.warning(f"Could not count indexed chunks for user {user_id}: {e}")
                self.last_diagnostics["count_error"] = str(e)

            context, sources = await self._retrieve_candidates(
                query, user_id=user_id, active_sources=active_sources
            )
            self.last_diagnostics["retrieved_source_count"] = len(sources)
            if not sources:
                if self.last_diagnostics.get("search_error"):
                    self.last_diagnostics["reason"] = "search_failed"
                elif self.last_diagnostics.get("user_chunk_count") == 0:
                    self.last_diagnostics["reason"] = "no_user_chunks"
                else:
                    self.last_diagnostics["reason"] = "no_matching_chunks"
                logger.warning(f"Document retrieval returned no sources: {self.last_diagnostics}")
            else:
                self.last_diagnostics["reason"] = "matched_sources"
            return context, sources
        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            self.last_diagnostics["reason"] = "retrieval_exception"
            self.last_diagnostics["error"] = str(e)
            return "", []

    async def _retrieve_candidates(
        self,
        query: str,
        top_k: Optional[int] = None,
        user_id: Optional[str] = None,
        active_sources: Optional[List[str]] = None,
    ) -> Tuple[str, List[str]]:
        """Retrieve and rerank candidates using the configured retrieval mode."""
        logger.info(f"Using Qdrant {self.retrieval_mode} retrieval")

        # Dynamically scale top_k for exhaustive/list queries ("all", "terms", "list", "what are")
        q_lower = query.lower()
        is_list_query = any(
            kw in q_lower
            for kw in ["all", "list", "terms", "components", "overview", "what are", "every"]
        )

        effective_k = top_k or (15 if is_list_query else self.top_k)

        # Fetch effective_k * 2 candidates from Qdrant
        results = await self._search_qdrant(
            query, effective_k * 2, user_id=user_id, active_sources=active_sources
        )

        candidate_docs = [doc for doc, _ in results]

        # Re-rank candidates against the query
        top_docs = await self.reranker.rerank(query, candidate_docs, top_k=effective_k)
        self.last_diagnostics["reranker_mode"] = self.reranker.last_mode

        context_parts = []
        sources = []
        citations: List[Citation] = []

        for i, doc in enumerate(top_docs):
            citation = self._citation_for_document(doc, i + 1)
            citations.append(citation)
            location = citation["filename"]
            if citation["page"] is not None:
                location = f"{location}, page {citation['page']}"
            elif citation["section"]:
                location = f"{location}, section {citation['section']}"
            context_parts.append(f"[{citation['id']}] {location}\n{doc.page_content}")

            source = citation["filename"]
            if citation["page"] is not None:
                source = f"{source} - page {citation['page']}"
            elif citation["section"]:
                source = f"{source} - section {citation['section']}"
            sources.append(f"Source {i + 1}: {source}")

        self.last_citations = citations

        logger.info(
            "%s search retrieved %s documents (effective_k=%s)",
            self.last_diagnostics.get("retrieval_mode", "dense").title(),
            len(top_docs),
            effective_k,
        )

        context = "\n\n".join(context_parts)
        return context, sources

    @property
    def retrieval_mode(self) -> str:
        """Get current retrieval mode."""
        dense = getattr(self, "dense_ready", False)
        if not dense:
            return "unavailable"
        sparse_ready = bool(
            self.sparse_enabled
            and getattr(self.qdrant, "_sparse_embedding_model", None) is not None
        )
        return "hybrid" if sparse_ready else "dense"

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
