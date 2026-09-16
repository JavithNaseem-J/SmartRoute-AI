import uuid
from pathlib import Path
from typing import List, Optional, cast

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from qdrant_client import models

from src.core.dependencies import get_embeddings, get_qdrant_client, get_redis_client
from src.retrieval.chunking import DocumentChunker
from src.utils.logger import logger


class DocumentIndexer:
    """Orchestrates document loading, chunking, and indexing into Qdrant."""

    def __init__(
        self,
        persist_dir: Path = Path("data/embeddings"),
        collection_name: str = "smartroute_docs",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name

        self.embeddings = get_embeddings()
        self.qdrant = get_qdrant_client()
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        logger.info(f"DocumentIndexer initialized: {collection_name}")

    async def _ensure_collection(self, vector_size: int):
        """Ensure collection exists with both dense and sparse configurations."""
        exists = await self.qdrant.collection_exists(self.collection_name)
        if not exists:
            vectors_config = {
                "dense": models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            }
            sparse_vectors_config = {"sparse": models.SparseVectorParams()}
            await self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
            )
            logger.info(f"Created new collection: {self.collection_name}")

    def load_file(
        self,
        file_path: Path,
        *,
        source: str,
        metadata: Optional[dict] = None,
    ) -> List[Document]:
        """Load one supported file and replace local loader source with durable metadata."""
        file_path = Path(file_path)
        loader_map = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".md": TextLoader,
        }
        loader_cls = loader_map.get(file_path.suffix.lower())
        if not loader_cls:
            return []

        documents = cast(List[Document], loader_cls(str(file_path)).load())
        for document in documents:
            document.metadata.update(metadata or {})
            document.metadata["source"] = source
            document.metadata["filename"] = file_path.name
        return documents

    async def aindex_documents(self, documents: List[Document]) -> int:
        """Async index documents into vector store (safe on running event loop)."""
        if not documents:
            return 0

        chunks = self.chunker.chunk_documents(documents)
        logger.info(f"Chunked into {len(chunks)} chunks")

        try:
            await self._async_add_documents(chunks)
        except Exception as e:
            logger.error(f"Vector indexing to Qdrant failed: {e}", exc_info=True)
            raise RuntimeError("Vector indexing failed") from e
        return len(chunks)

    async def _async_add_documents(self, chunks: List[Document]):
        texts = [doc.page_content for doc in chunks]

        # 1. Generate Dense Vectors
        dense_vectors = await self.embeddings.aembed_documents(texts)

        # Ensure collection exists with proper schema
        if dense_vectors:
            await self._ensure_collection(len(dense_vectors[0]))

        # 2. Generate Sparse Vectors if available
        sparse_supported = (
            hasattr(self.qdrant, "_sparse_embedding_model")
            and self.qdrant._sparse_embedding_model is not None  # type: ignore[attr-defined]
        )
        if sparse_supported:
            sparse_vectors_generator = self.qdrant._sparse_embedding_model.embed(texts)  # type: ignore[attr-defined]
            sparse_vectors_list = list(sparse_vectors_generator)

            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": d_vec,
                        "sparse": models.SparseVector(
                            indices=s_vec.indices.tolist(),
                            values=s_vec.values.tolist(),
                        ),
                    },
                    payload={
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    },
                )
                for doc, d_vec, s_vec in zip(chunks, dense_vectors, sparse_vectors_list)
            ]
        else:
            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={"dense": vec},
                    payload={
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    },
                )
                for doc, vec in zip(chunks, dense_vectors)
            ]

        await self.qdrant.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Added {len(chunks)} document chunks to index")

    async def _invalidate_user_cache(self, user_id: Optional[str]) -> None:
        if not user_id:
            return
        try:
            if await self.qdrant.collection_exists("semantic-cache"):
                await self.qdrant.delete(
                    collection_name="semantic-cache",
                    points_selector=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="user_id",
                                match=models.MatchValue(value=user_id),
                            )
                        ]
                    ),
                )
            redis = get_redis_client()
            async for key in redis.scan_iter(match=f"semantic_cache:{user_id}:*"):
                await redis.delete(key)
            logger.info(f"Invalidated semantic cache for user {user_id}.")
        except Exception as e:
            logger.warning(f"Failed to invalidate semantic cache for user {user_id}: {e}")

    async def adelete_document(
        self,
        filename: str,
        source: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """Purge a document's vectors from Qdrant and Redis cache."""
        deleted = False

        try:
            if await self.qdrant.collection_exists(self.collection_name):
                source_conditions: List[models.Condition] = [
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchValue(value=source),
                    )
                    if source
                    else models.FieldCondition(
                        key="metadata.filename",
                        match=models.MatchValue(value=filename),
                    ),
                ]
                must_conditions: List[models.Condition] = []
                if user_id:
                    must_conditions.append(
                        models.FieldCondition(
                            key="metadata.user_id",
                            match=models.MatchValue(value=user_id),
                        )
                    )
                await self.qdrant.delete(
                    collection_name=self.collection_name,
                    points_selector=models.Filter(
                        must=must_conditions or None,
                        should=source_conditions,
                    ),
                )
                logger.info(f"Purged vector points for document: {filename}")
                deleted = True
        except Exception as e:
            logger.error(f"Error purging vectors for {filename}: {e}")

        await self._invalidate_user_cache(user_id)

        return deleted

    async def aclear_all_documents(self, user_id: Optional[str] = None) -> None:
        """Clear document vectors from Qdrant and flush Redis cache."""
        try:
            if await self.qdrant.collection_exists(self.collection_name):
                points_selector = (
                    models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.user_id",
                                match=models.MatchValue(value=user_id),
                            )
                        ]
                    )
                    if user_id
                    else models.Filter()
                )
                await self.qdrant.delete(
                    collection_name=self.collection_name,
                    points_selector=points_selector,
                )
                logger.info(f"Deleted document vectors for user {user_id or 'all users'}")
        except Exception as e:
            logger.error(f"Error deleting Qdrant vectors: {e}")

        await self._invalidate_user_cache(user_id)

    def get_stats(self) -> dict:
        """Get indexer statistics."""
        return {
            "chunker": self.chunker.get_config(),
        }
