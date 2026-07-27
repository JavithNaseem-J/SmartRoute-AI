import asyncio
import uuid
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import (
    DirectoryLoader,
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

    def load_documents(
        self,
        docs_dir: Path,
        file_types: Optional[List[str]] = None,
    ) -> List[Document]:
        """Load documents from directory using LangChain loaders."""
        docs_dir = Path(docs_dir)
        file_types = file_types or ["pdf", "txt", "md"]
        all_docs = []

        loader_map = {
            "pdf": (PyPDFLoader, "**/*.pdf"),
            "txt": (TextLoader, "**/*.txt"),
            "md": (TextLoader, "**/*.md"),
        }

        for file_type in file_types:
            if file_type not in loader_map:
                continue

            loader_cls, glob_pattern = loader_map[file_type]
            loader = DirectoryLoader(
                str(docs_dir),
                glob=glob_pattern,
                loader_cls=loader_cls,  # type: ignore[arg-type]
                show_progress=True,
            )

            try:
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Error loading {file_type} files: {e}")

        return all_docs

    async def aindex_documents(self, documents: List[Document]) -> None:
        """Async index documents into vector store (safe on running event loop)."""
        if not documents:
            return

        chunks = self.chunker.chunk_documents(documents)
        logger.info(f"Chunked into {len(chunks)} chunks")

        await self._async_add_documents(chunks)

    def index_documents(self, documents: List[Document]) -> None:
        """Synchronous wrapper for index_documents."""
        if not documents:
            return
        asyncio.run(self.aindex_documents(documents))

    async def aindex_directory(
        self,
        docs_dir: Path,
        file_types: Optional[List[str]] = None,
    ) -> None:
        """Async load and index all documents from a directory."""
        documents = await asyncio.to_thread(self.load_documents, docs_dir, file_types)
        if not documents:
            return
        await self.aindex_documents(documents)

    def index_directory(
        self,
        docs_dir: Path,
        file_types: Optional[List[str]] = None,
    ) -> None:
        """Synchronous wrapper to load and index all documents from a directory."""
        asyncio.run(self.aindex_directory(docs_dir, file_types))

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to existing index."""
        chunks = self.chunker.chunk_documents(documents)
        asyncio.run(self._async_add_documents(chunks))

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

    async def adelete_document(
        self, filename: str, docs_dir: Path = Path("data/documents")
    ) -> bool:
        """Delete a document file from disk and purge its vectors from Qdrant and Redis cache."""
        target_file = docs_dir / filename
        deleted = False

        # 1. Remove file from disk
        if target_file.exists():
            target_file.unlink()
            deleted = True
            logger.info(f"Deleted document file: {target_file}")

        # 2. Delete vectors from Qdrant matching metadata source
        try:
            if await self.qdrant.collection_exists(self.collection_name):
                # Match metadata source equal or ending with filename
                await self.qdrant.delete(
                    collection_name=self.collection_name,
                    points_selector=models.Filter(
                        should=[
                            models.FieldCondition(
                                key="metadata.source",
                                match=models.MatchValue(value=str(target_file)),
                            ),
                            models.FieldCondition(
                                key="metadata.source",
                                match=models.MatchText(text=filename),
                            ),
                        ]
                    ),
                )
                logger.info(f"Purged vector points for document: {filename}")
        except Exception as e:
            logger.error(f"Error purging vectors for {filename}: {e}")

        # 3. Flush Redis semantic cache so stale answers aren't served
        try:
            redis = get_redis_client()
            await redis.flushdb()
            logger.info("Flushed Redis semantic cache after document deletion.")
        except Exception as e:
            logger.warning(f"Failed to flush Redis on document deletion: {e}")

        return deleted

    async def aclear_all_documents(self, docs_dir: Path = Path("data/documents")) -> None:
        """Clear all document files from disk, reset Qdrant collection, and flush Redis cache."""
        # 1. Remove all files from docs_dir
        if docs_dir.exists():
            for file_path in docs_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
            logger.info(f"Cleared all document files from {docs_dir}")

        # 2. Delete Qdrant collection
        try:
            if await self.qdrant.collection_exists(self.collection_name):
                await self.qdrant.delete_collection(self.collection_name)
                logger.info(f"Deleted Qdrant collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting Qdrant collection: {e}")

        # 3. Flush Redis cache
        try:
            redis = get_redis_client()
            await redis.flushdb()
            logger.info("Flushed Redis semantic cache on clear all.")
        except Exception as e:
            logger.warning(f"Failed to flush Redis on clear all: {e}")

    def get_stats(self) -> dict:
        """Get indexer statistics."""
        return {
            "chunker": self.chunker.get_config(),
        }
