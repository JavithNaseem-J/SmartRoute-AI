"""
VectorStore — Qdrant Cloud only.

QDRANT_URL + QDRANT_API_KEY are REQUIRED. The app will refuse to start without them.
No local ChromaDB fallback — this system runs in the cloud.

Get your free Qdrant Cloud cluster at: https://cloud.qdrant.io
"""

import json
import os
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore as LCQdrant
from qdrant_client import QdrantClient

from src.utils.logger import logger
from src.retrieval.embedder import Embedder

# BM25 documents are stored locally in the container's ephemeral disk.
# Qdrant stores vectors permanently in the cloud; BM25 is rebuilt on restart.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BM25_PATH = _PROJECT_ROOT / "data" / "bm25_index.json"


class VectorStore:
    """Qdrant Cloud vector store.

    QDRANT_URL and QDRANT_API_KEY must be set.
    Raises RuntimeError on startup if missing.
    """

    def __init__(
        self,
        persist_dir: Path = _PROJECT_ROOT
        / "data"
        / "embeddings",  # unused, kept for API compat
        collection_name: str = "smartroute_docs",
        embedder: Optional[Embedder] = None,
    ):
        self.collection_name = collection_name
        self.embedder = embedder or Embedder()
        self._vectordb = None

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url:
            raise RuntimeError(
                "QDRANT_URL is not set.\n"
                "This app requires Qdrant Cloud for vector storage.\n"
                "1. Create a free cluster at https://cloud.qdrant.io\n"
                "2. Copy the Cluster URL and generate an API key.\n"
                "3. Set QDRANT_URL=https://... and QDRANT_API_KEY=... in your env vars."
            )

        self._client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        logger.info("VectorStore → Qdrant Cloud connected")

        # Load collection if it already exists with data
        existing = [c.name for c in self._client.get_collections().collections]
        if self.collection_name in existing:
            count = self._client.count(self.collection_name).count
            if count > 0:
                self._vectordb = LCQdrant(
                    client=self._client,
                    collection_name=self.collection_name,
                    embeddings=self.embedder.embeddings,
                )
                logger.info(
                    f"####### Qdrant collection loaded: {count} documents #######"
                )
            else:
                logger.info(
                    "Qdrant collection exists but is empty — awaiting first index."
                )
        else:
            logger.info(
                f"Qdrant collection '{self.collection_name}' not found — will create on first index."
            )

    @property
    def is_ready(self) -> bool:
        return self._vectordb is not None

    def create(self, documents: List[Document]):
        """Index documents into Qdrant Cloud."""
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_key = os.getenv("QDRANT_API_KEY")
        logger.info(f"Indexing {len(documents)} documents into Qdrant...")
        self._vectordb = LCQdrant.from_documents(
            documents=documents,
            embedding=self.embedder.embeddings,
            url=qdrant_url,
            api_key=qdrant_key,
            collection_name=self.collection_name,
            force_recreate=False,
        )
        logger.info("####### Qdrant index created #######")
        return self._vectordb

    def add_documents(self, documents: List[Document]) -> None:
        if self._vectordb is None:
            self.create(documents)
            return
        self._vectordb.add_documents(documents)
        logger.info(f"Added {len(documents)} documents to Qdrant")

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if not self._vectordb:
            return []
        return self._vectordb.similarity_search(query, k=k)  # type: ignore

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        if not self._vectordb:
            return []
        return self._vectordb.similarity_search_with_score(query, k=k)  # type: ignore

    def as_retriever(self, **kwargs):
        return self._vectordb.as_retriever(**kwargs) if self._vectordb else None

    def save_bm25_index(self, documents: List[Document]) -> None:
        """Save BM25 corpus as local JSON in the container's temp storage.

        Qdrant stores vectors permanently; BM25 is rebuilt from this file on restart.
        In Render deployments, documents must be re-indexed after each restart.
        """
        _BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
        serializable = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in documents
        ]
        with open(_BM25_PATH, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False)
        logger.info(f"BM25 corpus saved: {len(documents)} documents")

    def load_bm25_documents(self) -> Optional[List[Document]]:
        if not _BM25_PATH.exists():
            return None
        try:
            with open(_BM25_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return [
                Document(page_content=d["page_content"], metadata=d.get("metadata", {}))
                for d in raw
            ]
        except Exception:
            return None

    def get_stats(self) -> dict:
        if not self._vectordb:
            return {
                "status": "not_initialized",
                "backend": "qdrant",
                "document_count": 0,
            }
        try:
            count = self._client.count(self.collection_name).count
            return {"status": "ready", "backend": "qdrant", "document_count": count}
        except Exception as e:
            return {"status": "error", "backend": "qdrant", "error": str(e)}
