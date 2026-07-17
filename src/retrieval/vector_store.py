"""
Smart Vector Store — auto-selects Qdrant Cloud or local ChromaDB.

Enterprise upgrade:
- When QDRANT_URL + QDRANT_API_KEY are set, uses Qdrant Cloud (free 1GB cluster).
  Qdrant is persistent in the cloud — documents survive container restarts.
- Falls back to local ChromaDB when env vars are absent (development / offline).

Qdrant free tier: https://cloud.qdrant.io  (1 cluster, 1GB, forever free)

Setup:
  1. Create free cluster at https://cloud.qdrant.io
  2. Copy "Cluster URL" and generate an API key.
  3. Set in .env:
       QDRANT_URL=https://xxxx.us-east4-0.gcp.cloud.qdrant.io
       QDRANT_API_KEY=your-api-key-here
"""
import json
import os
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from src.utils.logger import logger
from src.retrieval.embedder import Embedder

# ── Qdrant availability check ─────────────────────────────────────────────────
_QDRANT_AVAILABLE = False
try:
    from langchain_qdrant import QdrantVectorStore as LCQdrant
    from qdrant_client import QdrantClient
    _QDRANT_AVAILABLE = True
except ImportError:
    pass

# ── ChromaDB (local fallback) ─────────────────────────────────────────────────
from langchain_community.vectorstores import Chroma


class VectorStore:
    """Unified vector store — Qdrant Cloud in production, ChromaDB locally.

    The public API is identical regardless of the backend:
        .create(documents)
        .add_documents(documents)
        .similarity_search(query, k)
        .similarity_search_with_score(query, k)
        .as_retriever(**kwargs)
        .save_bm25_index(documents)   # always local JSON
        .load_bm25_documents()        # always local JSON
        .get_stats()
        .is_ready  (property)
    """

    def __init__(
        self,
        persist_dir: Path = Path("data/embeddings"),
        collection_name: str = "smartroute_docs",
        embedder: Optional[Embedder] = None,
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedder = embedder or Embedder()
        self._vectordb = None
        self._backend = "none"

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_key = os.getenv("QDRANT_API_KEY")

        if qdrant_url and _QDRANT_AVAILABLE:
            self._init_qdrant(qdrant_url, qdrant_key)
        else:
            if qdrant_url and not _QDRANT_AVAILABLE:
                logger.warning(
                    "QDRANT_URL is set but qdrant-client is not installed. "
                    "Run: pip install qdrant-client langchain-qdrant. "
                    "Falling back to ChromaDB."
                )
            self._init_chroma()

    # ── Backend initialisation ────────────────────────────────────────────────

    def _init_qdrant(self, url: str, api_key: Optional[str]) -> None:
        """Connect to Qdrant Cloud and load the collection if it exists."""
        try:
            self._qdrant_client = QdrantClient(url=url, api_key=api_key)
            existing = [c.name for c in self._qdrant_client.get_collections().collections]

            if self.collection_name in existing:
                self._vectordb = LCQdrant(
                    client=self._qdrant_client,
                    collection_name=self.collection_name,
                    embeddings=self.embedder.embeddings,
                )
                count = self._qdrant_client.count(self.collection_name).count
                if count > 0:
                    logger.info(f"####### Qdrant Cloud loaded: {count} documents #######")
                    self._backend = "qdrant"
                else:
                    self._vectordb = None
                    logger.info("Qdrant collection is empty — awaiting first index.")
            else:
                logger.info(f"Qdrant collection '{self.collection_name}' not found — will create on first index.")
                self._qdrant_client_ready = True
                self._backend = "qdrant_empty"

        except Exception as e:
            logger.warning(f"Qdrant Cloud connection failed ({e}). Falling back to ChromaDB.")
            self._init_chroma()

    def _init_chroma(self) -> None:
        """Load local ChromaDB if the persist directory exists."""
        if not self.persist_dir.exists():
            self._backend = "chroma_empty"
            return
        try:
            self._vectordb = Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self.embedder.embeddings,
                collection_name=self.collection_name,
            )
            count = self._vectordb._collection.count()
            if count > 0:
                logger.info(f"####### ChromaDB loaded: {count} documents #######")
                self._backend = "chroma"
            else:
                self._vectordb = None
                self._backend = "chroma_empty"
        except Exception as e:
            logger.error(f"Failed to load ChromaDB: {e}")
            self._vectordb = None
            self._backend = "chroma_empty"

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._vectordb is not None

    def create(self, documents: List[Document]):
        """Index documents. Creates Qdrant collection or ChromaDB on disk."""
        logger.info(f"Indexing {len(documents)} documents → backend={self._backend} ...")

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_key = os.getenv("QDRANT_API_KEY")

        if qdrant_url and _QDRANT_AVAILABLE and "qdrant" in self._backend:
            try:
                self._vectordb = LCQdrant.from_documents(
                    documents=documents,
                    embedding=self.embedder.embeddings,
                    url=qdrant_url,
                    api_key=qdrant_key,
                    collection_name=self.collection_name,
                    force_recreate=False,
                )
                self._backend = "qdrant"
                logger.info("####### Qdrant Cloud index created #######")
                return self._vectordb
            except Exception as e:
                logger.warning(f"Qdrant indexing failed ({e}). Falling back to ChromaDB.")

        # ChromaDB fallback
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._vectordb = Chroma.from_documents(
            documents=documents,
            embedding=self.embedder.embeddings,
            persist_directory=str(self.persist_dir),
            collection_name=self.collection_name,
        )
        self._backend = "chroma"
        logger.info(f"####### ChromaDB index created at {self.persist_dir} #######")
        return self._vectordb

    def add_documents(self, documents: List[Document]) -> None:
        if self._vectordb is None:
            self.create(documents)
            return
        self._vectordb.add_documents(documents)
        logger.info(f"Added {len(documents)} documents to {self._backend}")

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if not self._vectordb:
            return []
        return self._vectordb.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        if not self._vectordb:
            return []
        return self._vectordb.similarity_search_with_score(query, k=k)

    def as_retriever(self, **kwargs):
        return self._vectordb.as_retriever(**kwargs) if self._vectordb else None

    def save_bm25_index(self, documents: List[Document]) -> None:
        """Save BM25 corpus as local JSON (always local — Qdrant doesn't store raw text)."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        bm25_path = self.persist_dir / "bm25_index.json"
        serializable = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in documents
        ]
        with open(bm25_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False)
        logger.info(f"####### BM25 index saved: {bm25_path} #######")

    def load_bm25_documents(self) -> Optional[List[Document]]:
        bm25_path = self.persist_dir / "bm25_index.json"
        if not bm25_path.exists():
            return None
        try:
            with open(bm25_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return [Document(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in raw]
        except Exception:
            return None

    def get_stats(self) -> dict:
        if not self._vectordb:
            return {"status": "not_initialized", "backend": self._backend, "document_count": 0}
        try:
            if self._backend == "qdrant":
                count = self._qdrant_client.count(self.collection_name).count
            else:
                count = self._vectordb._collection.count()
            return {
                "status": "ready",
                "backend": self._backend,
                "document_count": count,
            }
        except Exception as e:
            return {"status": "error", "backend": self._backend, "error": str(e)}
