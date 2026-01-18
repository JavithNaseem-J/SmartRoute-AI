from pathlib import Path
from typing import List, Optional
import pickle
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.utils.logger import logger
from src.retrieval.embedder import Embedder


class VectorStore:
    """ChromaDB vector store wrapper."""
    
    def __init__(
        self,
        persist_dir: Path = Path("data/embeddings"),
        collection_name: str = "smartroute_docs",
        embedder: Optional[Embedder] = None,
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedder = embedder or Embedder()
        self._vectordb: Optional[Chroma] = None
        self._load_if_exists()
    
    def _load_if_exists(self) -> None:
        """Load existing vector store if it exists."""
        if not self.persist_dir.exists():
            return
        try:
            self._vectordb = Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self.embedder.embeddings,
                collection_name=self.collection_name
            )
            count = self._vectordb._collection.count()
            if count > 0:
                logger.info(f"####### Vector store loaded: {count} documents #######")
            else:
                self._vectordb = None
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            self._vectordb = None
    
    @property
    def is_ready(self) -> bool:
        return self._vectordb is not None
    
    def create(self, documents: List[Document]) -> Chroma:
        """Create a new vector store from documents."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creating vector store with {len(documents)} documents...")
        self._vectordb = Chroma.from_documents(
            documents=documents,
            embedding=self.embedder.embeddings,
            persist_directory=str(self.persist_dir),
            collection_name=self.collection_name
        )
        logger.info(f"####### Vector store created at {self.persist_dir} #######")
        return self._vectordb
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to existing vector store."""
        if self._vectordb is None:
            self.create(documents)
            return
        self._vectordb.add_documents(documents)
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if not self._vectordb:
            return []
        return self._vectordb.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        if not self._vectordb:
            return []
        return self._vectordb.similarity_search_with_score(query, k=k)
    
    def as_retriever(self, **kwargs):
        """Get a retriever interface."""
        return self._vectordb.as_retriever(**kwargs) if self._vectordb else None
    
    def save_bm25_index(self, documents: List[Document]) -> None:
        """Save documents for BM25 index (used in hybrid search)."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        bm25_path = self.persist_dir / "bm25_index.pkl"
        
        with open(bm25_path, 'wb') as f:
            pickle.dump(documents, f)
        logger.info(f"####### BM25 index saved at {bm25_path} #######")
    
    def load_bm25_documents(self) -> Optional[List[Document]]:
        """Load documents for BM25 index."""
        bm25_path = self.persist_dir / "bm25_index.pkl"
        if not bm25_path.exists():
            return None
        try:
            with open(bm25_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    
    def get_stats(self) -> dict:
        """Get vector store statistics."""
        if not self._vectordb:
            return {"status": "not_initialized", "document_count": 0}
        try:
            return {
                "status": "ready",
                "document_count": self._vectordb._collection.count(),
                "persist_dir": str(self.persist_dir)
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
