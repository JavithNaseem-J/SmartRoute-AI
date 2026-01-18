from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader
)
from langchain_core.documents import Document
from src.utils.logger import logger
from src.retrieval.chunking import DocumentChunker
from src.retrieval.vector_store import VectorStore
from src.retrieval.embedder import Embedder


class DocumentIndexer:
    """Orchestrates document loading, chunking, and indexing."""
    
    def __init__(
        self,
        persist_dir: Path = Path("data/embeddings"),
        collection_name: str = "smartroute_docs",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedder: Optional[Embedder] = None,
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        
        self.embedder = embedder or Embedder()
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.vector_store = VectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedder=self.embedder
        )
        
        logger.info(f"DocumentIndexer initialized: {collection_name}")
    
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
                logger.warning(f"Unknown file type: {file_type}")
                continue
            
            loader_cls, glob_pattern = loader_map[file_type]
            loader = DirectoryLoader(
                str(docs_dir),
                glob=glob_pattern,
                loader_cls=loader_cls,
                show_progress=True
            )
            
            try:
                docs = loader.load()
                logger.info(f"Loaded {len(docs)} {file_type.upper()} files")
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Error loading {file_type} files: {e}")
        
        return all_docs
    
    def index_documents(self, documents: List[Document], save_bm25: bool = True) -> None:
        """Index documents into vector store."""
        if not documents:
            logger.warning("No documents to index")
            return
        
        chunks = self.chunker.chunk_documents(documents)
        logger.info(f"Chunked into {len(chunks)} chunks")
        
        self.vector_store.create(chunks)
        
        if save_bm25:
            self.vector_store.save_bm25_index(chunks)
    
    def index_directory(
        self,
        docs_dir: Path,
        file_types: Optional[List[str]] = None,
        save_bm25: bool = True,
    ) -> None:
        """Load and index all documents from a directory."""
        logger.info(f"Indexing documents from {docs_dir}")
        
        documents = self.load_documents(docs_dir, file_types)
        
        if not documents:
            logger.warning("No documents found!")
            return
        
        self.index_documents(documents, save_bm25)
        logger.info(f"Successfully indexed documents")
    
    def add_documents(self, documents: List[Document], update_bm25: bool = True) -> None:
        """Add documents to existing index."""
        chunks = self.chunker.chunk_documents(documents)
        self.vector_store.add_documents(chunks)
        
        if update_bm25:
            existing_docs = self.vector_store.load_bm25_documents() or []
            all_docs = existing_docs + chunks
            self.vector_store.save_bm25_index(all_docs)
        
        logger.info(f"Added {len(chunks)} document chunks to index")
    
    def get_stats(self) -> dict:
        """Get indexer statistics."""
        return {
            "vector_store": self.vector_store.get_stats(),
            "chunker": self.chunker.get_config(),
            "embedder": self.embedder.get_config(),
        }
