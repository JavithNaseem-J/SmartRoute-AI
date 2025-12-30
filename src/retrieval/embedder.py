from pathlib import Path
from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.utils.logger import logger


class DocumentEmbedder:
    """Embed documents and build vector store for RAG"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        persist_dir: Path = Path("data/embeddings")
    ):
        """
        Initialize the document embedder.
        
        Args:
            model_name: HuggingFace embedding model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            persist_dir: Directory to persist the vector store
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_dir = Path(persist_dir)
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.vectordb = None
        logger.info("✓ DocumentEmbedder initialized")
    
    def load_documents(self,docs_dir: Path,glob_pattern: str = "**/*") -> List[Document]:
        """
        Load documents from a directory.
        
        Args:
            docs_dir: Directory containing documents
            glob_pattern: Pattern to match files
        
        Returns:
            List of loaded documents
        """
        docs_dir = Path(docs_dir)
        all_docs = []
        
        # Load PDFs
        pdf_loader = DirectoryLoader(str(docs_dir),glob="**/*.pdf",loader_cls=PyPDFLoader,show_progress=True)
        try:
            pdf_docs = pdf_loader.load()
            logger.info(f"Loaded {len(pdf_docs)} PDF pages")
            all_docs.extend(pdf_docs)
        except Exception as e:
            logger.warning(f"Error loading PDFs: {e}")
        
        # Load text files
        txt_loader = DirectoryLoader(str(docs_dir),glob="**/*.txt",loader_cls=TextLoader,show_progress=True)
        try:
            txt_docs = txt_loader.load()
            logger.info(f"Loaded {len(txt_docs)} text files")
            all_docs.extend(txt_docs)
        except Exception as e:
            logger.warning(f"Error loading text files: {e}")
        
        # Load markdown files
        md_loader = DirectoryLoader(str(docs_dir),glob="**/*.md",loader_cls=TextLoader,show_progress=True)
        try:
            md_docs = md_loader.load()
            logger.info(f"Loaded {len(md_docs)} markdown files")
            all_docs.extend(md_docs)
        except Exception as e:
            logger.warning(f"Error loading markdown files: {e}")
        
        logger.info(f"Total documents loaded: {len(all_docs)}")
        return all_docs
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of documents to chunk
        
        Returns:
            List of chunked documents
        """
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def embed_documents(self,documents: List[Document],collection_name: str = "smartroute_docs") -> Chroma:
        """
        Embed documents and create vector store.
        
        Args:
            documents: List of documents to embed
            collection_name: Name for the Chroma collection
        
        Returns:
            Chroma vector store
        """
        # Ensure persist directory exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Embedding {len(documents)} documents...")
        
        # Create vector store
        self.vectordb = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(self.persist_dir),
            collection_name=collection_name
        )
        
        logger.info(f"✓ Vector store created at {self.persist_dir}")
        return self.vectordb
    
    def build_vectorstore(self,docs_dir: Path,collection_name: str = "smartroute_docs") -> Chroma:
        """
        Complete pipeline: load, chunk, and embed documents.
        
        Args:
            docs_dir: Directory containing documents
            collection_name: Name for the Chroma collection
        
        Returns:
            Chroma vector store
        """
        logger.info(f"Building vector store from {docs_dir}")
        
        # Load documents
        documents = self.load_documents(docs_dir)
        
        if not documents:
            logger.warning("No documents found!")
            return None
        
        # Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Embed and create vector store
        vectordb = self.embed_documents(chunks, collection_name)
        
        return vectordb
    
    def add_documents(self,documents: List[Document]) -> None:
        """
        Add new documents to existing vector store.
        
        Args:
            documents: Documents to add
        """
        if self.vectordb is None:
            # Load existing or create new
            if self.persist_dir.exists():
                self.vectordb = Chroma(
                    persist_directory=str(self.persist_dir),
                    embedding_function=self.embeddings
                )
            else:
                logger.warning("No vector store exists. Creating new one.")
                self.embed_documents(documents)
                return
        
        # Chunk and add
        chunks = self.chunk_documents(documents)
        self.vectordb.add_documents(chunks)
        logger.info(f"Added {len(chunks)} chunks to vector store")
    
    def get_stats(self) -> dict:
        """Get vector store statistics."""
        if self.vectordb is None:
            return {"status": "not_initialized", "document_count": 0}
        
        try:
            collection = self.vectordb._collection
            count = collection.count()
            return {
                "status": "ready",
                "document_count": count,
                "persist_dir": str(self.persist_dir),
                "model": self.model_name,
                "chunk_size": self.chunk_size
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
