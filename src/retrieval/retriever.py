from pathlib import Path
from typing import List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.logger import logger


class DocumentRetriever:
    """RAG document retrieval with semantic search"""
    
    def __init__(
        self,
        persist_dir: Path = Path("data/embeddings"),
        top_k: int = 5,
        max_distance: float = 1.5  # ChromaDB uses L2 distance, lower = better
    ):
        self.persist_dir = Path(persist_dir)  # Ensure it's a Path object
        self.top_k = top_k
        self.max_distance = max_distance  # Maximum L2 distance to accept
        self.vectordb = None
        
        # Load embedding model
        logger.info("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load vector store if exists
        self._load_vectorstore()
    
    def _load_vectorstore(self):
        """Load or reload the vector store"""
        if self.persist_dir.exists():
            logger.info(f"Loading vector store from {self.persist_dir}")
            try:
                self.vectordb = Chroma(
                    persist_directory=str(self.persist_dir),
                    embedding_function=self.embeddings,
                    collection_name="smartroute_docs"  # Must match embedder's collection name
                )
                # Verify the vectordb has documents
                doc_count = self.vectordb._collection.count()
                if doc_count > 0:
                    logger.info(f"✓ Vector store loaded with {doc_count} documents")
                else:
                    logger.warning("Vector store exists but has no documents")
                    self.vectordb = None
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
                self.vectordb = None
        else:
            logger.warning(f"No vector store found at {self.persist_dir}")
            self.vectordb = None
    
    def reload(self):
        """Reload the vector store (call after new documents are added)"""
        logger.info("Reloading vector store...")
        self._load_vectorstore()
    
    def retrieve(self, query: str) -> Tuple[str, List[str]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query
        
        Returns:
            (context, sources) tuple
            context: Formatted context string
            sources: List of source references
        """
        
        if not self.vectordb:
            logger.warning("No vector store available")
            return "", []
        
        try:
            # Semantic search with scores
            results = self.vectordb.similarity_search_with_score(
                query,
                k=self.top_k
            )
            
            # Filter by max distance (lower = more similar in L2 distance)
            filtered = [
                (doc, score) for doc, score in results
                if score <= self.max_distance
            ]
            
            if not filtered:
                logger.info(f"No documents below distance threshold {self.max_distance}")
                return "", []
            
            logger.info(f"Retrieved {len(filtered)} relevant documents")
            
            # Format context
            context_parts = []
            sources = []
            
            for i, (doc, score) in enumerate(filtered):
                # Convert L2 distance to relevance score (0-1 scale)
                relevance = max(0, 1 - (score / 2))  # distance 0 = relevance 1, distance 2 = relevance 0
                
                # Add to context
                context_parts.append(
                    f"[Source {i+1}, Relevance: {relevance:.2f}]\n"
                    f"{doc.page_content}"
                )
                
                # Add source reference
                source = doc.metadata.get('source', 'Unknown')
                sources.append(f"Source {i+1}: {source}")
            
            context = "\n\n".join(context_parts)
            
            return context, sources
        
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return "", []
    
    @staticmethod
    def create_vectordb(
        docs_dir: Path,
        persist_dir: Path,
        chunk_size: int = 800,
        chunk_overlap: int = 150
    ):
        """
        Create vector database from documents (supports PDF and TXT)
        
        Args:
            docs_dir: Directory containing documents
            persist_dir: Where to save vector store
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        
        logger.info(f"Creating vector store from {docs_dir}")
        
        all_docs = []
        
        # Load PDF files
        pdf_files = list(Path(docs_dir).glob("**/*.pdf"))
        if pdf_files:
            logger.info(f"Found {len(pdf_files)} PDF files")
            for pdf_path in pdf_files:
                try:
                    loader = PyPDFLoader(str(pdf_path))
                    pdf_docs = loader.load()
                    all_docs.extend(pdf_docs)
                    logger.info(f"  ✓ Loaded {pdf_path.name}: {len(pdf_docs)} pages")
                except Exception as e:
                    logger.error(f"  ✗ Failed to load {pdf_path.name}: {e}")
        
        # Load TXT files
        txt_files = list(Path(docs_dir).glob("**/*.txt"))
        if txt_files:
            logger.info(f"Found {len(txt_files)} TXT files")
            for txt_path in txt_files:
                try:
                    loader = TextLoader(str(txt_path))
                    txt_docs = loader.load()
                    all_docs.extend(txt_docs)
                    logger.info(f"  ✓ Loaded {txt_path.name}")
                except Exception as e:
                    logger.error(f"  ✗ Failed to load {txt_path.name}: {e}")
        
        logger.info(f"Total documents loaded: {len(all_docs)}")
        
        if len(all_docs) == 0:
            raise ValueError(f"No documents found in {docs_dir}")
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(all_docs)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Create embeddings
        logger.info("Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create and persist vector store
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(persist_dir)
        )
        
        logger.info(f"✓ Vector store created at {persist_dir}")
        
        return vectordb
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store"""
        if not self.vectordb:
            return {'status': 'not_loaded'}
        
        try:
            collection = self.vectordb._collection
            count = collection.count()
            
            return {
                'status': 'loaded',
                'document_count': count,
                'top_k': self.top_k,
                'score_threshold': self.score_threshold
            }
        except:
            return {'status': 'error'}