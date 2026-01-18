from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.utils.logger import logger


class DocumentChunker:
    """Handles document chunking with configurable strategies."""
    
    def __init__(self,chunk_size: int = 500,chunk_overlap: int = 100,separators: Optional[List[str]] = None):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=self.separators
        )
        
        logger.info(
            f"DocumentChunker initialized: "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}"
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:

        if not documents:
            logger.warning("No documents to chunk")
            return []
        
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def chunk_text(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """
        Split a single text string into chunks.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of Document chunks
        """
        texts = self.text_splitter.split_text(text)
        documents = [
            Document(page_content=t, metadata=metadata or {})
            for t in texts
        ]
        return documents
    
    def get_config(self) -> dict:
        """Get chunker configuration."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separators": self.separators
        }
