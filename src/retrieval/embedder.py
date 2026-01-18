from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.utils.logger import logger


class Embedder:
    
    def __init__(self,model_name: str = "sentence-transformers/all-MiniLM-L6-v2",device: str = "cpu",normalize: bool = True):

        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        
        logger.info(f"Loading embedding model: {model_name}")
        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': normalize}
        )
        logger.info("####### Embedder initialized #######")
    
    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Get the underlying embeddings object for LangChain compatibility."""
        return self._embeddings
    
    def embed_text(self, text: str) -> List[float]:

        return self._embeddings.embed_query(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
 
        return self._embeddings.embed_documents(texts)
    
    def get_config(self) -> dict:
        """Get embedder configuration."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "normalize": self.normalize
        }
