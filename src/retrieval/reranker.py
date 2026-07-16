from typing import List
from langchain_core.documents import Document
from src.utils.logger import logger

try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_AVAILABLE = True
except ImportError:
    _CROSS_ENCODER_AVAILABLE = False


class DocumentReranker:
    """Re-ranks retrieved documents against the query using a Cross-Encoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

        if _CROSS_ENCODER_AVAILABLE:
            try:
                logger.info(f"Loading re-ranker model: {model_name}")
                self._model = CrossEncoder(model_name, device=device)
                logger.info("####### Re-ranker initialized #######")
            except Exception as e:
                logger.warning(f"Failed to load re-ranker model: {e}")
        else:
            logger.warning("sentence-transformers not installed. Re-ranking disabled.")

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Score documents against the query and return the top_k."""
        if not self.is_ready or not documents:
            return documents[:top_k]

        # Prepare pairs for scoring: [(query, doc1), (query, doc2), ...]
        pairs = [[query, doc.page_content] for doc in documents]
        
        try:
            scores = self._model.predict(pairs)
            
            # Combine docs with scores and sort descending
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Return just the top documents
            top_docs = [doc for doc, score in scored_docs[:top_k]]
            logger.info(f"Re-ranked {len(documents)} documents, returning top {top_k}")
            return top_docs
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            return documents[:top_k]

