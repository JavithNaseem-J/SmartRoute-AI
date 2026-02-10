from pathlib import Path
from typing import List, Tuple, Optional
from langchain_community.retrievers import BM25Retriever
from src.utils.logger import logger
from src.retrieval.vector_store import VectorStore
from src.retrieval.embedder import Embedder


class DocumentRetriever:
    """Handles document retrieval with dense, BM25, and hybrid search."""
    
    def __init__(
        self,
        persist_dir: Path = Path("data/embeddings"),
        collection_name: str = "smartroute_docs",
        top_k: int = 5,
        max_distance: float = 1.5,
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
    ):
        
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.top_k = top_k
        self.max_distance = max_distance
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        
        # Initialize components
        self.embedder = Embedder()
        self.vector_store = VectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedder=self.embedder
        )
        
        # BM25 retriever
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.dense_retriever = None
        
        # Load indexes
        self._load_bm25_index()
        self._setup_hybrid_retriever()
        
        logger.info("####### DocumentRetriever initialized #######")
    
    def _load_bm25_index(self) -> None:
        """Load BM25 index from saved documents."""
        docs = self.vector_store.load_bm25_documents()
        if docs:
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            self.bm25_retriever.k = self.top_k
            logger.info(f"####### BM25 index loaded: {len(docs)} documents #######")
        else:
            logger.warning("No BM25 index found - hybrid search unavailable")
    
    def _setup_hybrid_retriever(self) -> None:
        """Setup hybrid retriever combining BM25 and dense search."""
        if self.vector_store.is_ready:
            self.dense_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.top_k}
            )
            
            if self.bm25_retriever:
                logger.info(
                    f"####### Hybrid retriever ready "
                    f"(BM25: {self.bm25_weight}, Dense: {self.dense_weight}) #######"
                )
            else:
                logger.info("Using dense-only retrieval (BM25 not available)")
        else:
            logger.warning("No retrieval indexes available")
    
    def reload(self) -> None:
        """Reload all indexes (call after new documents are added)."""
        logger.info("Reloading retrieval indexes...")
        self.vector_store = VectorStore(
            persist_dir=self.persist_dir,
            collection_name=self.collection_name,
            embedder=self.embedder
        )
        self._load_bm25_index()
        self._setup_hybrid_retriever()
    
    def retrieve(self, query: str) -> Tuple[str, List[str]]:

        if not self.vector_store.is_ready:
            logger.warning("No vector store available")
            return "", []
        
        try:
            if self.bm25_retriever and self.dense_retriever:
                return self._retrieve_hybrid(query)
            else:
                return self._retrieve_dense(query)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return "", []
    
    def _retrieve_hybrid(self, query: str) -> Tuple[str, List[str]]:
        """Retrieve using hybrid BM25 + dense search."""
        logger.info("Using hybrid search (BM25 + Dense)")
        
        # Get results from both retrievers
        bm25_results = self.bm25_retriever.invoke(query) if self.bm25_retriever else []
        dense_results = self.dense_retriever.invoke(query) if self.dense_retriever else []
        
        # Combine and deduplicate results
        seen_content = set()
        combined_results = []
        
        # Add BM25 results (with weight)
        for doc in bm25_results[:self.top_k]:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                combined_results.append(doc)
        
        # Add dense results (with weight)
        for doc in dense_results[:self.top_k]:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                combined_results.append(doc)
        
        # Limit to top_k
        results = combined_results[:self.top_k]
        
        context_parts = []
        sources = []
        
        for i, doc in enumerate(results):
            context_parts.append(
                f"[Source {i+1}]\n{doc.page_content}"
            )
            source = doc.metadata.get('source', 'Unknown')
            sources.append(f"Source {i+1}: {source}")
        
        logger.info(f"Hybrid search retrieved {len(results)} documents")
        
        context = "\n\n".join(context_parts)
        return context, sources
    
    def _retrieve_dense(self, query: str) -> Tuple[str, List[str]]:
        """Retrieve using dense semantic search with score filtering."""
        logger.info("Using dense-only search")
        
        results = self.vector_store.similarity_search_with_score(
            query, k=self.top_k
        )
        
        # Filter by max distance
        filtered = [
            (doc, score) for doc, score in results
            if score <= self.max_distance
        ]
        
        if not filtered:
            logger.info(f"No documents below distance threshold {self.max_distance}")
            return "", []
        
        context_parts = []
        sources = []
        
        for i, (doc, score) in enumerate(filtered):
            # Convert L2 distance to relevance score (0-1)
            relevance = max(0, 1 - (score / 2))
            
            context_parts.append(
                f"[Source {i+1}, Relevance: {relevance:.2f}]\n{doc.page_content}"
            )
            source = doc.metadata.get('source', 'Unknown')
            sources.append(f"Source {i+1}: {source}")
        
        logger.info(f"Dense search retrieved {len(filtered)} documents")
        
        context = "\n\n".join(context_parts)
        return context, sources
    
    def retrieve_documents(self, query: str) -> List:

        if not self.vector_store.is_ready:
            return []
        
        if self.bm25_retriever and self.dense_retriever:
            # Manual hybrid retrieval
            bm25_results = self.bm25_retriever.invoke(query)[:self.top_k]
            dense_results = self.dense_retriever.invoke(query)[:self.top_k]
            
            # Combine and deduplicate
            seen_content = set()
            combined = []
            for doc in bm25_results + dense_results:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    combined.append(doc)
            return combined[:self.top_k]
        else:
            return self.vector_store.similarity_search(query, k=self.top_k)
    
    @property
    def retrieval_mode(self) -> str:
        """Get current retrieval mode."""
        if self.bm25_retriever and self.dense_retriever:
            return "hybrid"
        elif self.vector_store.is_ready:
            return "dense_only"
        else:
            return "unavailable"
    
    def get_stats(self) -> dict:
        """Get retriever statistics."""
        return {
            "status": "loaded" if self.vector_store.is_ready else "not_loaded",
            "retrieval_mode": self.retrieval_mode,
            "top_k": self.top_k,
            "max_distance": self.max_distance,
            "bm25_weight": self.bm25_weight if (self.bm25_retriever and self.dense_retriever) else None,
            "dense_weight": self.dense_weight if (self.bm25_retriever and self.dense_retriever) else None,
            "vector_store": self.vector_store.get_stats()
        }
