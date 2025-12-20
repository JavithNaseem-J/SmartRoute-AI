from typing import List, Dict
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.utils.logging import logger


class DocumentRetriever:
    """
    HYBRID: LangChain for retrieval, custom for filtering/formatting
    """
    
    def __init__(
        self,
        persist_dir: str = "data/embeddings",
        top_k: int = 5,
        score_threshold: float = 0.7
    ):
        self.persist_dir = persist_dir
        self.top_k = top_k
        self.score_threshold = score_threshold
        
        # LangChain: Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        logger.info(f"✓ Retriever initialized (top_k={top_k}, threshold={score_threshold})")
    
    def get_vectorstore(self, collection_name: str = "documents"):
        """LangChain: Load vector store"""
        return Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
    
    def retrieve(
        self,
        query: str,
        collection_name: str = "documents",
        top_k: int = None
    ) -> Dict:
        """
        LangChain for retrieval, custom for filtering
        """
        
        k = top_k or self.top_k
        
        try:
            vectorstore = self.get_vectorstore(collection_name)
            
            # LangChain: Similarity search with scores
            results = vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=k
            )
            
            # Custom: Filter by threshold
            filtered_results = [
                (doc, score) for doc, score in results
                if score >= self.score_threshold
            ]
            
            # Custom: Format results
            documents = [doc.page_content for doc, _ in filtered_results]
            metadatas = [doc.metadata for doc, _ in filtered_results]
            scores = [score for _, score in filtered_results]
            
            logger.info(
                f"✓ Retrieved {len(filtered_results)}/{len(results)} docs "
                f"above threshold"
            )
            
            return {
                'documents': documents,
                'metadatas': metadatas,
                'scores': scores,
                'count': len(documents)
            }
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {
                'documents': [],
                'metadatas': [],
                'scores': [],
                'count': 0
            }
    
    def format_context(self, retrieval_results: Dict) -> str:
        """Custom: Format context for your models"""
        
        if not retrieval_results['documents']:
            return ""
        
        context_parts = []
        for idx, (doc, meta) in enumerate(
            zip(retrieval_results['documents'], retrieval_results['metadatas']),
            1
        ):
            source = meta.get('source', 'Unknown')
            context_parts.append(f"[Source {idx}: {source}]\n{doc}\n")
        
        return "\n".join(context_parts)
    
    def get_sources(self, retrieval_results: Dict) -> List[str]:
        """Custom: Extract unique sources"""
        sources = set()
        for meta in retrieval_results['metadatas']:
            if 'source' in meta:
                sources.add(meta['source'])
        return list(sources)
    
    def retrieve_with_context(
        self,
        query: str,
        collection_name: str = "documents",
        top_k: int = None
    ) -> Dict:
        """
        Retrieve and format in one step
        LangChain retrieval + custom formatting
        """
        
        results = self.retrieve(query, collection_name, top_k)
        
        return {
            'context': self.format_context(results),
            'sources': self.get_sources(results),
            'document_count': results['count'],
            'raw_results': results
        }
    
    def get_retriever_as_langchain(self, collection_name: str = "documents"):
        """
        Export as LangChain retriever for compatibility
        Useful if you want to use with LangChain chains
        """
        vectorstore = self.get_vectorstore(collection_name)
        
        return vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": self.top_k,
                "score_threshold": self.score_threshold
            }
        )