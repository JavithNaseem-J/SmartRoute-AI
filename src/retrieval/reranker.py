import os
from typing import List

import aiohttp
from langchain_core.documents import Document

from src.utils.logger import logger


class DocumentReranker:
    """Re-ranks retrieved documents against the query using an external API."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        self.token = os.getenv("HF_TOKEN")
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        if not self.token:
            logger.warning("HF_TOKEN not set. Re-ranking API may rate-limit or reject requests.")
        else:
            logger.info(f"####### Re-ranker initialized with API: {model_name} #######")

    @property
    def is_ready(self) -> bool:
        # We can attempt requests even without token, but highly limited.
        return True

    async def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Score documents against the query and return the top_k via async HTTP API."""
        if not documents:
            return documents[:top_k]

        texts = [doc.page_content for doc in documents]
        payload = {"inputs": {"source_sentence": query, "sentences": texts}}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url, headers=self.headers, json=payload, timeout=10
                ) as response:
                    response.raise_for_status()
                    scores = await response.json()

            # scores should be a list of floats corresponding to sentences
            if isinstance(scores, list) and len(scores) == len(documents):
                scored_docs = list(zip(documents, scores))
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                top_docs = [doc for doc, score in scored_docs[:top_k]]
                logger.info(f"Re-ranked {len(documents)} documents via API, returning top {top_k}")
                return top_docs
            else:
                logger.warning(f"Unexpected response format from Reranker API: {scores}")
                return documents[:top_k]

        except Exception as e:
            logger.error(f"Re-ranking API call failed: {e}")
            return documents[:top_k]
