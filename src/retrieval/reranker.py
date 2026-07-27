import os
import re
from typing import List

import aiohttp
from langchain_core.documents import Document

from src.utils.logger import logger


class DocumentReranker:
    """Re-ranks retrieved documents against the query using an external API with local fallback."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        self.token = os.getenv("HF_TOKEN")
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        if not self.token:
            logger.warning("HF_TOKEN not set. Re-ranking will use local keyword fallback scorer.")
        else:
            logger.info(f"####### Re-ranker initialized with API: {model_name} #######")

    @property
    def is_ready(self) -> bool:
        return True

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Lowercase token set for keyword overlap scoring."""
        return set(re.findall(r"\w+", text.lower()))

    def _local_score(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        """Deterministic keyword-overlap reranking using Jaccard-inspired token scoring.

        For short queries like 'what is fill?' this ensures documents containing the
        exact word 'fill' as a section header or content keyword rank at the top.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return documents[:top_k]

        scored = []
        for doc in documents:
            doc_tokens = self._tokenize(doc.page_content)
            # Intersection-over-query (recall-biased) so single keywords score well
            overlap = len(query_tokens & doc_tokens)
            score = overlap / len(query_tokens)
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in scored[:top_k]]
        logger.info(f"Local keyword reranker scored {len(documents)} docs, returning top {top_k}")
        return top_docs

    async def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Score documents against the query and return the top_k.

        Attempts external HuggingFace cross-encoder API first; falls back to
        local keyword overlap scoring if the API is unavailable or token absent.
        """
        if not documents:
            return documents[:top_k]

        # Skip external API call if no token configured — use local scorer directly
        if not self.token:
            logger.info("No HF_TOKEN — using local keyword reranker")
            return self._local_score(query, documents, top_k)

        texts = [doc.page_content for doc in documents]
        payload = {"inputs": {"source_sentence": query, "sentences": texts}}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url, headers=self.headers, json=payload, timeout=10
                ) as response:
                    response.raise_for_status()
                    scores = await response.json()

            if isinstance(scores, list) and len(scores) == len(documents):
                scored_docs = list(zip(documents, scores))
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                top_docs = [doc for doc, score in scored_docs[:top_k]]
                logger.info(f"Re-ranked {len(documents)} documents via API, returning top {top_k}")
                return top_docs
            else:
                logger.warning(
                    f"Unexpected response format from Reranker API: {scores}. "
                    "Falling back to local keyword scorer."
                )
                return self._local_score(query, documents, top_k)

        except Exception as e:
            logger.warning(
                f"Re-ranking API call failed: {e}. Falling back to local keyword scorer."
            )
            return self._local_score(query, documents, top_k)
