import os
import re
from typing import List

import aiohttp
from langchain_core.documents import Document

from src.utils.logger import logger


class DocumentReranker:
    """Rerank retrieved documents with an explicit, observable strategy."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.mode = os.getenv("RERANKER_MODE", "local").lower()
        if self.mode not in {"local", "huggingface", "disabled"}:
            raise ValueError("RERANKER_MODE must be one of: local, huggingface, disabled")

        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        self.token = os.getenv("HF_TOKEN")
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self.last_mode = self.mode
        self.last_error: str | None = None

        if self.mode == "huggingface" and not self.token:
            logger.warning(
                "RERANKER_MODE=huggingface but HF_TOKEN is not set; local fallback will be used."
            )
        elif self.mode == "huggingface":
            logger.info(f"Re-ranker initialized with API: {model_name}")

    @property
    def is_ready(self) -> bool:
        return True

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"\w+", text.lower()))

    def _local_score(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return documents[:top_k]

        scored = []
        for doc in documents:
            doc_tokens = self._tokenize(doc.page_content)
            overlap = len(query_tokens & doc_tokens)
            score = overlap / len(query_tokens)
            scored.append((doc, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        top_docs = [doc for doc, _ in scored[:top_k]]
        logger.info(f"Local keyword reranker scored {len(documents)} docs, returning top {top_k}")
        return top_docs

    async def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Use the configured mode and expose any fallback through diagnostics."""
        self.last_error = None
        if not documents:
            return documents[:top_k]

        if self.mode == "disabled":
            self.last_mode = "disabled"
            return documents[:top_k]

        if self.mode == "local":
            self.last_mode = "local"
            return self._local_score(query, documents, top_k)

        if not self.token:
            self.last_mode = "local_fallback"
            self.last_error = "missing_hf_token"
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
                scored_docs.sort(key=lambda item: item[1], reverse=True)
                self.last_mode = "huggingface"
                logger.info(f"Re-ranked {len(documents)} documents via API, returning top {top_k}")
                return [doc for doc, _ in scored_docs[:top_k]]

            logger.warning(
                f"Unexpected response from reranker API: {scores}. Using local fallback."
            )
            self.last_mode = "local_fallback"
            self.last_error = "invalid_huggingface_response"
            return self._local_score(query, documents, top_k)
        except Exception as exc:
            logger.warning(f"Re-ranking API call failed: {exc}. Using local fallback.")
            self.last_mode = "local_fallback"
            self.last_error = type(exc).__name__
            return self._local_score(query, documents, top_k)
