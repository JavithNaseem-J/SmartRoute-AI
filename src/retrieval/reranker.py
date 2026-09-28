import os
import re
from typing import List

from langchain_core.documents import Document

from src.utils.logger import logger


class DocumentReranker:
    """Rerank retrieved documents with an explicit, observable strategy."""

    def __init__(self):
        self.mode = os.getenv("RERANKER_MODE", "local").lower()
        if self.mode not in {"local", "disabled"}:
            raise ValueError("RERANKER_MODE must be one of: local, disabled")
        self.last_mode = self.mode

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
        """Use the configured local mode or preserve Qdrant order."""
        if not documents:
            return documents[:top_k]

        if self.mode == "disabled":
            self.last_mode = "disabled"
            return documents[:top_k]

        self.last_mode = "local"
        return self._local_score(query, documents, top_k)
