from unittest.mock import patch

from langchain_core.documents import Document

from src.retrieval.retriever import DocumentRetriever
from src.utils.logger import logger


def main():
    logger.info("Initializing DocumentRetriever with mock Qdrant...")

    with patch("src.retrieval.retriever.VectorStore") as MockVectorStore:
        mock_vs = MockVectorStore.return_value
        mock_vs.is_ready = False  # Simulate Qdrant being down or empty

        retriever = DocumentRetriever()

        # Now let's save some dummy documents to BM25
        docs = [
            Document(page_content="This is a test document about AI.", metadata={"source": "test"})
        ]
        retriever.save_bm25_index(docs)

        # Check if BM25 is loaded
        if retriever.bm25_retriever:
            print("SUCCESS: BM25 retriever is loaded.")

            # Try a retrieval to ensure fallback runs (will use BM25 only since dense_retriever is None)
            context, sources = retriever.retrieve("AI")
            print(f"Retrieved {len(sources)} sources.")
            print(f"Context snippet: {context}")

        else:
            print("FAILED: BM25 retriever is NOT loaded.")


if __name__ == "__main__":
    main()
