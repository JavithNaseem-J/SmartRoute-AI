import os
from typing import List

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.embeddings import Embeddings

from src.utils.logger import logger


class Embedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize

        hf_token = os.getenv("HF_TOKEN")

        if not hf_token:
            logger.error("HF_TOKEN environment variable is missing!")
            raise ValueError(
                "HF_TOKEN is required for HuggingFaceInferenceAPIEmbeddings. Please add it to your .env file or Render Environment."
            )

        self._embeddings: Embeddings

        logger.info(
            f"Using HuggingFace Inference API for embeddings (Memory Optimized): {model_name}"
        )
        self._embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=hf_token, model_name=model_name
        )

        logger.info("####### Embedder initialized #######")

    @property
    def embeddings(self) -> Embeddings:
        """Get the underlying embeddings object for LangChain compatibility."""
        return self._embeddings

    def embed_text(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text)  # type: ignore

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings.embed_documents(texts)  # type: ignore

    def get_config(self) -> dict:
        """Get embedder configuration."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "normalize": self.normalize,
        }
