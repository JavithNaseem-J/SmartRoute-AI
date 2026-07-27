import sys
from pathlib import Path
import asyncio
import numpy as np
import yaml

sys.path.append(str(Path(__file__).parent.parent))
from src.core.dependencies import get_embeddings
from src.utils.logger import logger


async def generate_centroids():
    """Pre-compute cluster centroid embeddings for simple, medium, complex reference queries."""
    config_path = Path(__file__).parent.parent / "config" / "routing.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    reference_queries = config.get("reference_queries", {})
    if not reference_queries:
        logger.error("No reference_queries found in config/routing.yaml")
        return

    logger.info("Initializing HuggingFace embeddings for centroid generation...")
    embedder = get_embeddings()

    centroids = {}
    categories = ["simple", "medium", "complex"]

    for cat in categories:
        texts = reference_queries.get(cat, [])
        if not texts:
            logger.warning(f"No reference queries found for category: {cat}")
            continue

        logger.info(f"Generating embeddings for {len(texts)} {cat} reference queries...")
        try:
            embeddings_list = await embedder.aembed_documents(texts)
            embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
        except Exception as e:
            logger.warning(f"API error ({e}), using fastembed local generator...")
            from fastembed import TextEmbedding

            model = TextEmbedding("BAAI/bge-small-en-v1.5")
            embeddings_matrix = np.array(list(model.embed(texts)), dtype=np.float32)
        centroids[cat] = embeddings_matrix

    output_dir = Path(__file__).parent.parent / "data" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "reference_centroids.npy"

    # pyrefly: ignore [no-matching-overload]
    np.save(output_path, centroids, allow_pickle=True)
    logger.info(f"Centroids saved successfully to {output_path}")


if __name__ == "__main__":
    asyncio.run(generate_centroids())
