import re
from typing import Dict, List

import numpy as np

from src.core.dependencies import get_embeddings
from src.utils.logger import logger


class FeatureExtractor:
    """Extract features from queries for routing decisions."""

    FEATURE_ORDER = [
        "word_count",
        "sentence_count",
        "has_code",
        "has_technical_terms",
        "has_numbers",
        "question_depth",
        "is_multipart",
        "requires_reasoning",
        "is_analysis",
        "comma_count",
        "logic_operator_count",
        "symbol_density",
        # Semantic features
        "semantic_complexity",
        "simple_similarity",
        "medium_similarity",
        "complex_similarity",
    ]

    def __init__(self):
        self.technical_terms = {
            "algorithm",
            "database",
            "api",
            "server",
            "client",
            "cache",
            "optimization",
            "deployment",
            "architecture",
            "framework",
            "library",
            "dependency",
            "repository",
            "integration",
        }

        self.reasoning_keywords = {
            "why",
            "how",
            "explain",
            "analyze",
            "compare",
            "evaluate",
            "reasoning",
            "logic",
            "proof",
            "demonstrate",
            "justify",
        }

        self.analysis_keywords = {
            "analyze",
            "analysis",
            "evaluate",
            "assess",
            "review",
            "examine",
            "investigate",
            "synthesize",
            "critique",
        }

        try:
            self.embedder = get_embeddings()
            self.has_model = True
        except Exception as e:
            logger.warning(f"Could not load HuggingFaceEndpointEmbeddings: {e}")
            self.has_model = False

        try:
            from pathlib import Path

            import yaml

            config_path = Path(__file__).parent.parent.parent / "config" / "routing.yaml"
            with open(config_path, "r") as f:
                self.reference_queries = yaml.safe_load(f).get("reference_queries", {})
        except Exception as e:
            logger.warning(f"Could not load reference_queries from config: {e}")
            self.reference_queries = {}

        # Reference embeddings are computed lazily on first async call to avoid
        # blocking the event loop with a synchronous HTTP call at startup.
        self.ref_embeddings: dict = {}
        self._ref_embeddings_ready = False

    def _cosine_similarity_max(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute max cosine similarity of a (N, D) against b (M, D). Returns (N,)."""
        # a: (N, D), b: (M, D) -> dot: (N, M)
        dot = np.dot(a, b.T)
        norm_a = np.linalg.norm(a, axis=1, keepdims=True)  # (N, 1)
        norm_b = np.linalg.norm(b, axis=1, keepdims=True).T  # (1, M)
        sims = dot / (norm_a * norm_b + 1e-10)
        return sims.max(axis=1)  # (N,)

    # ------------------------------------------------------------------
    # Single-query extraction (used at inference time)
    # ------------------------------------------------------------------
    async def extract(self, query: str) -> Dict:
        """Extract features including semantic complexity for a single query."""
        # Reuse batch_extract_vectors to avoid math duplication
        vectors = await self.batch_extract_vectors([query])
        vector = vectors[0]

        # Reconstruct the feature dictionary matching FEATURE_ORDER
        features = {}
        for i, feat in enumerate(self.FEATURE_ORDER):
            val = vector[i]
            # Convert bool/int features back for backward compatibility
            if feat in [
                "has_code",
                "has_technical_terms",
                "has_numbers",
                "is_multipart",
                "requires_reasoning",
                "is_analysis",
            ]:
                features[feat] = bool(val)
            elif feat in [
                "word_count",
                "sentence_count",
                "question_depth",
                "comma_count",
                "logic_operator_count",
            ]:
                features[feat] = int(val)
            else:
                features[feat] = float(val)
        return features

    # ------------------------------------------------------------------
    # Batch extraction (used at training time & extraction)
    # ------------------------------------------------------------------
    async def batch_extract_vectors(self, queries: List[str]) -> np.ndarray:
        """
        Extract feature vectors for a list of queries efficiently.
        Encodes ALL queries in a single batched call to the Inference API.
        """
        n = len(queries)

        # --- Lazy async init of reference embeddings (runs once, non-blocking) ---
        if self.has_model and self.reference_queries and not self._ref_embeddings_ready:
            for k, texts in self.reference_queries.items():
                try:
                    flat_embeddings = await self.embedder.aembed_documents(texts)
                    self.ref_embeddings[k] = np.array(flat_embeddings, dtype=np.float32)
                except Exception as e:
                    logger.warning(f"Failed to fetch reference embeddings for {k}: {e}")
                    self.ref_embeddings[k] = np.zeros((1, 384), dtype=np.float32)
            self._ref_embeddings_ready = True

        # --- Lexical features (fast, no model needed) ---
        lexical = np.array(
            [self._lexical_vector(q) for q in queries], dtype=np.float32
        )  # shape: (n, 12)

        if not self.has_model or not self.ref_embeddings:
            # Pad semantic columns with zeros
            semantic = np.zeros((n, 4), dtype=np.float32)
        else:
            try:
                # Async network call for embeddings
                embeddings_list = await self.embedder.aembed_documents(queries)
                embeddings = np.array(embeddings_list, dtype=np.float32)  # shape: (n, 384)

                simple_max = self._cosine_similarity_max(embeddings, self.ref_embeddings["simple"])
                medium_max = self._cosine_similarity_max(embeddings, self.ref_embeddings["medium"])
                complex_max = self._cosine_similarity_max(
                    embeddings, self.ref_embeddings["complex"]
                )

                semantic = np.stack(
                    [complex_max - simple_max, simple_max, medium_max, complex_max], axis=1
                ).astype(np.float32)  # shape: (n, 4)
            except Exception as e:
                logger.error(f"Failed to fetch embeddings from API: {e}")
                semantic = np.zeros((n, 4), dtype=np.float32)

        return np.concatenate([lexical, semantic], axis=1)  # (n, 16)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_lexical(self, query: str) -> Dict:
        """Extract non-semantic features from a single query."""
        query_lower = query.lower()
        words = query.split()
        word_set = set(query_lower.split())

        logic_ops = {"if", "then", "and", "or", "else", "not", "when", "assume", "given"}
        logic_operator_count = sum(1 for w in word_set if w in logic_ops)

        # Count non-alphanumeric and non-space characters
        symbol_count = len(re.findall(r"[^\w\s]", query))
        symbol_density = symbol_count / max(len(query), 1)

        return {
            "word_count": len(words),
            "sentence_count": len([s for s in re.split(r"[.!?]+", query) if s.strip()]),
            "has_code": bool(re.search(r"```|def\s+\w+|class\s+\w+|import\s+\w+", query)),
            "has_technical_terms": bool(word_set & self.technical_terms),
            "has_numbers": bool(re.search(r"\d+", query)),
            "question_depth": min(query.count("?") + query.count(",") // 2, 5),
            "is_multipart": query.count("?") > 1
            or bool(re.search(r"\b(also|additionally|and)\b", query_lower)),
            "requires_reasoning": bool(word_set & self.reasoning_keywords),
            "is_analysis": bool(word_set & self.analysis_keywords),
            "comma_count": query.count(","),
            "logic_operator_count": logic_operator_count,
            "symbol_density": symbol_density,
        }

    def _lexical_vector(self, query: str) -> np.ndarray:
        """Return lexical features as a float32 array (12 values)."""
        f = self._extract_lexical(query)
        return np.array(
            [
                float(f[k]) if not isinstance(f[k], bool) else float(f[k])
                for k in self.FEATURE_ORDER[:12]
            ],
            dtype=np.float32,
        )

    def extract_vector(self, features: Dict) -> np.ndarray:
        """Convert a features dict to a numpy vector (used at inference time)."""
        return np.array([float(features.get(f, 0)) for f in self.FEATURE_ORDER], dtype=np.float32)
