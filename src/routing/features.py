import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.core.dependencies import get_embeddings
from src.routing.keywords import (
    ANALYSIS_KEYWORDS,
    REASONING_KEYWORDS,
    TECHNICAL_TERMS,
)
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
        # Length-decoupled features
        "flesch_kincaid_grade",
        "code_syntax_density",
        "centroid_margin",
        # Semantic features
        "semantic_complexity",
        "simple_similarity",
        "medium_similarity",
        "complex_similarity",
    ]

    def __init__(self):
        self.technical_terms = TECHNICAL_TERMS
        self.reasoning_keywords = REASONING_KEYWORDS
        self.analysis_keywords = ANALYSIS_KEYWORDS

        try:
            self.embedder = get_embeddings()
            self.has_model = True
        except Exception as e:
            logger.warning(f"Could not load HuggingFaceEndpointEmbeddings: {e}")
            self.has_model = False

        # Load pre-computed reference centroids from file (0ms offline loading)
        self.ref_embeddings: dict = {}
        centroids_path = (
            Path(__file__).parent.parent.parent / "data" / "models" / "reference_centroids.npy"
        )
        if centroids_path.exists():
            try:
                loaded = np.load(centroids_path, allow_pickle=True).item()
                self.ref_embeddings = loaded
                self._ref_embeddings_ready = True
            except Exception as e:
                logger.warning(f"Could not load reference_centroids.npy: {e}")
                self._ref_embeddings_ready = False
        else:
            self._ref_embeddings_ready = False

        # FastEmbed fallback model initialized ONCE
        try:
            from fastembed import TextEmbedding

            self.fastembed_model = TextEmbedding("BAAI/bge-small-en-v1.5")
        except Exception as e:
            logger.warning(f"Could not load fastembed local model: {e}")
            # pyrefly: ignore [bad-assignment]
            self.fastembed_model = None

    def _cosine_similarity_max(self, a: np.ndarray, b: np.ndarray) -> "np.ndarray[Any, Any]":
        """Compute max cosine similarity of a (N, D) against b (M, D). Returns (N,)."""
        dot = np.dot(a, b.T)
        norm_a = np.linalg.norm(a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(b, axis=1, keepdims=True).T
        sims = dot / (norm_a * norm_b + 1e-10)
        return sims.max(axis=1)  # type: ignore[no-any-return]

    async def extract(self, query: str) -> Dict:
        """Extract features for a single query."""
        vectors = await self.batch_extract_vectors([query])
        vector = vectors[0]

        features: Dict[str, Any] = {}
        for i, feat in enumerate(self.FEATURE_ORDER):
            val = vector[i]
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

    async def batch_extract_vectors(self, queries: List[str]) -> "np.ndarray[Any, Any]":
        """Extract feature vectors for a list of queries efficiently."""
        n = len(queries)

        lexical = np.array(
            [self._lexical_vector(q) for q in queries], dtype=np.float32
        )  # shape: (n, 15)

        if not self.ref_embeddings:
            semantic: "np.ndarray[Any, Any]" = np.zeros((n, 4), dtype=np.float32)
        else:
            try:
                embeddings_list = await self.embedder.aembed_documents(queries)
                embeddings = np.array(embeddings_list, dtype=np.float32)
            except Exception:
                if self.fastembed_model is not None:
                    embeddings = np.array(
                        list(self.fastembed_model.embed(queries)), dtype=np.float32
                    )
                else:
                    embeddings = np.zeros((n, 384), dtype=np.float32)

            if np.all(embeddings == 0):
                semantic = np.zeros((n, 4), dtype=np.float32)
            else:
                simple_max = self._cosine_similarity_max(embeddings, self.ref_embeddings["simple"])
                medium_max = self._cosine_similarity_max(embeddings, self.ref_embeddings["medium"])
                complex_max = self._cosine_similarity_max(
                    embeddings, self.ref_embeddings["complex"]
                )

                semantic = np.stack(
                    [complex_max - simple_max, simple_max, medium_max, complex_max],
                    axis=1,
                ).astype(np.float32)

        return np.concatenate([lexical, semantic], axis=1)

    def _extract_lexical(self, query: str) -> Dict:
        """Extract non-semantic features from a single query."""
        query_lower = query.lower()
        words = query.split()
        word_count = max(len(words), 1)
        word_set = set(query_lower.split())

        logic_ops = {
            "if",
            "then",
            "and",
            "or",
            "else",
            "not",
            "when",
            "assume",
            "given",
        }
        logic_operator_count = sum(1 for w in word_set if w in logic_ops)

        symbol_count = len(re.findall(r"[^\w\s]", query))
        symbol_density = symbol_count / max(len(query), 1)

        # Flesch-Kincaid Readability Estimation
        sentences = len([s for s in re.split(r"[.!?]+", query) if s.strip()]) or 1
        syllables = sum(len(re.findall(r"[aeiouy]+", w)) for w in query_lower.split()) or 1
        flesch_kincaid = 0.39 * (word_count / sentences) + 11.8 * (syllables / word_count) - 15.59

        # Code Syntax Density
        code_tokens = len(
            re.findall(
                r"[{}=\[\];]|def\s+|class\s+|import\s+|async\s+|return\b|select\b|from\b|where\b",
                query_lower,
            )
        )
        code_syntax_density = code_tokens / word_count

        return {
            "word_count": len(words),
            "sentence_count": sentences,
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
            "flesch_kincaid_grade": float(flesch_kincaid),
            "code_syntax_density": float(code_syntax_density),
            "centroid_margin": 0.0,  # Computed in semantic stack
        }

    def _lexical_vector(self, query: str) -> "np.ndarray[Any, Any]":
        """Return lexical features as a float32 array (15 values)."""
        f = self._extract_lexical(query)
        return np.array(
            [
                float(f[k]) if not isinstance(f[k], bool) else float(f[k])
                for k in self.FEATURE_ORDER[:15]
            ],
            dtype=np.float32,
        )

    def extract_vector(self, features: Dict) -> np.ndarray:
        """Convert a features dict to a numpy vector (used at inference time)."""
        return np.array([float(features.get(f, 0)) for f in self.FEATURE_ORDER], dtype=np.float32)
