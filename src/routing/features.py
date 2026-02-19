import re
from typing import Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class FeatureExtractor:
    """Extract features from queries for routing decisions."""
    
    FEATURE_ORDER = [
        'word_count', 'sentence_count', 'has_code', 'has_technical_terms',
        'has_numbers', 'question_depth', 'is_multipart', 'requires_reasoning',
        'is_analysis', 'comma_count',
        # Semantic features
        'semantic_complexity', 'simple_similarity', 'complex_similarity'
    ]

    def __init__(self):
        self.technical_terms = {
            'algorithm', 'database', 'api', 'server', 'client', 'cache',
            'optimization', 'deployment', 'architecture', 'framework',
            'library', 'dependency', 'repository', 'integration'
        }
        
        self.reasoning_keywords = {
            'why', 'how', 'explain', 'analyze', 'compare', 'evaluate',
            'reasoning', 'logic', 'proof', 'demonstrate', 'justify'
        }
        
        self.analysis_keywords = {
            'analyze', 'analysis', 'evaluate', 'assess', 'review',
            'examine', 'investigate', 'synthesize', 'critique'
        }
        
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.has_model = True
        except Exception as e:
            print(f"Warning: Could not load SentenceTransformer: {e}")
            self.has_model = False

        self.reference_queries = {
            'simple': [
                "What is X?", "Define Y", "Who is Z?",
                "Simple definition of A", "List the B"
            ],
            'complex': [
                "Analyze the impact of...", "Compare and contrast...",
                "Evaluate the effectiveness of...", "Synthesize findings from...",
                "Critique the methodology of..."
            ]
        }

        # Pre-compute reference embeddings once at init
        if self.has_model:
            self.ref_embeddings = {
                k: self.embedder.encode(v, show_progress_bar=False)
                for k, v in self.reference_queries.items()
            }
        else:
            self.ref_embeddings = {}

    # ------------------------------------------------------------------
    # Single-query extraction (used at inference time)
    # ------------------------------------------------------------------
    def extract(self, query: str) -> Dict:
        """Extract features including semantic complexity for a single query."""
        features = self._extract_lexical(query)

        if self.has_model and self.ref_embeddings:
            emb = self.embedder.encode(query, show_progress_bar=False)
            simple_sim = float(np.max(cosine_similarity([emb], self.ref_embeddings['simple'])[0]))
            complex_sim = float(np.max(cosine_similarity([emb], self.ref_embeddings['complex'])[0]))
            features['semantic_complexity'] = complex_sim - simple_sim
            features['simple_similarity'] = simple_sim
            features['complex_similarity'] = complex_sim
        else:
            features['semantic_complexity'] = 0.0
            features['simple_similarity'] = 0.0
            features['complex_similarity'] = 0.0

        return features

    # ------------------------------------------------------------------
    # Batch extraction (used at training time — much faster)
    # ------------------------------------------------------------------
    def batch_extract_vectors(self, queries: List[str], batch_size: int = 256) -> np.ndarray:
        """
        Extract feature vectors for a list of queries efficiently.
        Encodes ALL queries in a single batched call to SentenceTransformer
        instead of one-by-one, which is 50-100x faster for large datasets.
        """
        n = len(queries)

        # --- Lexical features (fast, no model needed) ---
        lexical = np.array(
            [self._lexical_vector(q) for q in queries],
            dtype=np.float32
        )  # shape: (n, 10)

        if not self.has_model or not self.ref_embeddings:
            # Pad semantic columns with zeros
            semantic = np.zeros((n, 3), dtype=np.float32)
        else:
            print(f"  Encoding {n} queries with SentenceTransformer (batch_size={batch_size})...")
            # Single batched encode call — this is the key fix
            embeddings = self.embedder.encode(
                queries,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )  # shape: (n, 384)

            simple_sims = cosine_similarity(embeddings, self.ref_embeddings['simple'])  # (n, 5)
            complex_sims = cosine_similarity(embeddings, self.ref_embeddings['complex'])  # (n, 5)

            simple_max = simple_sims.max(axis=1)   # (n,)
            complex_max = complex_sims.max(axis=1)  # (n,)

            semantic = np.stack(
                [complex_max - simple_max, simple_max, complex_max],
                axis=1
            ).astype(np.float32)  # shape: (n, 3)

        return np.concatenate([lexical, semantic], axis=1)  # (n, 13)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_lexical(self, query: str) -> Dict:
        """Extract non-semantic features from a single query."""
        query_lower = query.lower()
        words = query.split()
        word_set = set(query_lower.split())

        return {
            'word_count': len(words),
            'sentence_count': len([s for s in re.split(r'[.!?]+', query) if s.strip()]),
            'has_code': bool(re.search(r'```|def\s+\w+|class\s+\w+|import\s+\w+', query)),
            'has_technical_terms': bool(word_set & self.technical_terms),
            'has_numbers': bool(re.search(r'\d+', query)),
            'question_depth': min(query.count('?') + query.count(',') // 2, 5),
            'is_multipart': query.count('?') > 1 or bool(re.search(r'\b(also|additionally|and)\b', query_lower)),
            'requires_reasoning': bool(word_set & self.reasoning_keywords),
            'is_analysis': bool(word_set & self.analysis_keywords),
            'comma_count': query.count(','),
        }

    def _lexical_vector(self, query: str) -> np.ndarray:
        """Return lexical features as a float32 array (10 values)."""
        f = self._extract_lexical(query)
        return np.array(
            [float(f[k]) if not isinstance(f[k], bool) else float(f[k])
             for k in self.FEATURE_ORDER[:10]],
            dtype=np.float32
        )

    def extract_vector(self, features: Dict) -> np.ndarray:
        """Convert a features dict to a numpy vector (used at inference time)."""
        return np.array(
            [float(features.get(f, 0)) for f in self.FEATURE_ORDER],
            dtype=np.float32
        )