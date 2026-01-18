import re
from typing import Dict
import numpy as np


class FeatureExtractor:
    """Extract features from queries for routing decisions."""
    
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
    
    def extract(self, query: str) -> Dict:
        """Extract 10 key features from query."""
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
    
    def extract_vector(self, features: Dict) -> np.ndarray:
        """Convert features dict to numpy vector for ML model."""
        feature_order = [
            'word_count', 'sentence_count', 'has_code', 'has_technical_terms',
            'has_numbers', 'question_depth', 'is_multipart', 'requires_reasoning',
            'is_analysis', 'comma_count'
        ]
        
        return np.array([
            int(features.get(f, 0)) if isinstance(features.get(f, 0), bool) 
            else features.get(f, 0) 
            for f in feature_order
        ], dtype=np.float32)
    