import re
from typing import Dict, List
import numpy as np


class FeatureExtractor:
    """Extract features from queries for routing decisions"""
    
    def __init__(self):
        # Code patterns
        self.code_patterns = [
            r'```', r'def\s+\w+', r'class\s+\w+', r'import\s+\w+',
            r'function\s+\w+', r'const\s+\w+', r'let\s+\w+', r'var\s+\w+'
        ]
        
        # Domain keywords
        self.technical_terms = {
            'algorithm', 'database', 'api', 'server', 'client', 'cache',
            'optimization', 'deployment', 'architecture', 'framework',
            'library', 'dependency', 'repository', 'integration'
        }
        
        self.reasoning_keywords = {
            'why', 'how', 'explain', 'analyze', 'compare', 'evaluate',
            'reasoning', 'logic', 'proof', 'demonstrate', 'justify'
        }
        
        self.creative_keywords = {
            'story', 'poem', 'creative', 'write', 'imagine', 'describe',
            'narrative', 'character', 'plot', 'scene'
        }
    
    def extract(self, query: str) -> Dict:
        """Extract all features from query"""
        query_lower = query.lower()
        words = query.split()
        sentences = re.split(r'[.!?]+', query)
        
        features = {
            # Length features
            'char_count': len(query),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            
            # Complexity indicators
            'has_code': self._has_code(query),
            'has_technical_terms': self._has_technical_terms(query_lower),
            'has_numbers': bool(re.search(r'\d+', query)),
            'has_urls': bool(re.search(r'https?://', query)),
            
            # Question complexity
            'question_marks': query.count('?'),
            'question_depth': self._calculate_question_depth(query),
            'is_multipart': self._is_multipart(query),
            
            # Content type
            'requires_reasoning': self._requires_reasoning(query_lower),
            'is_creative': self._is_creative(query_lower),
            'is_analysis': self._is_analysis(query_lower),
            
            # Punctuation
            'comma_count': query.count(','),
            'semicolon_count': query.count(';'),
            'exclamation_count': query.count('!'),
        }
        
        return features
    
    def extract_vector(self, features: Dict) -> np.ndarray:
        """Convert features dict to numpy vector for ML model"""
        feature_order = [
            'char_count', 'word_count', 'sentence_count', 'avg_word_length',
            'has_code', 'has_technical_terms', 'has_numbers', 'has_urls',
            'question_marks', 'question_depth', 'is_multipart',
            'requires_reasoning', 'is_creative', 'is_analysis',
            'comma_count', 'semicolon_count', 'exclamation_count'
        ]
        
        vector = []
        for feat in feature_order:
            val = features.get(feat, 0)
            # Convert boolean to int
            if isinstance(val, bool):
                val = int(val)
            vector.append(val)
        
        return np.array(vector, dtype=np.float32)
    
    def _has_code(self, query: str) -> bool:
        """Check if query contains code"""
        for pattern in self.code_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _has_technical_terms(self, query_lower: str) -> bool:
        """Check for technical terminology"""
        words = set(query_lower.split())
        return bool(words.intersection(self.technical_terms))
    
    def _calculate_question_depth(self, query: str) -> int:
        """Calculate depth/complexity of question"""
        depth = 0
        
        # Multiple questions
        depth += query.count('?')
        
        # Nested clauses (commas)
        depth += query.count(',') // 2
        
        # Conditional/hypothetical
        if re.search(r'\b(if|when|would|could|should)\b', query, re.IGNORECASE):
            depth += 1
        
        # Comparison words
        if re.search(r'\b(compare|vs|versus|difference|better|worse)\b', query, re.IGNORECASE):
            depth += 1
        
        return min(depth, 5)  # Cap at 5
    
    def _is_multipart(self, query: str) -> bool:
        """Check if query has multiple parts"""
        # Look for enumeration
        if re.search(r'\b(first|second|third|1\.|2\.|3\.)\b', query, re.IGNORECASE):
            return True
        
        # Multiple questions
        if query.count('?') > 1:
            return True
        
        # Explicit parts
        if re.search(r'\b(also|additionally|furthermore|moreover|and)\b', query, re.IGNORECASE):
            return True
        
        return False
    
    def _requires_reasoning(self, query_lower: str) -> bool:
        """Check if query requires reasoning"""
        words = set(query_lower.split())
        return bool(words.intersection(self.reasoning_keywords))
    
    def _is_creative(self, query_lower: str) -> bool:
        """Check if query is creative writing"""
        words = set(query_lower.split())
        return bool(words.intersection(self.creative_keywords))
    
    def _is_analysis(self, query_lower: str) -> bool:
        """Check if query requires analysis"""
        analysis_terms = {
            'analyze', 'analysis', 'evaluate', 'assess', 'review',
            'examine', 'investigate', 'study', 'interpret', 'critique'
        }
        words = set(query_lower.split())
        return bool(words.intersection(analysis_terms))