import re
from typing import Dict, List

class FeatureExtractor:
    """Extract features from queries for routing decisions"""
    
    def __init__(self):
        self.code_patterns = [
            r'```', r'def\s+\w+', r'class\s+\w+', r'import\s+\w+',
            r'function\s+\w+', r'const\s+\w+', r'let\s+\w+', r'var\s+\w+'
        ]
        
        self.technical_terms = {
            'algorithm', 'database', 'api', 'server', 'client', 'cache',
            'optimization', 'deployment', 'architecture', 'framework',
            'library', 'dependency', 'repository', 'integration', 'authentication',
            'authorization', 'encryption', 'decryption', 'protocol', 'endpoint'
        }
        
        self.reasoning_keywords = {
            'why', 'how', 'explain', 'analyze', 'compare', 'evaluate',
            'reasoning', 'logic', 'proof', 'demonstrate', 'justify'
        }
        
        self.creative_keywords = {
            'story', 'poem', 'creative', 'write', 'imagine', 'describe',
            'narrative', 'character', 'plot', 'scene'
        }
    
    def extract(self, query: str, context: Dict = None) -> Dict:
        """Extract all features from query"""
        query_lower = query.lower()
        
        features = {
            'token_count': self._estimate_tokens(query),
            'char_count': len(query),
            'word_count': len(query.split()),
            'has_code': self._has_code(query),
            'has_technical_terms': self._has_technical_terms(query_lower),
            'question_depth': self._calculate_question_depth(query),
            'is_multipart': self._is_multipart(query),
            'requires_reasoning': self._requires_reasoning(query_lower),
            'is_creative': self._is_creative(query_lower),
            'is_analysis': self._is_analysis(query_lower),
            'has_numbers': bool(re.search(r'\d+', query)),
            'has_urls': bool(re.search(r'https?://', query)),
            'num_sentences': len(re.split(r'[.!?]+', query)),
        }
        
        # Add context features if available
        if context:
            features['has_context'] = True
            features['context_length'] = context.get('length', 0)
        else:
            features['has_context'] = False
            features['context_length'] = 0
        
        return features
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count"""
        # Approximate: 1 token ≈ 4 characters
        return len(text) // 4
    
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
        question_marks = query.count('?')
        depth += question_marks
        
        # Nested clauses
        depth += query.count(',') // 2
        
        # Conditional/hypothetical
        if re.search(r'\bif\b|\bwhen\b|\bwould\b|\bcould\b', query, re.IGNORECASE):
            depth += 1
        
        # Comparison words
        if re.search(r'\bcompare\b|\bvs\b|\bversus\b|\bdifference\b', query, re.IGNORECASE):
            depth += 1
        
        return depth
    
    def _is_multipart(self, query: str) -> bool:
        """Check if query has multiple parts"""
        # Look for enumeration
        if re.search(r'\b(first|second|third|1\.|2\.|3\.)', query, re.IGNORECASE):
            return True
        
        # Multiple questions
        if query.count('?') > 1:
            return True
        
        # Explicit parts
        if re.search(r'\b(also|additionally|furthermore|moreover)\b', query, re.IGNORECASE):
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
            'examine', 'investigate', 'study', 'interpret'
        }
        words = set(query_lower.split())
        return bool(words.intersection(analysis_terms))