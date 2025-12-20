import numpy as np
from typing import Dict, List
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from .features import FeatureExtractor

class QueryClassifier:
    """Classifies queries to determine routing strategy"""
    
    def __init__(self, model_path: str = None):
        self.feature_extractor = FeatureExtractor()
        self.classifier = None
        self.complexity_thresholds = {
            'simple': 0.3,
            'medium': 0.6,
            'complex': 0.9
        }
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.classifier = RandomForestClassifier(n_estimators=100)
    
    def extract_features(self, query: str, context: Dict = None) -> Dict:
        """Extract features from query"""
        return self.feature_extractor.extract(query, context)
    
    def classify_complexity(self, query: str, context: Dict = None) -> str:
        """Classify query complexity"""
        features = self.extract_features(query, context)
        
        # Calculate complexity score based on features
        score = self._calculate_complexity_score(features)
        
        if score < self.complexity_thresholds['simple']:
            return 'simple'
        elif score < self.complexity_thresholds['medium']:
            return 'medium'
        elif score < self.complexity_thresholds['complex']:
            return 'complex'
        else:
            return 'expert'
    
    def _calculate_complexity_score(self, features: Dict) -> float:
        """Calculate complexity score from features"""
        score = 0.0
        
        # Token length factor
        token_count = features['token_count']
        if token_count > 500:
            score += 0.3
        elif token_count > 200:
            score += 0.2
        elif token_count > 100:
            score += 0.1
        
        # Technical terms
        if features['has_code']:
            score += 0.2
        if features['has_technical_terms']:
            score += 0.15
        
        # Question complexity
        if features['question_depth'] > 2:
            score += 0.2
        elif features['question_depth'] > 1:
            score += 0.1
        
        # Multi-part questions
        if features['is_multipart']:
            score += 0.15
        
        # Reasoning required
        if features['requires_reasoning']:
            score += 0.2
        
        # Creative writing
        if features['is_creative']:
            score += 0.15
        
        return min(score, 1.0)
    
    def classify_task_type(self, query: str) -> str:
        """Classify the type of task"""
        features = self.extract_features(query)
        
        if features['has_code']:
            return 'code_generation'
        elif features['is_creative']:
            return 'creative_writing'
        elif features['is_analysis']:
            return 'analysis'
        elif features['requires_reasoning']:
            return 'reasoning'
        else:
            return 'qa'
    
    def predict_quality_requirement(self, query: str, context: Dict = None) -> str:
        """Predict required quality level"""
        complexity = self.classify_complexity(query, context)
        task_type = self.classify_task_type(query)
        
        # High-stakes tasks need high quality
        high_quality_tasks = ['code_generation', 'analysis', 'reasoning']
        
        if complexity in ['complex', 'expert']:
            return 'high'
        elif task_type in high_quality_tasks:
            return 'medium'
        else:
            return 'low'
    
    def train(self, X: List[Dict], y: List[str]):
        """Train classifier on labeled data"""
        # Convert features to matrix
        feature_matrix = self._features_to_matrix(X)
        self.classifier.fit(feature_matrix, y)
    
    def _features_to_matrix(self, features_list: List[Dict]) -> np.ndarray:
        """Convert list of feature dicts to matrix"""
        feature_names = [
            'token_count', 'has_code', 'has_technical_terms',
            'question_depth', 'is_multipart', 'requires_reasoning',
            'is_creative', 'is_analysis'
        ]
        
        matrix = []
        for features in features_list:
            row = [
                features.get(name, 0) if isinstance(features.get(name), (int, float))
                else int(features.get(name, False))
                for name in feature_names
            ]
            matrix.append(row)
        
        return np.array(matrix)
    
    def save_model(self, path: str):
        """Save trained classifier"""
        with open(path, 'wb') as f:
            pickle.dump(self.classifier, f)
    
    def load_model(self, path: str):
        """Load trained classifier"""
        with open(path, 'rb') as f:
            self.classifier = pickle.load(f)