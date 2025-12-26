import pickle
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .features import FeatureExtractor


class ComplexityClassifier:
    """ML-based query complexity classifier"""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        # Complexity classes: 0=simple, 1=medium, 2=complex
        self.classes = ['simple', 'medium', 'complex']
        self.is_trained = False
        
        if model_path and model_path.exists():
            self.load(model_path)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Train the classifier
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,) - integers 0, 1, 2
        
        Returns:
            Training accuracy
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate accuracy
        accuracy = self.model.score(X_scaled, y)
        
        return accuracy
    
    def predict(self, query: str) -> Tuple[str, float]:
        """
        Predict complexity of a query
        
        Args:
            query: Input query string
        
        Returns:
            (complexity, confidence) tuple
            complexity: 'simple', 'medium', or 'complex'
            confidence: 0.0 to 1.0
        """
        if not self.is_trained:
            # Fallback to rule-based if not trained
            return self._rule_based_classification(query)
        
        # Extract features
        features = self.feature_extractor.extract(query)
        feature_vector = self.feature_extractor.extract_vector(features)
        
        # Scale
        X = feature_vector.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        complexity = self.classes[prediction]
        confidence = probabilities[prediction]
        
        return complexity, confidence
    
    def _rule_based_classification(self, query: str) -> Tuple[str, float]:
        """Fallback rule-based classification"""
        features = self.feature_extractor.extract(query)
        
        word_count = features['word_count']
        has_reasoning = features['requires_reasoning']
        is_analysis = features['is_analysis']
        question_depth = features['question_depth']
        
        # Simple queries
        if word_count <= 15 and question_depth <= 1 and not has_reasoning:
            return 'simple', 0.8
        
        # Complex queries
        if (word_count >= 30 or question_depth >= 3 or 
            has_reasoning or is_analysis):
            return 'complex', 0.7
        
        # Medium queries (default)
        return 'medium', 0.6
    
    def get_feature_importance(self) -> dict:
        """Get feature importance scores"""
        if not self.is_trained:
            return {}
        
        feature_names = [
            'char_count', 'word_count', 'sentence_count', 'avg_word_length',
            'has_code', 'has_technical_terms', 'has_numbers', 'has_urls',
            'question_marks', 'question_depth', 'is_multipart',
            'requires_reasoning', 'is_creative', 'is_analysis',
            'comma_count', 'semicolon_count', 'exclamation_count'
        ]
        
        importances = self.model.feature_importances_
        
        return dict(zip(feature_names, importances))
    
    def save(self, path: Path):
        """Save trained model"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, path: Path):
        """Load trained model"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']


def generate_synthetic_training_data():
    """Generate synthetic training data for classifier"""
    
    # Simple queries
    simple_queries = [
        "What is machine learning?",
        "Define neural network",
        "Who invented Python?",
        "When was AI created?",
        "Where is Dubai?",
        "What does API stand for?",
        "Name the capital",
        "List programming languages",
        "What is 2+2?",
        "Define photosynthesis",
    ] * 50  # 500 examples
    
    # Medium queries
    medium_queries = [
        "How does machine learning work?",
        "Why is deep learning popular?",
        "Explain the difference between AI and ML",
        "Compare supervised and unsupervised learning",
        "Describe how neural networks learn",
        "What are the benefits of cloud computing?",
        "How do transformers work in NLP?",
        "Explain gradient descent algorithm",
        "Why use transfer learning?",
        "Summarize the history of AI",
    ] * 50  # 500 examples
    
    # Complex queries
    complex_queries = [
        "Analyze the impact of AI on job markets and provide evaluation",
        "Evaluate the ethical implications of autonomous vehicles in detail",
        "Synthesize research findings on climate change policy",
        "Compare different approaches to AGI development",
        "Analyze the relationship between data privacy and AI",
        "Evaluate quantum computing theories and implications",
        "Argue for AI regulation using multiple perspectives",
        "Synthesize renewable energy research comprehensively",
        "Analyze drug development from discovery to market",
        "Evaluate economic models for developing nations",
    ] * 50  # 500 examples
    
    queries = simple_queries + medium_queries + complex_queries
    labels = [0] * len(simple_queries) + [1] * len(medium_queries) + [2] * len(complex_queries)
    
    return queries, labels