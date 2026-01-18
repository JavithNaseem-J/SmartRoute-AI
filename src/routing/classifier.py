import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from src.routing.features import FeatureExtractor


class ComplexityClassifier:
    """ML-based query complexity classifier"""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.model = LGBMClassifier(
            n_estimators=50,
            max_depth=4,
            num_leaves=15,
            random_state=42,
            class_weight='balanced',
            verbose=-1  # Suppress LightGBM warnings
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

        if not self.is_trained:
            raise RuntimeError(
                "Classifier not trained. Run 'python scripts/train_classifier.py' first."
            )
        
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
    
    def get_feature_importance(self) -> dict:
        """Get feature importance scores"""
        if not self.is_trained:
            return {}
        
        feature_names = [
            'word_count', 'sentence_count', 'has_code', 'has_technical_terms',
            'has_numbers', 'question_depth', 'is_multipart', 'requires_reasoning',
            'is_analysis', 'comma_count'
        ]
        
        importances = self.model.feature_importances_
        
        return dict(zip(feature_names, importances))
    
    def save(self, path: Path):
        """Save trained model using joblib (safer than pickle)"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, path)
    
    def load(self, path: Path):
        """Load trained model using joblib"""
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']