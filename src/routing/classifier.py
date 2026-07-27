from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
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
            class_weight="balanced",
            verbose=-1,  # Suppress LightGBM warnings
        )

        # Complexity classes: 0=simple, 1=medium, 2=complex
        self.classes = ["simple", "medium", "complex"]
        self.is_trained = False

        if model_path and model_path.exists():
            self.load(model_path)
        else:
            self._auto_train()

    def _auto_train(self):
        """Auto-train on synthetic features if pre-trained model file is absent."""
        from src.utils.logger import logger

        logger.info("Pre-trained classifier not found — running fast auto-training fallback...")
        import asyncio
        import random

        subjects = ["AI", "Python", "Machine Learning", "Data Science", "SQL", "Docker", "API"]
        actions_simple = ["What is", "Define", "Who created", "When was", "List features of"]
        actions_medium = ["How does", "Why use", "Explain concept of", "Describe benefits of"]
        actions_complex = [
            "Analyze impact of",
            "Evaluate performance of",
            "Critique architectural design of",
        ]

        queries = []
        labels = []
        for _ in range(100):
            queries.append(f"{random.choice(actions_simple)} {random.choice(subjects)}?")
            labels.append(0)
            queries.append(f"{random.choice(actions_medium)} {random.choice(subjects)} in tech?")
            labels.append(1)
            queries.append(
                f"{random.choice(actions_complex)} {random.choice(subjects)}, providing comprehensive trade-off analysis."
            )
            labels.append(2)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # In async loop context, run feature extractions via sync call
            X_list = [self.feature_extractor.extract_sync(q) for q in queries]
        else:
            X_list = [asyncio.run(self.feature_extractor.extract(q)) for q in queries]

        X = np.array(X_list)
        y = np.array(labels)
        self.train(X, y)

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
        # pyrefly: ignore [bad-argument-type]
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Calculate accuracy
        accuracy = self.model.score(X_scaled, y)

        return float(accuracy)

    async def predict(self, query: str) -> Tuple[str, float]:
        if not self.is_trained:
            self._auto_train()

        # Extract features (now an async network call)
        features = await self.feature_extractor.extract(query)
        feature_vector = self.feature_extractor.extract_vector(features)

        # Scale
        X = feature_vector.reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Predict (very fast, safe on main thread)
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]

        complexity = self.classes[prediction]
        confidence = probabilities[prediction]

        return complexity, confidence

    def get_feature_importance(self) -> dict:
        """Return feature importance scores mapped to feature names."""
        if not self.is_trained:
            return {}
        importances = self.model.feature_importances_
        feature_names = self.feature_extractor.FEATURE_ORDER
        return {name: float(score) for name, score in zip(feature_names, importances)}

    def save(self, path: Path):
        """Save trained model using joblib (safer than pickle)"""
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "is_trained": self.is_trained,
        }

        joblib.dump(model_data, path)

    def load(self, path: Path):
        """Load trained model using joblib"""
        model_data = joblib.load(path)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.is_trained = model_data["is_trained"]
