from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import warnings

import joblib
import numpy as np
import sklearn
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

from src.routing.features import FeatureExtractor


class ComplexityClassifier:
    """ML-based query complexity classifier"""

    ARTIFACT_VERSION = 2

    def __init__(self, model_path: Optional[Path] = None, auto_train: bool = True):
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
        self.metadata: Dict[str, Any] = {}

        if model_path and model_path.exists():
            self.load(model_path)
        elif auto_train:
            self._auto_train()

    def _auto_train(self):
        """Auto-train on synthetic features if pre-trained model file is absent."""
        from src.utils.logger import logger

        logger.info("Pre-trained classifier not found — running fast auto-training fallback...")
        import asyncio
        import random

        rng = random.Random(42)

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
            queries.append(f"{rng.choice(actions_simple)} {rng.choice(subjects)}?")
            labels.append(0)
            queries.append(f"{rng.choice(actions_medium)} {rng.choice(subjects)} in tech?")
            labels.append(1)
            queries.append(
                f"{rng.choice(actions_complex)} {rng.choice(subjects)}, providing comprehensive trade-off analysis."
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
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but LGBMClassifier was fitted",
            )
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
        # LightGBM synthesizes Column_* names for ndarray training data and its
        # sklearn adapter emits a false warning when inference also uses ndarray.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but LGBMClassifier was fitted",
            )
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

    def save(self, path: Path, metadata: Optional[Dict[str, Any]] = None):
        """Save a versioned model bundle with training and feature metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "is_trained": self.is_trained,
            "artifact_version": self.ARTIFACT_VERSION,
            "feature_order": list(self.feature_extractor.FEATURE_ORDER),
            "sklearn_version": sklearn.__version__,
            "metadata": metadata or {},
        }

        joblib.dump(model_data, path)

    def load(self, path: Path):
        """Load trained model using joblib"""
        model_data = joblib.load(path)

        artifact_version = model_data.get("artifact_version")
        if artifact_version != self.ARTIFACT_VERSION:
            raise RuntimeError(
                "Classifier artifact is obsolete. Run `uv run python scripts/train_classifier.py`."
            )
        feature_order = model_data.get("feature_order")
        if feature_order != list(self.feature_extractor.FEATURE_ORDER):
            raise RuntimeError("Classifier feature schema does not match runtime feature order.")
        artifact_sklearn = model_data.get("sklearn_version")
        if artifact_sklearn != sklearn.__version__:
            raise RuntimeError(
                "Classifier scikit-learn version mismatch: "
                f"artifact={artifact_sklearn}, runtime={sklearn.__version__}. Retrain the artifact."
            )

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.is_trained = model_data["is_trained"]
        self.metadata = model_data.get("metadata", {})
