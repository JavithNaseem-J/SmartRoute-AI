import json
from pathlib import Path

import sklearn

from scripts.train_classifier import get_training_data, load_evaluation_data, normalize_query
from src.routing.classifier import ComplexityClassifier


ROOT = Path(__file__).resolve().parent.parent


def test_classifier_training_and_evaluation_sets_do_not_overlap():
    training, _ = get_training_data()
    evaluation, _ = load_evaluation_data()

    normalized_training = {normalize_query(query) for query in training}
    normalized_evaluation = {normalize_query(query) for query in evaluation}

    assert len(normalized_training) == len(training)
    assert normalized_training.isdisjoint(normalized_evaluation)


def test_classifier_artifact_matches_runtime_and_recorded_metrics():
    classifier = ComplexityClassifier(
        ROOT / "models" / "classifiers" / "complexity_classifier.pkl",
        auto_train=False,
    )
    metrics = json.loads(
        (ROOT / "models" / "classifiers" / "complexity_classifier.metrics.json").read_text(
            encoding="utf-8"
        )
    )

    assert classifier.is_trained
    assert classifier.metadata["artifact_version"] == classifier.ARTIFACT_VERSION
    assert metrics["versions"]["scikit_learn"] == sklearn.__version__
    assert metrics["train_evaluation_overlap"] == 0
    assert metrics["evaluation_macro_f1"] >= 0.9
