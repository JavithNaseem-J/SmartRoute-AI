import asyncio
import hashlib
import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable

import lightgbm
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

sys.path.append(str(Path(__file__).parent.parent))

from src.routing.classifier import ComplexityClassifier
from src.routing.features import FeatureExtractor


ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "classifiers" / "complexity_classifier.pkl"
METRICS_PATH = ROOT / "models" / "classifiers" / "complexity_classifier.metrics.json"
EVAL_PATH = ROOT / "data" / "evaluation" / "routing_eval.json"


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def _append_unique(
    rows: list[tuple[str, int]], seen: set[str], queries: Iterable[str], label: int
) -> None:
    for query in queries:
        normalized = normalize_query(query)
        if normalized and normalized not in seen:
            seen.add(normalized)
            rows.append((query.strip(), label))


def get_training_data() -> tuple[list[str], list[int]]:
    """Build a deterministic, duplicate-free synthetic training corpus."""
    subjects = [
        "artificial intelligence",
        "Python",
        "database indexing",
        "REST APIs",
        "Docker",
        "Kubernetes",
        "cloud storage",
        "machine learning",
        "neural networks",
        "SQL joins",
        "Git branching",
        "network latency",
        "encryption",
        "load balancing",
        "message queues",
        "React",
        "data pipelines",
        "vector databases",
        "retrieval augmented generation",
        "continuous integration",
        "operating systems",
        "distributed caching",
        "authentication",
        "observability",
        "unit testing",
        "graph algorithms",
        "object-oriented programming",
        "HTTP",
        "data normalization",
        "serverless computing",
    ]
    rows: list[tuple[str, int]] = []
    seen: set[str] = set()

    template_groups = (
        (
            0,
            [
                "What is {subject}?",
                "Define {subject}.",
                "What does {subject} mean?",
                "Name three features of {subject}.",
                "Give a one-sentence summary of {subject}.",
                "Is {subject} commonly used in software development?",
            ],
        ),
        (
            1,
            [
                "Explain how {subject} works for a beginner.",
                "Give a short practical example of using {subject}.",
                "What are the main advantages and limitations of {subject}?",
                "Provide a step-by-step guide to set up {subject} for a small project.",
                "Compare {subject} with a traditional alternative and give two differences.",
                "How would you troubleshoot a basic problem involving {subject}?",
            ],
        ),
        (
            2,
            [
                "Design a production architecture for {subject}, including failure recovery, security, scalability, and operational trade-offs.",
                "Analyze competing approaches to {subject} under strict latency, cost, and consistency constraints, then recommend one.",
                "Evaluate three architectures for {subject}, quantify their risks, and propose a staged migration plan.",
                "Diagnose a multi-region failure involving {subject} and develop a fault-tolerant remediation strategy.",
                "Create a comprehensive governance and observability plan for {subject}, considering compliance and incident response.",
                "Critique the design of {subject} for a high-traffic system and synthesize improvements with explicit trade-offs.",
            ],
        ),
    )
    for label, templates in template_groups:
        _append_unique(
            rows,
            seen,
            (template.format(subject=subject) for subject in subjects for template in templates),
            label,
        )

    return [query for query, _ in rows], [label for _, label in rows]


def load_evaluation_data() -> tuple[list[str], list[int]]:
    payload = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    queries = [str(item["query"]) for item in payload]
    labels = [int(item["label"]) for item in payload]
    if len(queries) != len({normalize_query(query) for query in queries}):
        raise RuntimeError("Routing evaluation set contains duplicate queries.")
    return queries, labels


def dataset_fingerprint(queries: list[str], labels: list[int]) -> str:
    rows = [f"{label}\t{normalize_query(query)}" for query, label in zip(queries, labels)]
    return hashlib.sha256("\n".join(sorted(rows)).encode("utf-8")).hexdigest()


async def main() -> None:
    train_queries, train_labels = get_training_data()
    eval_queries, eval_labels = load_evaluation_data()

    overlap = set(map(normalize_query, train_queries)) & set(map(normalize_query, eval_queries))
    if overlap:
        raise RuntimeError(
            f"Training/evaluation leakage detected: {len(overlap)} overlapping queries."
        )
    duplicates = len(train_queries) - len(set(map(normalize_query, train_queries)))
    if duplicates:
        raise RuntimeError(f"Training data contains {duplicates} duplicate queries.")

    extractor = FeatureExtractor()
    X_train = await extractor.batch_extract_vectors(train_queries)
    X_eval = await extractor.batch_extract_vectors(eval_queries)
    y_train = np.asarray(train_labels)
    y_eval = np.asarray(eval_labels)

    classifier = ComplexityClassifier(auto_train=False)
    train_accuracy = classifier.train(X_train, y_train)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted",
        )
        predictions = classifier.model.predict(classifier.scaler.transform(X_eval))

    report = classification_report(
        y_eval,
        predictions,
        target_names=classifier.classes,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "artifact_version": classifier.ARTIFACT_VERSION,
        "dataset_fingerprint": dataset_fingerprint(train_queries, train_labels),
        "training_examples": len(train_queries),
        "unique_training_examples": len(set(map(normalize_query, train_queries))),
        "evaluation_examples": len(eval_queries),
        "train_evaluation_overlap": 0,
        "class_counts": dict(Counter(train_labels)),
        "train_accuracy": round(float(train_accuracy), 6),
        "evaluation_accuracy": round(float(accuracy_score(y_eval, predictions)), 6),
        "evaluation_macro_f1": round(float(f1_score(y_eval, predictions, average="macro")), 6),
        "confusion_matrix": confusion_matrix(y_eval, predictions).tolist(),
        "classification_report": report,
        "feature_order": list(FeatureExtractor.FEATURE_ORDER),
        "semantic_centroids_loaded": bool(extractor.ref_embeddings),
        "versions": {
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "numpy": np.__version__,
        },
    }

    classifier.save(MODEL_PATH, metadata=metrics)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    summary = {key: value for key, value in metrics.items() if key != "classification_report"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
