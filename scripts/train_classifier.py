import numpy as np
from sklearn.model_selection import train_test_split
from src.routing.classifier import ComplexityClassifier
from src.routing.features import FeatureExtractor
from src.utils.logging import logger


def generate_synthetic_data():
    """Generate synthetic training data"""
    
    # Simple queries
    simple_queries = [
        "What is machine learning?",
        "Define neural network",
        "Who invented Python?",
        "When was AI created?",
        "Where is Dubai located?",
        "What does API stand for?",
        "Name the capital of UAE",
        "List programming languages",
        "What is 2+2?",
        "Define photosynthesis",
    ] * 50
    
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
    ] * 50
    
    # Complex queries
    complex_queries = [
        "Analyze the impact of AI on job markets and provide evaluation",
        "Evaluate the ethical implications of autonomous vehicles",
        "Synthesize research findings on climate change policy",
        "Compare different approaches to AGI development",
        "Analyze the relationship between data privacy and AI",
        "Evaluate quantum computing theories and implications",
        "Argue for AI regulation using multiple perspectives",
        "Synthesize renewable energy research comprehensively",
        "Analyze drug development from discovery to market",
        "Evaluate economic models for developing nations",
    ] * 50
    
    all_queries = simple_queries + medium_queries + complex_queries
    labels = (
        [0] * len(simple_queries) +
        [1] * len(medium_queries) +
        [2] * len(complex_queries)
    )
    
    return all_queries, labels


def main():
    logger.info("Generating training data...")
    queries, labels = generate_synthetic_data()
    
    logger.info(f"Generated {len(queries)} examples")
    logger.info(f"Simple: {labels.count(0)}, Medium: {labels.count(1)}, Complex: {labels.count(2)}")
    
    # Extract features
    logger.info("Extracting features...")
    feature_extractor = FeatureExtractor()
    
    X = []
    for query in queries:
        features = feature_extractor.extract(query)
        feature_vector = feature_extractor.extract_vector_features(features)
        X.append(feature_vector)
    
    X = np.array(X)
    y = np.array(labels)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train
    logger.info("Training classifier...")
    classifier = ComplexityClassifier()
    train_accuracy = classifier.train(X_train, y_train)
    
    # Evaluate
    X_test_scaled = classifier.scaler.transform(X_test)
    test_accuracy = classifier.model.score(X_test_scaled, y_test)
    
    logger.info(f"Training accuracy: {train_accuracy:.2%}")
    logger.info(f"Test accuracy: {test_accuracy:.2%}")
    
    # Feature importance
    importance = classifier.get_feature_importance()
    logger.info("Top features:")
    for feature, score in list(importance.items())[:5]:
        logger.info(f"  {feature}: {score:.4f}")
    
    # Save
    model_path = "models/classifier/complexity_classifier.pkl"
    classifier.save(model_path)
    logger.info(f"✓ Model saved to {model_path}")
    
    # Test examples
    logger.info("\nTesting on examples:")
    examples = [
        "What is AI?",
        "How does machine learning work?",
        "Analyze ethical AI implications across stakeholders"
    ]
    
    for example in examples:
        complexity, confidence = classifier.predict(example)
        logger.info(f"  '{example[:40]}...' → {complexity} ({confidence:.2%})")
    
    logger.info("\n✓ Training complete!")


if __name__ == "__main__":
    main()