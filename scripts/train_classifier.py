import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.routing.classifier import ComplexityClassifier, generate_synthetic_training_data
from src.routing.features import FeatureExtractor


def main():
    print("="*60)
    print("TRAINING QUERY COMPLEXITY CLASSIFIER")
    print("="*60)
    
    # Step 1: Generate training data
    print("\n[1/5] Generating synthetic training data...")
    queries, labels = generate_synthetic_training_data()
    
    print(f"✓ Generated {len(queries)} examples:")
    print(f"  - Simple: {labels.count(0)} queries")
    print(f"  - Medium: {labels.count(1)} queries")
    print(f"  - Complex: {labels.count(2)} queries")
    
    # Step 2: Extract features
    print("\n[2/5] Extracting features...")
    feature_extractor = FeatureExtractor()
    
    X = []
    for i, query in enumerate(queries):
        if i % 200 == 0:
            print(f"  Processed {i}/{len(queries)} queries...")
        
        features = feature_extractor.extract(query)
        feature_vector = feature_extractor.extract_vector(features)
        X.append(feature_vector)
    
    X = np.array(X)
    y = np.array(labels)
    
    print(f"✓ Extracted features: {X.shape}")
    
    # Step 3: Split data
    print("\n[3/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"✓ Train set: {len(X_train)} examples")
    print(f"✓ Test set: {len(X_test)} examples")
    
    # Step 4: Train classifier
    print("\n[4/5] Training Random Forest classifier...")
    classifier = ComplexityClassifier()
    
    train_accuracy = classifier.train(X_train, y_train)
    print(f"✓ Training accuracy: {train_accuracy:.2%}")
    
    # Evaluate on test set
    X_test_scaled = classifier.scaler.transform(X_test)
    test_accuracy = classifier.model.score(X_test_scaled, y_test)
    print(f"✓ Test accuracy: {test_accuracy:.2%}")
    
    # Feature importance
    print("\n[5/5] Analyzing feature importance...")
    importance = classifier.get_feature_importance()
    
    print("\nTop 10 Most Important Features:")
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, score) in enumerate(sorted_features[:10], 1):
        print(f"  {i}. {feature}: {score:.4f}")
    
    # Save model
    save_path = Path("models/classifiers/complexity_classifier.pkl")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    classifier.save(save_path)
    print(f"\n✓ Model saved to: {save_path}")
    
    # Test on real examples
    print("\n" + "="*60)
    print("TESTING ON EXAMPLE QUERIES")
    print("="*60)
    
    test_examples = [
        ("What is AI?", "simple"),
        ("Define machine learning", "simple"),
        ("How does neural network training work?", "medium"),
        ("Explain the difference between AI and ML", "medium"),
        ("Analyze the ethical implications of AI in healthcare comprehensively", "complex"),
        ("Evaluate different approaches to AGI development with detailed reasoning", "complex")
    ]
    
    correct = 0
    for query, expected in test_examples:
        predicted, confidence = classifier.predict(query)
        is_correct = "✓" if predicted == expected else "✗"
        
        if predicted == expected:
            correct += 1
        
        print(f"\n{is_correct} Query: '{query[:60]}...'")
        print(f"  Expected: {expected}")
        print(f"  Predicted: {predicted} (confidence: {confidence:.2%})")
    
    accuracy = correct / len(test_examples)
    print(f"\n✓ Test Examples Accuracy: {accuracy:.2%} ({correct}/{len(test_examples)})")
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel saved at: {save_path}")
    print("You can now run the inference pipeline with:")
    print("  python -c 'from src.pipeline.inference import InferencePipeline; p = InferencePipeline(); print(p.run(\"What is AI?\"))'")
    

if __name__ == "__main__":
    main()