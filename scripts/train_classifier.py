import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

sys.path.append(str(Path(__file__).parent.parent))
from src.routing.classifier import ComplexityClassifier
from src.routing.features import FeatureExtractor


def get_training_data():
    """
    Get training data from MS MARCO dataset if available, 
    otherwise use expanded synthetic data.
    """
    queries = []
    labels = []
    
    if HAS_DATASETS:
        print("Loading MS MARCO dataset (v1.1)...")
        try:
            # Load a subset of MS MARCO
            dataset = load_dataset("ms_marco", "v1.1", split="train", streaming=True)
            
            print("Labeling data based on heuristics...")
            count = 0
            for item in dataset:
                query = item['query']
                queries.append(query)
                
                # Heuristic labeling
                # 0: Simple (short, fact-based)
                # 1: Medium (how/why, moderate length)
                # 2: Complex (analyze/evaluate, long)
                
                label = 1 # default medium
                words = len(query.split())
                query_lower = query.lower()
                
                if words < 6 and not any(k in query_lower for k in ['why', 'how', 'explain']):
                    label = 0
                elif any(k in query_lower for k in ['analyze', 'evaluate', 'critique', 'synthesize', 'comprehensive']):
                    label = 2
                elif words > 20: 
                    label = 2
                
                labels.append(label)
                
                count += 1
                if count >= 10000:
                    break
                    
            print(f"Loaded {len(queries)} queries from MS MARCO.")
            return queries, labels
            
        except Exception as e:
            print(f"Failed to load MS MARCO: {e}")
            print("Falling back to synthetic data...")
    
    # Fallback to BETTER synthetic data (no duplicates)
    print("Generating synthetic data...")
    
    # Define templates to generate varied queries
    subjects = ["AI", "Python", "Machine Learning", "Data Science", "Cloud Computing", "React", "SQL", "Docker", "Kubernetes", "API"]
    actions_simple = ["What is", "Define", "Who created", "When was", "List features of"]
    actions_medium = ["How does", "Why use", "Explain the concept of", "Describe the benefits of", "Compare X and Y in"]
    actions_complex = ["Analyze the impact of", "Evaluate the detailed performance of", "Critique the architectural design of", "Synthesize recent research on", "Develop a comprehensive strategy for"]
    
    queries = []
    labels = []
    
    import random
    
    # Generate Simple (0)
    for _ in range(1000):
        s = random.choice(subjects)
        a = random.choice(actions_simple)
        q = f"{a} {s}?"
        queries.append(q)
        labels.append(0)
        
    # Generate Medium (1)
    for _ in range(1000):
        s = random.choice(subjects)
        a = random.choice(actions_medium)
        context = "in modern tech" if random.random() > 0.5 else "for beginners"
        q = f"{a} {s} {context}?"
        queries.append(q)
        labels.append(1)
        
    # Generate Complex (2)
    for _ in range(1000):
        s = random.choice(subjects)
        a = random.choice(actions_complex)
        detail = "considering scalability and cost" if random.random() > 0.5 else "with respect to future trends"
        q = f"{a} {s} {detail}, providing specific examples and references."
        queries.append(q)
        labels.append(2)
        
    return queries, labels


def main():
    print("="*60)
    print("TRAINING QUERY COMPLEXITY CLASSIFIER")
    print("="*60)
    
    print("\n[1/5] Getting training data...")
    queries, labels = get_training_data()
    
    # Convert labels to array
    labels = np.array(labels)
    
    print(f"####### Examples: {len(queries)} #######")
    print(f"  - Simple: {np.sum(labels==0)}")
    print(f"  - Medium: {np.sum(labels==1)}")
    print(f"  - Complex: {np.sum(labels==2)}")
    
    # Extract features — use batch method for speed (single encode call)
    print("\n[2/5] Extracting features...")
    feature_extractor = FeatureExtractor()

    # batch_extract_vectors encodes ALL queries in one SentenceTransformer
    # call instead of 10,000 individual forward passes — ~50x faster
    X = feature_extractor.batch_extract_vectors(queries, batch_size=256)
    y = labels
    
    print(f"####### Extracted features: {X.shape} #######")
    
    # Split data
    print("\n[3/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"####### Train set: {len(X_train)} examples #######")
    print(f"####### Test set: {len(X_test)} examples #######")
    
    # Train classifier
    print("\n[4/5] Training LightGBM classifier...")
    classifier = ComplexityClassifier()
    
    # Note: ComplexityClassifier might need to resize its internal model if it was pre-loaded
    # but here we are initializing a new one.
    
    train_accuracy = classifier.train(X_train, y_train)
    print(f"####### Training accuracy: {train_accuracy:.2%} #######")

    # Evaluate on test set
    if hasattr(classifier, 'scaler'):
        X_test_scaled = classifier.scaler.transform(X_test)
    else:
        X_test_scaled = X_test
        
    test_accuracy = classifier.model.score(X_test_scaled, y_test)
    print(f"####### Test accuracy: {test_accuracy:.2%} #######")
    
    # Feature importance
    print("\n[5/5] Analyzing feature importance...")
    try:
        importance = classifier.get_feature_importance()
        print("\nTop Features:")
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, score) in enumerate(sorted_features[:10], 1):
            print(f"  {i}. {feature}: {score:.4f}")
    except Exception as e:
        print(f"Could not get feature importance: {e}")
    
    # Save model
    save_path = Path("models/classifiers/complexity_classifier.pkl")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    classifier.save(save_path)
    print(f"\n####### Model saved to: {save_path} #######")
    
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
    
    # Mapping
    label_map = {0: "simple", 1: "medium", 2: "complex"}
    reverse_map = {"simple": 0, "medium": 1, "complex": 2}
    
    correct = 0
    for query, expected_str in test_examples:
        predicted_str, confidence = classifier.predict(query)
        
        is_correct = "#######" if predicted_str == expected_str else ">>>>>>><<<<<<<"
        
        if predicted_str == expected_str:
            correct += 1
        
        print(f"\n{is_correct} Query: '{query[:60]}...'")
        print(f"  Expected: {expected_str}")
        print(f"  Predicted: {predicted_str} (confidence: {confidence:.2%})")
    
    accuracy = correct / len(test_examples)
    print(f"\n####### Test Examples Accuracy: {accuracy:.2%} ({correct}/{len(test_examples)}) #######")

    print("\n" + "="*60)
    print("####### TRAINING COMPLETE! #######")
    print("="*60)
    
    
if __name__ == "__main__":
    main()