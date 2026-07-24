## 1. Feature Engineering

- [x] 1.1 Expand `reference_queries` in `src/routing/features.py` to include at least 20 simple and 20 complex queries.
- [x] 1.2 Add a new `medium` key to `reference_queries` containing at least 20 medium complexity queries.
- [x] 1.3 Update `FEATURE_ORDER` array to include `medium_similarity`, `logic_operator_count`, and `symbol_density`.
- [x] 1.4 Implement the logic inside the `extract()` method to calculate cosine similarity for the new `medium` references.
- [x] 1.5 Implement logic inside `extract()` to calculate logic operator counts (e.g., "if", "then", "and", "or").
- [x] 1.6 Implement logic inside `extract()` to calculate symbol density (ratio of code symbols to text length).

## 2. Synthetic Data Pipeline

- [x] 2.1 Create `scripts/generate_training_data.py` to connect to Llama-3 to synthetically generate 1,000 queries per complexity level.
- [x] 2.2 Format the synthetic output to dump directly into a clean `data/training/synthetic_queries.csv` file.

## 3. Classifier Retraining

- [x] 3.1 Update or create `scripts/train_classifier.py` to read `synthetic_queries.csv`.
- [x] 3.2 Ensure the script scales features using `StandardScaler` and retrains the `LGBMClassifier`.
- [x] 3.3 Validate the newly trained model successfully saves to `models/classifiers/complexity_classifier.pkl`.
- [x] 3.4 Write unit tests in `tests/test_router.py` to ensure the upgraded `FeatureExtractor` does not crash during live inference.
