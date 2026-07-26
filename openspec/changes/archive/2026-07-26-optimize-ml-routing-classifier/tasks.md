## 1. Offline Centroid Generation & Dataset Curation

- [x] 1.1 Create `scripts/generate_centroids.py` to pre-compute reference cluster centroids and export `data/models/reference_centroids.npy`
- [x] 1.2 Update `data/training/synthetic_queries.csv` with length-decoupled hard negatives (long simple queries and short complex queries)

## 2. Feature Extractor Optimization

- [x] 2.1 Update `src/routing/features.py` to load pre-computed `reference_centroids.npy` in `<1ms` without live HTTP calls
- [x] 2.2 Add `flesch_kincaid_grade`, `code_syntax_density`, and `centroid_margin` to `FEATURE_ORDER` in `src/routing/features.py`

## 3. Model Retraining & Router Hysteresis

- [x] 3.1 Re-run `scripts/train_classifier.py` to train LightGBM on updated feature set and dataset, updating `models/classifiers/complexity_classifier.pkl`
- [x] 3.2 Update `src/routing/router.py` to apply cost-biased hysteresis thresholding for uncertain complex queries (`confidence < 0.75`)

## 4. Verification & Testing

- [x] 4.1 Run unit tests (`pytest tests/`) to verify router accuracy and feature extraction
- [x] 4.2 Verify zero HTTP calls during feature extraction and test precision on long simple queries
