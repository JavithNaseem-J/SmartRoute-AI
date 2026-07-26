## Why

In enterprise production, misclassifying simple queries as complex routes low-complexity traffic to flagship models (e.g. Claude 3.5 / GPT-4o), unnecessarily driving up OpenRouter API costs by 10x–50x. 

The current LightGBM routing classifier experiences misclassifications on long simple queries (e.g. polite multi-word questions) because feature extraction relied on live HTTP calls for reference embeddings (which fell back to 0.0 vectors when unauthenticated), forcing the model to rely on naive text length shortcuts.

This change optimizes the ML routing classifier to achieve >98% precision on simple queries, 0ms offline vector centroid feature extraction, length-decoupled training dataset curation, and cost-biased probability calibration.

## What Changes

- **Offline Vector Centroids**: Store pre-computed reference cluster centroids in binary `.npy` format (`data/models/reference_centroids.npy`), eliminating live HTTP network dependencies during feature extraction.
- **Enhanced Feature Set**: Add length-decoupled features including readability grade (Flesch-Kincaid), code syntax density, and relative centroid distance margins to decouple text length from complexity.
- **Dataset Curation with Hard Negatives**: Re-balance `synthetic_queries.csv` with length-decoupled hard negatives (long simple queries and short complex queries).
- **Probability Calibration & Cost-Biased Hysteresis**: Calibrate LightGBM prediction probabilities and route uncertain complex queries (`confidence < threshold`) to cheap models to prevent cost spikes.

## Capabilities

### New Capabilities
- None

### Modified Capabilities
- `ml-router-enhancements`: Update ML classifier feature extraction, offline centroid loading, and confidence hysteresis routing requirements.

## Impact

- **Affected Code**: `src/routing/features.py`, `src/routing/classifier.py`, `src/routing/router.py`, `scripts/train_classifier.py`, `data/training/synthetic_queries.csv`, `config/routing.yaml`.
- **Dependencies**: Added `data/models/reference_centroids.npy` binary artifact.
- **Performance**: Feature extraction latency reduced from ~150ms to <1ms with 0 external network dependencies.
