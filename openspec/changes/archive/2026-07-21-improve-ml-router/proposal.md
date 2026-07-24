## Why

The current hybrid LightGBM ML router performs well for cost and latency, but its decision boundaries are heavily reliant on superficial lexical features and a tiny set of semantic anchors (only 5 "simple" and 5 "complex" queries). To achieve FAANG-level routing accuracy without sacrificing the 5ms latency or adding third-party API dependencies (like Cross-Encoders), we must dramatically enhance the router's feature space and training data. 

## What Changes

- Expands the semantic anchor dataset in `reference_queries` to ~20-30 robust examples per category.
- Introduces a new `medium` semantic anchor list and a corresponding `medium_similarity` feature.
- Implements advanced linguistic features (e.g., Flesch-Kincaid readability, logic operator counts, symbol density) to better capture query intent.
- Introduces a synthetic data generation script to produce a large, diverse dataset of labeled queries using an LLM.
- Retrains the LightGBM classifier on the new, massive synthetic dataset to hyper-tune the decision boundaries.

## Capabilities

### New Capabilities
- `ml-router-enhancements`: Upgrades the deterministic ML routing heuristics and semantic anchors to achieve higher accuracy.

### Modified Capabilities

## Impact

- **Affected Code**: `src/routing/features.py`, `src/routing/classifier.py`
- **New Scripts**: `scripts/generate_training_data.py` (if not existing), `scripts/train_classifier.py`
- **Performance**: Negligible latency increase (<1ms) for new feature extraction; massive accuracy boost in routing decisions.
