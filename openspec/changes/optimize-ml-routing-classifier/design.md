## Context

The current LightGBM query complexity classifier suffers from false-positive "complex" classifications on long simple queries. This is driven by two technical constraints:
1. Feature extraction in `src/routing/features.py` attempted live HTTP embedding calls against Hugging Face. Unauthenticated calls returned HTTP 401, setting all semantic features (`simple_similarity`, `complex_similarity`) to `0.0`.
2. Synthetic datasets relied on sentence length as a proxy for complexity, training LightGBM to use text length as a strong decision tree split rule.

## Goals / Non-Goals

**Goals:**
- Eliminate live HTTP API calls during feature extraction by using offline pre-computed cluster centroids (`data/models/reference_centroids.npy`).
- Re-balance training datasets (`data/training/synthetic_queries.csv`) with length-decoupled hard negatives (long simple queries and short complex queries).
- Expand feature engineering with Flesch-Kincaid grade level, code token density, and relative centroid distance margins.
- Add cost-biased probability thresholding in `src/routing/router.py` to route low-confidence complex predictions (`confidence < 0.75`) to cheap models.

**Non-Goals:**
- Replacing LightGBM with heavy deep learning models (e.g. BERT/LLM classifiers) that introduce latency or RAM overhead.

## Decisions

### 1. Offline Pre-computed Centroid Matrix over Live API Calls
- **Decision**: Generate `data/models/reference_centroids.npy` offline. At runtime, load numpy matrices in `FeatureExtractor.__init__` and perform vector dot products in memory (<1ms).
- **Alternative Considered**: Calling HuggingFace Endpoint API asynchronously during feature extraction. Rejected due to network latency (150ms+), rate limits, and 401 failure modes.

### 2. Length-Decoupled Feature Engineering
- **Decision**: Add `flesch_kincaid_grade`, `code_syntax_density`, and `centroid_margin` (`max(complex_sim) - max(simple_sim)`) to `FEATURE_ORDER`.
- **Alternative Considered**: Training solely on text embeddings. Rejected because tree-based classifiers benefit heavily from explicit syntactic and readability signals.

### 3. Cost-Biased Hysteresis Fallback
- **Decision**: If `complexity == "complex"` and `confidence < 0.75`, fallback to the cheap model (`nvidia/nemotron-nano-9b-v2:free` or `openrouter/free`).
- **Rationale**: In enterprise LLM gateways, false-positive complex queries cost 50x more, whereas false-negative complex queries gracefully handled by cheap models carry negligible risk.

## Risks / Trade-offs

- **[Risk] Centroid Drift** → *Mitigation*: Provide `scripts/generate_centroids.py` to regenerate `reference_centroids.npy` whenever reference queries in `routing.yaml` are modified.
- **[Risk] Code Density False Positives** → *Mitigation*: Normalize code token counts by query length so short SQL snippets are properly weighted.
