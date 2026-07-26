# ML Router Enhancements Delta Spec

## ADDED Requirements

### Requirement: Offline Pre-computed Centroid Embeddings
The ML Feature Extractor SHALL load reference cluster centroids from a pre-computed binary numpy file (`data/models/reference_centroids.npy`) without performing live external HTTP API requests.

#### Scenario: Offline feature extraction
- **WHEN** a query is submitted to the Feature Extractor
- **THEN** the system SHALL compute cosine similarity against pre-loaded centroid vectors in memory in under 1ms without making external network calls.

### Requirement: Length-Decoupled Feature Extraction
The ML Feature Extractor SHALL compute readability index (Flesch-Kincaid), code token syntax density, and relative centroid distance margins to decouple string length from query complexity.

#### Scenario: Evaluating long simple queries
- **WHEN** a user submits a long query containing polite conversational text but simple factual intent
- **THEN** the system SHALL compute a high readability score and low centroid margin, preventing false-positive complex classifications.

### Requirement: Probability Calibration & Hysteresis Thresholding
The Query Router SHALL calibrate LightGBM output probabilities and apply a cost-biased hysteresis threshold to route low-confidence complex predictions (`confidence < 0.75`) to the cost-effective cheap model.

#### Scenario: Uncertain complexity classification
- **WHEN** the ML classifier predicts `complex` with a confidence score below 0.75
- **THEN** the system SHALL escalate/fallback to the cost-optimized cheap model to prevent unnecessary API expenditure.
