## ADDED Requirements

### Requirement: Enhanced Semantic Anchoring
The ML Feature Extractor SHALL evaluate query embeddings against expanded sets of at least 20 reference queries for each complexity level (simple, medium, complex).

#### Scenario: Medium Complexity Query Evaluation
- **WHEN** a user submits a query requiring multi-step aggregation but no deep logical proofs
- **THEN** the system SHALL calculate a `medium_similarity` feature score that heavily biases the router towards the medium complexity model.

### Requirement: Advanced Linguistic Features
The ML Feature Extractor SHALL compute advanced heuristics including logic operator density and symbol density.

#### Scenario: Code-heavy query parsing
- **WHEN** a user submits a short query with dense coding syntax (e.g., brackets, arrows)
- **THEN** the system SHALL accurately flag high symbol density, guiding the LightGBM classifier to flag it as complex despite its short length.

### Requirement: Reproducible Model Retraining
The system SHALL include automated scripts to generate diverse synthetic query datasets via an LLM and retrain the LightGBM classifier end-to-end.

#### Scenario: Updating the routing boundaries
- **WHEN** the system administrators want to update routing logic
- **THEN** they SHALL be able to run `scripts/generate_training_data.py` followed by `scripts/train_classifier.py` to transparently re-tune the decision trees.
