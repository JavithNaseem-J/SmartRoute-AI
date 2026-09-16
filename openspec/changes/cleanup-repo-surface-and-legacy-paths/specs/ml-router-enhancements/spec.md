## ADDED Requirements

### Requirement: Centroid artifact ownership is deterministic
The ML routing system SHALL define exactly how `data/models/reference_centroids.npy` is produced and made available to production.

#### Scenario: Centroid feature is enabled
- **WHEN** semantic centroid routing is intended to be active
- **THEN** the Docker/runtime environment includes a valid `reference_centroids.npy` artifact before `FeatureExtractor` initializes

#### Scenario: Centroid artifact is absent by design
- **WHEN** the centroid artifact is intentionally not shipped
- **THEN** the semantic centroid feature is removed or documented as disabled rather than silently appearing active

### Requirement: Classifier training is deterministic in deployment builds
The Docker build SHALL NOT introduce non-deterministic classifier behavior by retraining from random synthetic data without a fixed source of truth.

#### Scenario: Docker image build
- **WHEN** the production Docker image is built
- **THEN** the classifier artifact used at runtime is either copied from a tracked deterministic artifact or trained from deterministic inputs with fixed seeds
