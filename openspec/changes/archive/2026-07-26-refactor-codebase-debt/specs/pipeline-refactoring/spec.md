## ADDED Requirements

### Requirement: Consolidated Cost and Metric Logging
The `InferencePipeline` SHALL centralize query metric logging to prevent duplicate logging calls across streaming and non-streaming endpoints.

#### Scenario: Logging completed query execution
- **WHEN** a query completes execution in `run`, `astream_run`, or cache hit paths
- **THEN** metrics are logged to `CostTracker` using a unified helper method

### Requirement: Shared Keyword Definitions and Client Setup
The routing and retrieval subsystems SHALL avoid duplicating keywords and initialization logic.

#### Scenario: Extracting features and initializing sparse models
- **WHEN** `HeuristicClassifier` and `FeatureExtractor` evaluate query complexity
- **THEN** they use a single shared set of classification keywords
- **WHEN** Qdrant clients are initialized for indexing or retrieval
- **THEN** sparse vector models are configured centrally
