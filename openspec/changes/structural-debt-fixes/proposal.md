## Why

Following a recent codebase audit, several areas of structural debt were identified that threaten maintainability and future scalability. These include duplicated logic in feature extraction, over-engineered OTEL configuration, mixed responsibilities in the vector store, and hardcoded data in the routing layer. Addressing these now will streamline the architecture and prevent subtle production bugs.

## What Changes

- Deduplicate semantic similarity math in `src/routing/features.py` by making `extract()` call `batch_extract_vectors()`.
- Refactor OpenTelemetry configuration in `src/utils/tracing.py` to use standard environment variables (`OTEL_EXPORTER_OTLP_PROTOCOL`) instead of brittle substring matching on endpoint URLs.
- Decouple local BM25 keyword search index management from the cloud Qdrant `VectorStore` class and move it to the `DocumentRetriever`.
- Extract 60+ hardcoded routing reference queries from `src/routing/features.py` into a dedicated configuration file (e.g. `config/routing.yaml`).

## Capabilities

### New Capabilities

*(None - pure refactoring)*

### Modified Capabilities

*(None - pure refactoring, no spec-level behavioral changes)*

## Impact

- **Routing Layer**: Minor internal refactoring to simplify feature extraction and externalize configuration.
- **Tracing**: OTEL configuration will adhere to industry standards; operators may need to explicitly pass `OTEL_EXPORTER_OTLP_PROTOCOL` if they relied on the undocumented substring matching.
- **Retrieval**: Cleaner separation between cloud vector operations and local disk-based BM25 operations.
