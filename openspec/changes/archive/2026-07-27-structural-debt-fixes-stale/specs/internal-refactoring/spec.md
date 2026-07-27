## ADDED Requirements

### Requirement: Standardized OpenTelemetry Configuration
The system SHALL use standard OTEL environment variables for configuring the tracing exporter.

#### Scenario: Configuring HTTP exporter
- **WHEN** `OTEL_EXPORTER_OTLP_PROTOCOL` is set to `http/protobuf`
- **THEN** the system uses the HTTP exporter

### Requirement: Decoupled BM25 Index
The system SHALL NOT manage local BM25 indexes from within the cloud `VectorStore` class.

#### Scenario: Initializing retrieval
- **WHEN** the `DocumentRetriever` is initialized
- **THEN** it manages the BM25 local JSON index internally
