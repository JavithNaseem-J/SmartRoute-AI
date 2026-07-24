# Semantic Caching

## Purpose
Intercept and cache LLM responses based on semantic similarity to minimize duplicate API calls and lower inference latency.

## Requirements

### Requirement: Semantic Caching of Queries
The system SHALL intercept incoming queries and retrieve previously generated LLM responses if a semantically similar query was processed recently.

#### Scenario: Exact match or high similarity query
- **WHEN** an incoming query has a cosine similarity > 0.95 with a cached query in Qdrant
- **THEN** the system SHALL return the cached response payload from Redis without calling the upstream LLM API
- **AND** the latency SHALL be significantly lower than a full LLM inference roundtrip.

#### Scenario: Low similarity query
- **WHEN** an incoming query does not match any cached query above the similarity threshold
- **THEN** the system SHALL process the query normally through the LLM pipeline and cache the resulting response asynchronously.
