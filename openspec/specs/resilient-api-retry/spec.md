# Resilient API Retry

## Purpose
Ensure all external API integrations (like LLMs) fail gracefully and retry appropriately.

## Requirements

### Requirement: Resilient API Retries
The system SHALL wrap all external LLM and Embedder API calls with an exponential backoff and retry mechanism to gracefully handle transient network or provider failures.

#### Scenario: Transient API failure
- **WHEN** the external LLM provider returns a 429 Rate Limit or 5xx Server Error
- **THEN** the system SHALL retry the request up to 3 times with exponential backoff (e.g., 1s, 2s, 4s).

#### Scenario: Persistent API failure
- **WHEN** the external LLM provider fails consecutively past the maximum retry limit
- **THEN** the system SHALL abort the request gracefully and return an appropriate 503 Service Unavailable or fallback error to the client, preventing complete API crashes.
