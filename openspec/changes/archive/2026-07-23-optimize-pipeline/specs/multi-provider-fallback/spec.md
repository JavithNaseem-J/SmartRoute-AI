## ADDED Requirements

### Requirement: Fallback on API Error
The system SHALL attempt to use a fallback LLM provider if the primary provider throws a rate limit or service unavailable error.

#### Scenario: Primary Provider Overloaded
- **WHEN** the primary NVIDIA NIM provider returns a `RateLimitError` or `APIError`
- **THEN** the `ModelManager` intercepts the exception and seamlessly reroutes the request to the configured fallback provider (e.g., Groq)

### Requirement: Multi-Provider Base Classes
The system SHALL support multiple provider implementations derived from `BaseLLM`.

#### Scenario: Groq Provider Configuration
- **WHEN** a model configured in `models.yaml` under `groq_models` is requested
- **THEN** a `GroqModel` instance is instantiated instead of `NvidiaModel`
