## ADDED Requirements

### Requirement: Unified Pipeline Setup Context
The `InferencePipeline` SHALL extract semantic caching, routing, cost estimation, budget checking, and retrieval into a unified internal method that prepares a context object, rather than duplicating this logic across execution endpoints.

#### Scenario: Running inference endpoints
- **WHEN** the `run` or `astream_run` endpoints are invoked
- **THEN** they MUST both call the same internal setup method to prepare context before connecting to the selected model

### Requirement: Standardized Message Context
The system SHALL assemble conversational context and prompts into standard message arrays (`[{"role": "system", ...}, {"role": "user", ...}]`) BEFORE calling any `BaseLLM` implementation.

#### Scenario: Calling an LLM provider
- **WHEN** the `InferencePipeline` invokes `agenerate` or `astream` on a model
- **THEN** the model MUST receive pre-formatted message context rather than separate prompt, context, and history strings
