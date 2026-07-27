## ADDED Requirements

### Requirement: Technical Debt Resolution
The system SHALL eliminate duplicate structure, duplicate logic, and over-engineering from the `src/` codebase without changing the external API or routing logic.

#### Scenario: Codebase Maintainability
- **WHEN** developers review or modify the codebase
- **THEN** they encounter unified async database clients, a single shared embedding model, and clean OTEL configurations.
