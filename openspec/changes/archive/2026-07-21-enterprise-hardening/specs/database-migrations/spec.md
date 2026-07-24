## ADDED Requirements

### Requirement: Decoupled Database Migrations
The system SHALL NOT execute database schema migrations automatically during the API web server boot phase.

#### Scenario: Safe horizontal scaling
- **WHEN** the deployment orchestrator provisions multiple container instances simultaneously
- **THEN** the database migrations SHALL run strictly once as a pre-deployment step, preventing lock contention or race conditions among the web container instances.
