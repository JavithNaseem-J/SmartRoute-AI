# Database Migrations

## Purpose
Ensure database migrations are decoupled from web container startup to prevent scaling race conditions.

## Requirements

### Requirement: Decoupled Database Migrations
The system SHALL NOT execute database schema migrations automatically during the API web server boot phase.

#### Scenario: Safe horizontal scaling
- **WHEN** the deployment orchestrator provisions multiple container instances simultaneously
- **THEN** the database migrations SHALL run strictly once as a pre-deployment step, preventing lock contention or race conditions among the web container instances.

### Requirement: Alembic Migrations
The database schema SHALL be managed exclusively via Alembic migrations.

#### Scenario: First application deployment
- **WHEN** the backend is deployed or started for the first time
- **THEN** Alembic migrations are executed against the Postgres database to create the required tables (e.g., query logs, memory)
