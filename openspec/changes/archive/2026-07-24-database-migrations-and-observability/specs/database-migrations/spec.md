## ADDED Requirements

### Requirement: Alembic Migrations
The database schema SHALL be managed exclusively via Alembic migrations.

#### Scenario: First application deployment
- **WHEN** the backend is deployed or started for the first time
- **THEN** Alembic migrations are executed against the Postgres database to create the required tables (e.g., query logs, memory)
