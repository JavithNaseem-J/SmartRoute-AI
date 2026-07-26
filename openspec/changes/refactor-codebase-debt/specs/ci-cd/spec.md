## MODIFIED Requirements

### Requirement: Docker Build Validation
The CI pipeline SHALL validate that the Docker images can be built successfully using valid job dependency names.

#### Scenario: PR contains Dockerfile changes
- **WHEN** a pull request is opened or updated
- **THEN** GitHub Actions runs the `docker-validate` job and failure alerts accurately depend on valid job names
