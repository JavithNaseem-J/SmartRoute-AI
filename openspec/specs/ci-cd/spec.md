# CI/CD

## Purpose
TBD

## Requirements

### Requirement: Automated Pull Request Testing
The repository SHALL execute a CI pipeline on every Pull Request to the main branch to validate the codebase.

#### Scenario: PR is opened or updated
- **WHEN** a pull request is opened or updated
- **THEN** GitHub Actions runs linting (flake8/black) and all tests using `pytest`
- **THEN** the pipeline must pass before the PR can be merged

### Requirement: Docker Build Validation
The CI pipeline SHALL validate that the Docker images can be built successfully.

#### Scenario: PR contains Dockerfile changes
- **WHEN** a pull request is opened
- **THEN** GitHub Actions runs `docker-compose build` to ensure the environment builds without errors
