# CI/CD

## Purpose
Validate the backend, frontend, and single Render Docker image before deployment.

## Requirements

### Requirement: Automated Pull Request Testing
The repository SHALL execute a CI pipeline on every Pull Request to the main branch to validate the codebase.

#### Scenario: PR is opened or updated
- **WHEN** a pull request is opened or updated
- **THEN** GitHub Actions runs linting, type checking, frontend build, Docker validation, and all tests using `pytest`
- **THEN** the pipeline must pass before the PR can be merged

### Requirement: Docker Build Validation
The CI pipeline SHALL validate that the single Docker image can be built successfully using valid job dependency names.

#### Scenario: PR contains Dockerfile changes
- **WHEN** a pull request is opened or updated
- **THEN** GitHub Actions runs the `docker-validate` job and failure alerts accurately depend on valid job names
