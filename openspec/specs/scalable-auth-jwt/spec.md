# Scalable Auth JWT

## Purpose
Manage client authentication securely through granular, standard JWT bearer tokens using Supabase.

## Requirements

### Requirement: JWT-based Client Authentication
The system SHALL replace the static global API key authentication with granular JWT bearer token authentication validated against the Supabase Auth service.

#### Scenario: Valid JWT token provided
- **WHEN** a client provides a valid JWT in the `Authorization: Bearer <token>` header
- **THEN** the API SHALL decode the token, extract the user identifier, and allow access to the protected endpoints.

#### Scenario: Invalid or missing token
- **WHEN** a client provides an expired, invalid, or missing JWT
- **THEN** the API SHALL reject the request with a 401 Unauthorized HTTP response status code.
