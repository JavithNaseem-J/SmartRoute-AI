## ADDED Requirements

### Requirement: Document Listing API
The system SHALL provide an endpoint `GET /v1/documents` that lists all currently stored document files with metadata (name, size, modification time).

#### Scenario: List indexed documents
- **WHEN** client sends a GET request to `/v1/documents`
- **THEN** system returns a JSON list of available document files and their metadata

### Requirement: Document Deletion API
The system SHALL provide an endpoint `DELETE /v1/documents/{filename}` that deletes the specified document from disk, purges its vector points from Qdrant, flushes semantic cache, and reloads the retriever.

#### Scenario: Delete specific document
- **WHEN** client sends a DELETE request to `/v1/documents/{filename}`
- **THEN** system removes the local file, purges Qdrant points with matching source metadata, invalidates cache, and returns success

### Requirement: Bulk Document Clear API
The system SHALL provide an endpoint `DELETE /v1/documents` that deletes all document files, resets the Qdrant collection, and flushes semantic cache.

#### Scenario: Clear all documents
- **WHEN** client sends a DELETE request to `/v1/documents`
- **THEN** system removes all document files, deletes the Qdrant collection, flushes Redis cache, and returns success
