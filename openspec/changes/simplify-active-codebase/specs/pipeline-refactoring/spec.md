## ADDED Requirements

### Requirement: Removing unused inference methods preserves active contracts
The pipeline SHALL retain complete and streaming inference, tenant-scoped retrieval, budget enforcement, and citation validation while unused forwarding methods are removed.

#### Scenario: Streamed RAG answer
- **WHEN** a user queries an active uploaded document
- **THEN** the answer and terminal event include only valid structured citations
