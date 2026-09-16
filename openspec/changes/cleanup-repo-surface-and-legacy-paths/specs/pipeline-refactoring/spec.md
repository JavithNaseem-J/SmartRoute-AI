## ADDED Requirements

### Requirement: API routes are separated by responsibility
The FastAPI application SHALL separate system, auth, query, analytics, document, and static-asset responsibilities into focused modules or routers once legacy cleanup reduces route complexity.

#### Scenario: Backend route navigation
- **WHEN** a developer needs to change document upload or deletion behavior
- **THEN** they can work in a document-focused API module instead of editing unrelated system, analytics, and static-serving code

### Requirement: Frontend stateful flows are separated from shell rendering
The React app SHALL separate chat/session, document-management, streaming-query, and analytics state from high-level layout rendering.

#### Scenario: Frontend chat maintenance
- **WHEN** a developer changes streaming query behavior
- **THEN** they can work in a focused hook or service without editing sidebar, knowledge-base, and analytics layout markup

### Requirement: Prompt input primitives are not bundled with application-specific RAG behavior
The prompt input UI SHALL separate reusable UI primitives from SmartRoute-specific controls such as RAG upload, routing strategy selection, and document upload behavior.

#### Scenario: Prompt UI reuse
- **WHEN** a developer changes strategy selection or RAG upload behavior
- **THEN** tooltip, dialog, textarea, and button primitives do not need unrelated edits
