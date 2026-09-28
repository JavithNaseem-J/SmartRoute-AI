## ADDED Requirements

### Requirement: Cleanup removes unused code without changing active workflows
The cleanup SHALL remove only paths with confirmed callers and SHALL reduce net source lines, including new change documentation.

#### Scenario: Document selected in composer
- **WHEN** a PDF, TXT, or Markdown file is selected or dropped with RAG enabled
- **THEN** it is uploaded to the knowledge base without creating an unsent chat attachment

#### Scenario: Legacy source labels
- **WHEN** an answer has no structured citations
- **THEN** the UI does not fabricate citation buttons from old source strings

#### Scenario: One key for either active LLM provider
- **WHEN** GroqCloud or OpenRouter is selected
- **THEN** only `LLM_API_KEY` authenticates the selected provider

#### Scenario: Local reranking only
- **WHEN** document retrieval reranks candidates
- **THEN** it uses the local scorer or preserves Qdrant order, without an external reranker API key
