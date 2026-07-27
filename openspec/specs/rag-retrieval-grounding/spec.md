# RAG Retrieval Grounding

## Requirements

### Requirement: Strict Document Grounding Prompt
The system SHALL format inference pipeline messages with strict document-first anti-hallucination instructions whenever retrieved context is present.

#### Scenario: Document query with context
- **WHEN** user query is processed with non-empty retrieved context
- **THEN** system prompt directs the LLM to answer strictly from the context, prioritizing section definitions over general knowledge

### Requirement: Local Keyword-Boosted Reranking Fallback
The system SHALL score and sort candidate document chunks using keyword and term overlap when external reranking API calls fail or HF token is unconfigured.

#### Scenario: HF Reranker API error or missing token
- **WHEN** external reranker API fails or returns non-200 status
- **THEN** system re-ranks documents locally using keyword score matching and returns the top-k chunks

### Requirement: Header-Aware Document Chunking
The system SHALL chunk documents using header-preserving separators so markdown headers and section titles remain attached to their body paragraphs.

#### Scenario: Indexing markdown and text files with section headers
- **WHEN** indexing documents containing section headers like `#`, `##`, or `###`
- **THEN** chunks retain headers attached to section content rather than orphaned lines
