## Why

Short queries (e.g., "what is fill?", "what is normalize?", "specs") on uploaded RAG documents frequently fall back to general LLM parametric memory instead of retrieving relevant text chunks. This occurs because dense embeddings prioritize paragraph semantic density over short keyword terms, external reranking APIs fail or time out silently, and system prompts lack strict document grounding directives.

## What Changes

- **Strict Document Grounding System Prompt**: Enhance `InferencePipeline._build_messages` to enforce strict document-first QA rules whenever retrieved context is present.
- **BM25 + Dense Hybrid Retrieval**: Implement BM25 keyword retrieval fused with Qdrant dense vector search using Reciprocal Rank Fusion (RRF) to ensure exact keyword and single-word matches are retrieved.
- **Deterministic Local Reranking & Fallback**: Replace external HuggingFace HTTP reranking API calls with a local BM25/keyword cross-scoring fallback to prevent silent retrieval failures.
- **Section & Header-Aware Text Chunking**: Update text splitting separators to preserve markdown headers (`#`, `##`, `###`) and section boundaries in chunk payloads.

## Capabilities

### New Capabilities
- `rag-retrieval-grounding`: Hybrid BM25+Dense retrieval, deterministic local reranking, header-aware document chunking, and strict anti-hallucination document QA grounding.

### Modified Capabilities
<!-- None -->

## Impact

- `src/retrieval/retriever.py`: Updated hybrid retrieval with BM25 + Qdrant fallback scoring.
- `src/retrieval/reranker.py`: Local deterministic keyword scoring fallback added when HF API fails or token is absent.
- `src/retrieval/chunking.py`: Updated RecursiveCharacterTextSplitter separators for header retention.
- `src/pipeline/inference.py`: Updated system prompt for strict anti-hallucination document grounding.
- `tests/test_retriever.py`: Added test cases for short single-term queries and exact header retrieval.
