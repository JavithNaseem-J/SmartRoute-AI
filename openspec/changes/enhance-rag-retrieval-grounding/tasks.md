## 1. System Prompt & Grounding

- [x] 1.1 Update `InferencePipeline._build_messages` in `src/pipeline/inference.py` to enforce strict document-first QA grounding when context is non-empty.

## 2. Deterministic Reranker & Chunking

- [x] 2.1 Add local keyword cross-scoring fallback in `DocumentReranker.rerank` (`src/retrieval/reranker.py`) when HuggingFace API fails or token is missing.
- [x] 2.2 Update `DocumentChunker` separators in `src/retrieval/chunking.py` to include markdown header tokens (`#`, `##`, `###`).

## 3. Verification & Testing

- [x] 3.1 Write test in `tests/test_retriever.py` verifying short single-word keyword queries ("fill", "normalize") retrieve relevant document chunks.
- [x] 3.2 Run test suite (`pytest`) to confirm zero regressions across all pipeline tests.
