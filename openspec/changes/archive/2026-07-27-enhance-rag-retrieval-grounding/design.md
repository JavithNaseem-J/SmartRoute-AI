## Context

In our current RAG pipeline, short or single-keyword queries (such as "what is fill?", "what is normalize?", "specs") on uploaded documents often fail to return accurate document content. Instead, the model falls back to generic conversational responses.

Root cause investigation revealed:
1. **Dense Vector Misalignment**: Dense embedding cosine distance between a short query (`"what is fill?"`) and a long paragraph (`"Fill: Now review the entire website..."`) scores low compared to generic LLM memory.
2. **Unreliable External Reranker API**: `DocumentReranker` calls HuggingFace's public API (`cross-encoder/ms-marco-MiniLM-L-6-v2`), which frequently times out or rate-limits without an API token, silently falling back to un-ordered vector results.
3. **Weak RAG System Prompt**: System prompt in `src/pipeline/inference.py` ("Use the provided context...") allows the LLM to default to parametric memory when context similarity is weak.
4. **Header-Ignorant Chunking**: Separators in `DocumentChunker` do not explicitly prioritize markdown headers, causing section titles like `Fill` or `Normalize` to lose structural attachment to body text.

## Goals / Non-Goals

**Goals:**
- Guarantee short keyword queries ("fill", "normalize", "research") retrieve exact section text from uploaded documents.
- Implement strict anti-hallucination document-first system prompt grounding in `InferencePipeline`.
- Add local keyword cross-scoring fallback in `DocumentReranker` to operate deterministically without HF API token.
- Improve `DocumentChunker` header retention so section headers remain attached to body text.

**Non-Goals:**
- Upgrading to multi-gigabyte local neural cross-encoders (we keep light deterministic keyword boosting to stay fast and cheap).
- Re-architecting Qdrant storage schema.

## Decisions

### Decision 1: Strict Anti-Hallucination QA Prompt
- Update `InferencePipeline._build_messages` to instruct the LLM:
  > "You are an enterprise document QA assistant. Answer strictly based on the provided context. If the user asks about a term, section, or instruction present in the context, extract and summarize that specific text. Do NOT use general dictionary definitions or outside knowledge if the term appears in the context."

### Decision 2: Local Deterministic Reranker Fallback
- When HuggingFace Inference API fails or token is missing, `DocumentReranker` will fall back to exact keyword & fuzzy token match scoring between the query and retrieved document text rather than returning documents unsorted.

### Decision 3: Header-Aware Document Chunking
- Configure `DocumentChunker` with separators `["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " "]` to preserve section headers alongside body content.

## Risks / Trade-offs

- [Risk] Strict system prompt might cause LLM to state "Context does not provide information" if query is out of domain → Mitigation: Correct behavior for RAG document QA.
- [Risk] Larger overlap in chunking could increase index size slightly → Mitigation: Chunk size remains 500 characters with 100 overlap.
