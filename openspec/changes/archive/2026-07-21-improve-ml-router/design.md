## Context

The current `FeatureExtractor` provides basic lexical features (`word_count`, `has_code`) and matches against only 10 total semantic anchor queries. The LightGBM classifier predicts three complexity levels (`simple`, `medium`, `complex`), but lacks a "medium" anchor list, and its training data is undocumented/unreproducible.

## Goals / Non-Goals

**Goals:**
- Improve routing accuracy without modifying the core LightGBM/hybrid architecture.
- Expand reference queries for robust semantic matching.
- Add advanced linguistic features (readability, logic operators, symbol density).
- Provide a reproducible synthetic data generation pipeline to retrain the classifier.

**Non-Goals:**
- Introducing external API calls for routing (e.g., Cross-Encoders).
- Changing the existing model definitions or provider logic.
- Post-retrieval routing (context-aware routing).

## Decisions

- **Medium Similarity Feature:** Add a list of 25 medium complexity reference queries to calculate `medium_similarity` to help the classifier separate the middle ground.
- **Readability & Logic Metrics:** Calculate Flesch-Kincaid proxy and logic keywords density to measure intent complexity beyond just word counts.
- **Synthetic Training Data Generation Script:** Use the `Llama-3-8B` API to synthetically generate 1,000 queries per complexity level. This ensures training diversity and reproducibility without manually hand-curating 3,000 prompts.

## Risks / Trade-offs

- **Risk:** Increasing reference queries to 75 (25x3) adds minor latency to the `SentenceTransformer` cosine similarity calculation.
- **Mitigation:** Vectorizing the cosine similarity calculation inside numpy and leveraging the small `all-MiniLM-L6-v2` embedding dimensions (384) ensures latency stays <2ms.
- **Risk:** Flesch-Kincaid or complex linguistic features may be slow to compute in python.
- **Mitigation:** Use fast heuristic regexes instead of heavy NLP libraries like Spacy.
