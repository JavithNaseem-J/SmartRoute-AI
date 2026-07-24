## Context

The codebase audit revealed structural debt across the routing, retrieval, and tracing modules. While the current implementations function correctly, they represent anti-patterns (such as duplicate math logic, hardcoded configuration data, and blurred boundaries between remote/local concerns) that will impede future scaling and increase maintenance overhead. This design focuses on isolating responsibilities and extracting configuration.

## Goals / Non-Goals

**Goals:**
- Eliminate duplication in semantic feature extraction.
- Externalize static routing data (reference queries).
- Strictly decouple cloud vector db concerns from local disk BM25 concerns.
- Standardize the OpenTelemetry configuration logic.

**Non-Goals:**
- Changing the actual routing algorithm, embeddings model, or model thresholds.
- Adding new OpenTelemetry capabilities or migrating away from Langfuse.
- Re-architecting the vector storage system entirely; we are purely fixing the class boundary.

## Decisions

**1. Deduplicate Semantic Similarity (`src/routing/features.py`)**
- *Decision*: Route single query extraction through the batched extraction pipeline (`batch_extract_vectors([query])[0]`).
- *Rationale*: Maintains a single source of truth for similarity logic. The minimal overhead of wrapping a single query in a list is a worthwhile trade-off for eliminating math duplication.

**2. Extract Reference Queries (`config/routing.yaml`)**
- *Decision*: Move the `reference_queries` dictionary out of the `FeatureExtractor` class constructor into a YAML config file.
- *Rationale*: Allows non-engineers or automated tuning scripts to adjust reference queries without modifying the codebase. It aligns with how the main router config works.

**3. Move BM25 Logic to Retriever (`src/retrieval/vector_store.py` -> `retriever.py`)**
- *Decision*: Strip `save_bm25_index` and `load_bm25_documents` out of the `VectorStore` class and move them into `DocumentRetriever`.
- *Rationale*: The `VectorStore` acts as an adapter for Qdrant (a remote vector database). BM25 uses local JSON and a pure Python ranking function, which is orchestrational logic belonging in the retriever layer.

**4. Standardize OTEL Configuration (`src/utils/tracing.py`)**
- *Decision*: Use the standard `OTEL_EXPORTER_OTLP_PROTOCOL` environment variable instead of checking `is_langfuse = "langfuse" in endpoint`.
- *Rationale*: Substring matching on the domain name is brittle and violates 12-factor app principles. Standard OTEL variables are the accepted way to configure exporter types.

## Risks / Trade-offs

- **Risk: Breaking existing deployed configurations** → By requiring standard OTEL protocol variables, existing deployments that relied on the auto-detection might fail to send traces.
- **Mitigation:** Document this as a breaking change in deployment scripts (e.g., ensure `.env` templates are updated with `OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf`).
