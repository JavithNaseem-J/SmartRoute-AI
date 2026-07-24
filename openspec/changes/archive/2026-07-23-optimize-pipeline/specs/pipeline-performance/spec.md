## ADDED Requirements

### Requirement: Thread Pool Reusability
The document retrieval logic SHALL reuse a single thread pool instead of creating and destroying one per query.

#### Scenario: Hybrid Search Execution
- **WHEN** `_retrieve_hybrid` is invoked concurrently for multiple queries
- **THEN** the system uses a shared module-level `ThreadPoolExecutor` to execute the BM25 and dense searches

### Requirement: Non-Blocking Database Writes
Database query logging SHALL NOT block the main asynchronous event loop.

#### Scenario: Query Logging
- **WHEN** a query successfully completes and `CostTracker.log_query` is called
- **THEN** the SQLAlchemy database commit happens in a separate thread (e.g., via `asyncio.to_thread` or an executor) without halting the `asyncio` event loop

### Requirement: Modern Event Loop Retrieval
The application SHALL use modern non-deprecated methods for obtaining the current event loop.

#### Scenario: Event Loop Access
- **WHEN** `asyncio` methods require the current loop
- **THEN** `asyncio.get_running_loop()` is used instead of the deprecated `asyncio.get_event_loop()`
