## ADDED Requirements

### Requirement: Accurate Cost Key Validation
The budget checking logic SHALL reference the correct configuration key (`nvidia_models`) when estimating cost for default NVIDIA models.

#### Scenario: Estimating NVIDIA query cost
- **WHEN** `BudgetManager.estimate_query_cost` is called for an NVIDIA model
- **THEN** it retrieves the cost correctly from the `nvidia_models` dictionary instead of returning a hardcoded default

### Requirement: Valid Fallback Model Routing
The budget exhaustion logic SHALL route queries to a fallback model that actually exists in the configuration.

#### Scenario: Daily Budget Exceeded
- **WHEN** the daily budget limit is exceeded
- **THEN** the router modifies the routing decision to use `nemotron_nano_8b` (or the default strategy's simple model) rather than a non-existent `llama_3_1_8b`
