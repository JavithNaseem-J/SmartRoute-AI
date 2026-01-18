# =============================================================================
# RETRIEVAL CONSTANTS
# =============================================================================

# Chunking defaults
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Embedding model
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DEVICE = "cpu"

# Vector store
DEFAULT_COLLECTION_NAME = "smartroute_docs"
DEFAULT_PERSIST_DIR = "data/embeddings"

# Retrieval
DEFAULT_TOP_K = 5
DEFAULT_MAX_DISTANCE = 1.5  # L2 distance threshold (lower = more similar)
DEFAULT_BM25_WEIGHT = 0.4
DEFAULT_DENSE_WEIGHT = 0.6


# =============================================================================
# COST TRACKING CONSTANTS
# =============================================================================

# Baseline cost for savings calculation (GPT-4 equivalent)
BASELINE_COST_PER_QUERY = 0.15  # USD

# Database
DEFAULT_DB_PATH = "data/costs/usage.db"

# Query truncation
MAX_QUERY_LOG_LENGTH = 200


# =============================================================================
# API CONSTANTS
# =============================================================================

# Rate limiting
DEFAULT_QUERY_RATE_LIMIT = "30/minute"
DEFAULT_STATS_RATE_LIMIT = "60/minute"

# Query validation
MAX_QUERY_LENGTH = 500
MIN_QUERY_LENGTH = 1


# =============================================================================
# MODEL TIERS
# =============================================================================

MODEL_TIERS = {
    "simple": {
        "description": "Simple queries, greetings, basic facts",
        "max_complexity_score": 0.33
    },
    "medium": {
        "description": "Explanations, summaries, moderate reasoning",
        "max_complexity_score": 0.66
    },
    "complex": {
        "description": "Analysis, code generation, deep reasoning",
        "max_complexity_score": 1.0
    }
}


# =============================================================================
# ROUTING STRATEGIES
# =============================================================================

ROUTING_STRATEGIES = ["cost_optimized", "quality_first", "balanced"]
DEFAULT_ROUTING_STRATEGY = "cost_optimized"


# =============================================================================
# PATHS
# =============================================================================

DEFAULT_CONFIG_DIR = "config"
DEFAULT_MODELS_DIR = "models"
DEFAULT_CLASSIFIER_PATH = "models/classifiers/complexity_classifier.pkl"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_DATA_DIR = "data"
