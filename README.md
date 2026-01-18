# 🚀 SmartRoute-AI

**Cost-Optimized LLM Routing System with RAG**

> Intelligently routes queries to the most cost-effective model while maintaining quality. Uses Groq's **free API** for 100% cost savings.
---

## 🎯 Features

- **Smart Routing** - ML classifier routes queries to optimal models based on complexity
- **Tiered Models** - Simple → 8B, Medium → 17B, Complex → 70B
- **RAG Integration** - Hybrid search with ChromaDB + BM25
- **Cost Tracking** - Real-time monitoring with budget alerts
- **Dashboard** - Streamlit analytics UI

---

## 🏗️ Architecture

```mermaid
flowchart TB
    subgraph Input
        Q[📝 User Query]
    end
    
    subgraph Routing["🧠 Smart Routing"]
        FE[Feature Extractor<br/>10 features]
        CL[LightGBM Classifier]
        FE --> CL
    end
    
    subgraph Models["⚡ Groq Models"]
        M1[Llama 3.1 8B<br/>Simple queries]
        M2[Llama 4 Scout 17B<br/>Medium queries]
        M3[Llama 3.3 70B<br/>Complex queries]
    end
    
    subgraph RAG["📚 RAG Pipeline"]
        VS[(ChromaDB)]
        BM[BM25 Index]
        RR[Reciprocal Rank Fusion]
        VS --> RR
        BM --> RR
    end
    
    subgraph Tracking["💰 Cost Management"]
        CT[Cost Tracker<br/>SQLite]
        BM2[Budget Manager]
        CT --> BM2
    end
    
    Q --> FE
    CL -->|simple| M1
    CL -->|medium| M2
    CL -->|complex| M3
    Q -.->|if RAG enabled| RAG
    RR -.-> M1 & M2 & M3
    M1 & M2 & M3 --> CT
    CT --> R[📤 Response]
```

---

## 📁 Project Structure

```
SmartRoute-AI/
├── api/
│   └── main.py              # FastAPI endpoints
├── app.py                   # Streamlit dashboard
├── config/
│   ├── models.yaml          # Model configurations
│   └── routing.yaml         # Routing strategies & budgets
├── src/
│   ├── routing/
│   │   ├── router.py        # Query router
│   │   ├── classifier.py    # LightGBM complexity classifier
│   │   └── features.py      # Feature extraction (10 features)
│   ├── models/
│   │   ├── model_manager.py # Model loading & caching
│   │   └── groq_model.py    # Groq API wrapper
│   ├── retrieval/
│   │   ├── retriever.py     # Hybrid retrieval (dense + sparse)
│   │   ├── vector_store.py  # ChromaDB wrapper
│   │   ├── embedder.py      # HuggingFace embeddings
│   │   ├── indexer.py       # Document indexing
│   │   └── chunking.py      # Text chunking
│   ├── pipeline/
│   │   └── inference.py     # Main orchestration pipeline
│   ├── cost/
│   │   ├── tracker.py       # Cost logging (SQLite)
│   │   └── budget.py        # Budget management
│   └── utils/
│       └── logger.py        # Logging configuration
├── scripts/
│   └── train_classifier.py  # Train complexity classifier
├── tests/                   # Pytest tests
├── data/
│   ├── documents/           # RAG documents
│   └── embeddings/          # ChromaDB persistence
└── models/
    └── classifiers/         # Trained classifier
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/SmartRoute-AI.git
cd SmartRoute-AI

# Create environment
conda create -n SmartRoute-AI python=3.10 -y
conda activate SmartRoute-AI

# Install dependencies
pip install poetry
poetry install
```

### 2. Get Groq API Key (FREE)

1. Go to [console.groq.com/keys](https://console.groq.com/keys)
2. Create a free account and generate API key
3. Setup environment:

```bash
cp .env.example .env
# Add your key: GROQ_API_KEY=gsk_xxxxx
```

### 3. Train Classifier

```bash
python scripts/train_classifier.py
```

### 4. Run

**Streamlit Dashboard:**
```bash
streamlit run app.py
# Open http://localhost:8501
```

**FastAPI Server:**
```bash
uvicorn api.main:app --reload
# Open http://localhost:8000/docs
```

---

## 🔄 How It Works

```mermaid
sequenceDiagram
    participant U as User
    participant R as Router
    participant C as Classifier
    participant M as Model
    participant T as Tracker

    U->>R: Send Query
    R->>C: Extract Features
    C->>R: Complexity (simple/medium/complex)
    R->>M: Route to appropriate model
    M->>R: Generate response
    R->>T: Log cost & usage
    T->>U: Return response + metadata
```

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/query` | POST | Process query (requires API key) |
| `/stats` | GET | Get usage statistics |
| `/models` | GET | List available models |

### Example Request

```bash
curl -X POST "http://localhost:8000/query" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "strategy": "cost_optimized"}'
```

---

## ⚙️ Configuration

### Routing Strategies

| Strategy | Description |
|----------|-------------|
| `cost_optimized` | Minimize cost, use smallest suitable model |
| `quality_first` | Maximize quality, use larger models |
| `balanced` | Balance between cost and quality |

### Model Tiers

<<<<<<< HEAD
| Complexity | Model | Speed |
|------------|-------|-------|
| Simple | Llama 3.1 8B | ~560 tok/sec |
| Medium | Llama 4 Scout 17B | ~400 tok/sec |
| Complex | Llama 3.3 70B | ~280 tok/sec |

---
=======
```yaml
# config/models.yaml
groq_models:
  llama_3_1_8b:      # Tier 1: Simple queries
  llama-4-scout-17b  # Tier 2: Medium queries  
  llama_3_3_70b:     # Tier 3: Complex queries
```

## 📈 app

Run the Streamlit dashboard for analytics:

```bash
streamlit run app.py
```

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 🐳 Docker

<<<<<<< HEAD
```bash
docker-compose up --build
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE)
