# 🚀 SmartRoute-AI

**Cost-Optimized LLM Routing System with RAG**

> Intelligently routes queries to the most cost-effective model while maintaining quality. Achieve **70%+ cost savings** by using smaller models for simple queries and powerful models only when needed.


---

## 🎯 Key Features

- **Smart Routing**: ML classifier routes queries to optimal models based on complexity
- **Tiered Model System**: Simple → 8B, Medium → 32B, Complex → 70B
- **RAG Integration**: Retrieval-Augmented Generation with ChromaDB
- **Cost Tracking**: Real-time cost monitoring and budget management
- **Multiple Providers**: Groq (free), OpenAI, Anthropic support
- **Dashboard**: Streamlit analytics dashboard

## 📊 Cost Savings

| Query Type | Traditional (GPT-4) | SmartRoute-AI | Savings |
|------------|---------------------|---------------|---------|
| Simple | $0.03 | $0.00 (Groq) | **100%** |
| Medium | $0.03 | $0.00 (Groq) | **100%** |
| Complex | $0.03 | $0.00 (Groq) | **100%** |

*Using Groq's free tier for all queries!*

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Query     │────▶│  Complexity  │────▶│  Model Router   │
│   Input     │     │  Classifier  │     │  (Tiered)       │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                    ┌──────────────────────────────┼──────────────────────────────┐
                    │                              │                              │
                    ▼                              ▼                              ▼
            ┌───────────────┐            ┌───────────────┐            ┌───────────────┐
            │ Llama 3.1 8B  │            │  Qwen 32B     │            │ Llama 3.3 70B │
            │   (Simple)    │            │   (Medium)    │            │   (Complex)   │
            │  560 tok/sec  │            │  400 tok/sec  │            │  280 tok/sec  │
            └───────────────┘            └───────────────┘            └───────────────┘
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/SmartRoute-AI.git
cd SmartRoute-AI

# Using conda
conda create -n SmartRoute-AI python=3.10
conda activate SmartRoute-AI

# Install dependencies
pip install poetry
poetry install
```

### 2. Get API Key (FREE)

1. Go to [console.groq.com/keys](https://console.groq.com/keys)
2. Sign up and create an API key
3. Copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=your-key-here
```

### 3. Train Classifier

```bash
python scripts/train_classifier.py
```

### 4. Run API

```bash
# Set API key and run
$env:GROQ_API_KEY = "your-key-here"  # Windows PowerShell
export GROQ_API_KEY="your-key-here"  # Linux/Mac

python api/main.py
```

### 5. Test It!

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) and try:

```json
{
  "query": "What is machine learning?",
  "strategy": "cost_optimized",
  "use_retrieval": true
}
```

## 📁 Project Structure

```
SmartRoute-AI/
├── api/                    # FastAPI application
│   └── main.py
├── config/                 # Configuration files
│   ├── models.yaml         # Model definitions
│   └── routing.yaml        # Routing strategies
├── dashboard/              # Streamlit dashboard
│   └── app.py
├── data/
│   ├── documents/          # PDF documents for RAG
│   ├── embeddings/         # Vector store (ChromaDB)
│   └── costs/              # Cost tracking database
├── models/
│   └── classifiers/        # Trained ML classifiers
├── src/
│   ├── cost/               # Cost tracking & budgets
│   ├── models/             # LLM wrappers (Groq, OpenAI, Local)
│   ├── pipeline/           # Main inference pipeline
│   ├── retrieval/          # RAG components
│   └── routing/            # Query router & classifier
├── tests/                  # Unit tests
├── notebooks/              # Jupyter notebooks
└── scripts/                # Utility scripts
```

## 🔧 Configuration

### Routing Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `cost_optimized` | Uses smallest effective model | Production (default) |
| `quality_first` | Always uses 70B model | High-stakes queries |
| `balanced` | Middle ground | General use |

### Model Tiers

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
# Install dev dependencies
poetry install --with dev

# Run tests
pytest tests/ -v
```

## 📄 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Process a query |
| `/stats` | GET | Get usage statistics |
| `/savings` | GET | View cost savings |
| `/budget` | GET | Check budget status |
| `/health` | GET | Health check |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.
