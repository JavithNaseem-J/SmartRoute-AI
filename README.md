# 🚀 SmartRoute-AI

**Cost-Optimized LLM Routing System with RAG**

> Intelligently routes queries to the most cost-effective model while maintaining quality. Uses Groq's **free API** for 100% cost savings.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🎯 Features

- **Smart Routing** - ML classifier routes queries to optimal models based on complexity
- **Tiered Models** - Simple → 8B, Medium → 17B, Complex → 70B
- **RAG Integration** - Hybrid search with ChromaDB + BM25
- **Cost Tracking** - Real-time monitoring with budget alerts
- **Dashboard** - Streamlit analytics UI

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/JavithNaseem-J/SmartRoute-AI.git
cd SmartRoute-AI

# Create environment
conda create -n SmartRoute-AI python=3.10 -y
conda activate SmartRoute-AI

# Install dependencies
pip install -r requirements.txt
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

```bash
streamlit run app.py
# Open http://localhost:8501
```

---

## 🐳 Docker

```bash
docker-compose up --build
# Open http://localhost:8501
```

---

## 🚀 Deploy on Render

### Quick Deploy

1. Push code to GitHub
2. Go to [dashboard.render.com](https://dashboard.render.com)
3. Click **New** → **Blueprint**
4. Connect your repository
5. Add `GROQ_API_KEY` environment variable
6. Deploy!

**Live URL:** `https://smartroute-ai.onrender.com`

---

## 📦 Dependencies

| File | Purpose |
|------|---------|
| `requirements.txt` | Production dependencies |
| `requirements-dev.txt` | Development + testing tools |

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.
