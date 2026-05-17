# Ask My Docs — Local RAG Pipeline

A production-grade Retrieval-Augmented Generation (RAG) system that runs entirely on your local machine. No cloud APIs required. Point it at any document and ask questions — it retrieves the most relevant chunks, reranks them, and generates a cited answer using a local LLM.

---

## How it works

```
User query
    │
    ▼
┌─────────────────────────────────┐
│        Hybrid Retrieval         │
│  BM25 (sparse) + FAISS (dense)  │
└─────────────────────────────────┘
    │
    ▼
  RRF Fusion  ←  Reciprocal Rank Fusion (k=60)
    │
    ▼
  Cross-Encoder Reranker  ←  MiniLM-L6-v2
    │
    ▼
  Ollama LLM  ←  qwen2.5-coder:14b (local)
    │
    ▼
  Cited Answer  ←  [1] [2] [3] source references
```

---

## Tech stack

| Component | Library / Tool |
|---|---|
| Orchestration | LangChain, LangChain-Community |
| Vector store | FAISS (local) |
| Sparse retrieval | BM25 via rank-bm25 |
| Embeddings | nomic-embed-text via Ollama |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | qwen2.5-coder:14b via Ollama |
| Serving | FastAPI + Uvicorn |
| Vector DB (optional) | Qdrant |
| CI/CD | GitHub Actions |
| Containerisation | Docker + Docker Compose |
| Evaluation | RAGAS + golden dataset (20 questions) |

---

## Project structure

```
ask-my-docs/
├── app/
│   ├── api/v1/              # FastAPI route handlers
│   ├── core/
│   │   └── config.py        # Centralised settings (Pydantic)
│   ├── retrieval/
│   │   ├── hybrid_retriever.py       # BM25 + FAISS + RRF
│   │   └── cross_encoder_reranker.py # MiniLM reranker
│   ├── generation/
│   │   ├── prompt_builder.py  # Citation-enforced prompt
│   │   └── llm_client.py      # Ollama LLM wrapper
│   └── ingestion/
│       └── loader.py          # DOCX chunker
├── tests/
│   ├── unit/                  # Import + config tests
│   ├── integration/           # Full pipeline tests
│   └── evals/
│       ├── golden_dataset.json  # 20 Q&A evaluation pairs
│       └── ragas_eval.py        # RAGAS metric runner
├── infra/
│   └── docker-compose.yml    # Qdrant + Elasticsearch
├── .github/workflows/
│   └── ci.yml                # Lint → Unit Tests → CI
├── ask.py                    # Interactive Q&A entrypoint
└── requirements.txt
```

---

## Prerequisites

- macOS (Apple Silicon or Intel)
- [Homebrew](https://brew.sh)
- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Ollama](https://ollama.com)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Sakshat101/ask-my-docs.git
cd ask-my-docs
```

### 2. Create and activate virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull Ollama models

```bash
ollama pull qwen2.5-coder:14b     # LLM for generation
ollama pull nomic-embed-text      # Embeddings
```

### 5. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` with your values:

```env
# ── Local Ollama (no API key needed) ──────────────────────
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen2.5-coder:14b
EMBEDDING_MODEL=nomic-embed-text

# ── Vector store ──────────────────────────────────────────
VECTOR_BACKEND=faiss
QDRANT_URL=http://localhost:6333

# ── Optional: swap Ollama for OpenAI ─────────────────────
# See "Switching to OpenAI" section below before uncommenting
# OPENAI_API_KEY=sk-...

# ── Optional: Anthropic Claude ────────────────────────────
# ANTHROPIC_API_KEY=sk-ant-...

# ── Optional: Cohere Rerank API ───────────────────────────
# COHERE_API_KEY=...

LOG_LEVEL=INFO
```

### 6. Start infrastructure (optional — only needed for integration tests)

```bash
docker compose -f infra/docker-compose.yml up -d
```

### 7. Run the pipeline

```bash
# Start Ollama server in a separate terminal tab
ollama serve

# Set PYTHONPATH and run
export PYTHONPATH=$(pwd)
python ask.py
```

---

## Switching to OpenAI (instead of Ollama)

> By default the pipeline runs fully locally using Ollama. If you want to use OpenAI instead, make these three changes:

**1. Update `.env`:**

```env
OPENAI_API_KEY=sk-your-real-key-here   # uncomment and fill in
# OLLAMA_BASE_URL=...                  # comment this out
```

**2. Update `app/retrieval/hybrid_retriever.py`:**

Change the import at the top:
```python
# Remove this line:
from langchain_ollama import OllamaEmbeddings

# Add this line:
from langchain_openai import OpenAIEmbeddings
```

Change the default embeddings in `__init__`:
```python
# Remove:
self.embeddings = embeddings or OllamaEmbeddings(model="nomic-embed-text")

# Add:
self.embeddings = embeddings or OpenAIEmbeddings()
```

**3. Update `app/generation/llm_client.py`:**

Change the LLM client:
```python
# Remove:
from langchain_ollama import OllamaLLM
self._llm = OllamaLLM(model=model, base_url=base_url, temperature=temperature)

# Add:
from langchain_openai import ChatOpenAI
self._llm = ChatOpenAI(model="gpt-4o", temperature=temperature)
```

---

## Switching to Anthropic Claude (instead of Ollama)

**1. Update `.env`:**

```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**2. Update `app/generation/llm_client.py`:**

```python
# Remove:
from langchain_ollama import OllamaLLM
self._llm = OllamaLLM(...)

# Add:
from langchain_anthropic import ChatAnthropic
self._llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=temperature)
```

Then install the Anthropic integration:
```bash
pip install langchain-anthropic
```

---

## Switching to Cohere Reranker (instead of MiniLM)

The cross-encoder runs locally by default. To use Cohere's managed reranker:

**1. Update `.env`:**
```env
COHERE_API_KEY=your-cohere-key
```

**2. Update `app/retrieval/cross_encoder_reranker.py`:**
```python
# Replace the CrossEncoder call with:
import cohere
co = cohere.Client(os.environ["COHERE_API_KEY"])
results = co.rerank(query=query, documents=[d.page_content for d in documents], top_n=top_k)
```

---

## Running tests

```bash
# Unit tests (no services needed)
pytest tests/unit/ -v

# Integration tests (requires Docker services running)
pytest tests/integration/ -v

# RAGAS evaluation against golden dataset
export PYTHONPATH=$(pwd)
python tests/evals/ragas_eval.py \
  --dataset tests/evals/golden_dataset.json \
  --output  reports/ragas-results.json
```

### CI/CD metric thresholds

| Metric | Minimum threshold |
|---|---|
| Faithfulness | 0.80 |
| Answer relevancy | 0.75 |
| Context precision | 0.70 |
| Context recall | 0.70 |

---

## GitHub Actions secrets

To enable the full CI pipeline, add these secrets in your repo under **Settings → Secrets and variables → Actions**:

| Secret | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Only if using OpenAI | Embeddings + generation |
| `ANTHROPIC_API_KEY` | Only if using Claude | Generation |
| `COHERE_API_KEY` | Optional | Managed reranking |
| `QDRANT_URL` | Optional | Qdrant Cloud endpoint |
| `QDRANT_API_KEY` | Optional | Qdrant Cloud auth |
| `SLACK_WEBHOOK_URL` | Optional | Failure alerts |

> The default local setup (Ollama) does not need any GitHub secrets to run unit and lint checks.

---

## Current CI status

| Job | Status | Notes |
|---|---|---|
| Lint & Type Check | Passing | ruff + mypy |
| Unit Tests | In progress | Fixing Ollama CI compatibility |
| Integration Tests | Skipped | Needs API keys in secrets |
| RAGAS Evaluation | Skipped | Needs API keys in secrets |

---

## Roadmap

- [ ] Fix unit tests for CI environment (mock Ollama calls)
- [ ] Build FastAPI `/query` endpoint
- [ ] Add Streamlit UI for document upload + Q&A
- [ ] Wire in API keys for full integration test run
- [ ] Add Slack webhook for CI failure alerts
- [ ] Support PDF ingestion alongside DOCX

---

## License

MIT
