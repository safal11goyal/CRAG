# 📚 Adaptive CRAG — Corrective & Adaptive RAG System

> A production-grade **Retrieval-Augmented Generation (RAG)** pipeline powered by **Gemini**, **Cohere**, **Pinecone**, and **LangGraph** — with hybrid retrieval, evidence verification, query rewriting, web search fallback, and persistent conversation history.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🤖 **LLM** | Google Gemini (`gemini-3.5-flash-lite`) via `langchain-google-genai` |
| 🔍 **Embeddings** | Cohere `embed-english-v3.0` (1024-dim) |
| 📦 **Vector Store** | Pinecone (cloud-native, serverless) |
| 🔎 **Hybrid Search** | Dense (Pinecone) + Sparse (BM25) fused via **Reciprocal Rank Fusion (RRF)** |
| 🧠 **RAG Workflow** | Adaptive/Corrective RAG via **LangGraph `StateGraph`** |
| ✅ **Evidence Verification** | LLM-graded structured output with `pydantic` |
| ✏️ **Query Rewriting** | Iterative refinement on verification failure |
| 🌐 **Web Search Fallback** | Tavily Search when local docs are insufficient |
| 💾 **Persistence** | PostgreSQL (Supabase) via SQLAlchemy for full conversation history |
| 🖥️ **UI** | Streamlit web app (`app.py`) |
| 🖧 **CLI** | Interactive terminal interface (`main.py`) |

---

## 🗂️ Project Structure

```
RAG/
├── app.py                    # Streamlit web UI
├── main.py                   # Terminal CLI interface
├── config.py                 # Typed settings (pydantic-settings + .env)
├── database.py               # PostgreSQL persistence (SQLAlchemy ORM)
├── requirements.txt          # Python dependencies
├── .env                      # API keys and configuration (not committed)
│
└── pipeline/
    ├── __init__.py           # Package exports
    ├── pinecone_store.py     # Custom Pinecone VectorStore wrapper
    ├── ingestion.py          # Document loading, chunking, indexing
    ├── retrieval.py          # Hybrid retrieval (BM25 + Vector + RRF)
    ├── query.py              # Query understanding & rewriting
    ├── verification.py       # Evidence verification (structured LLM output)
    ├── generation.py         # Grounded answer generation
    ├── web_search.py         # Tavily web search fallback
    └── graph.py              # LangGraph StateGraph workflow
```

---

## 🔄 System Flowchart

```mermaid
flowchart TD
    A([User Question]) --> B[check_retrieval_needed]

    B -- Conversational / Greeting --> C["direct_chat_node\n(LLM with chat history)"]
    C --> Z([Response to User])

    B -- Factual / Document Question --> D["understand_query_node\n(Query Optimizer LLM)"]

    D --> E[retrieve_node]

    subgraph HYBRID_RETRIEVAL["Hybrid Retrieval via RRF"]
        direction TB
        E1["Pinecone Dense Search\n(Cohere Embeddings)"]
        E2["BM25 Sparse Search\n(Keyword Index)"]
        E3["Reciprocal Rank Fusion\n(RRF Merging)"]
        E1 & E2 --> E3
    end

    E --> HYBRID_RETRIEVAL
    E3 --> F["verify_evidence_node\n(Structured LLM Verifier)"]

    F -- Sufficient Evidence --> G["generate_answer_node\n(Grounded Answer Generation)"]
    G --> Z

    F -- "Insufficient Evidence\n(iterations < max)" --> H["improve_query_node\n(Query Refinement LLM)"]
    H --> E

    F -- "Max Iterations Reached" --> I["web_search_node\n(Tavily Web Search API)"]
    I -- Results Found --> G
    I -- No Results --> J["handle_insufficient_evidence\n(Graceful Failure Message)"]
    J --> Z

    subgraph PERSISTENCE["Persistence Layer"]
        direction LR
        DB[("PostgreSQL\n(Supabase)")]
    end

    Z --> PERSISTENCE
```

---

## 🏗️ Architecture Deep Dive

### 1. Configuration — `config.py`
- Uses **`pydantic-settings`** `BaseSettings` for typed, validated settings loaded from `.env`
- Integer fields (`CHUNK_SIZE`, `RETRIEVAL_K`, etc.) are automatically coerced — no manual `int(os.getenv(...))` calls
- Database URL normalized via **`sqlalchemy.engine.make_url`**, which correctly handles URL-encoded passwords (e.g. `%40` for `@`)
- All module-level constants (`config.GOOGLE_API_KEY`, `config.CHUNK_SIZE`, etc.) remain importable unchanged by all callers

### 2. Document Ingestion — `pipeline/ingestion.py`
- Supports **PDF** (`PyPDFLoader`), **TXT** (`TextLoader`), **DOCX** (`Docx2txtLoader`)
- Files processed in sorted order via `pathlib.Path.iterdir()` for reproducible, cross-platform indexing
- Chunked by `RecursiveCharacterTextSplitter` (default: 1000 tokens, 200 overlap)
- Embeddings batched via Cohere with exponential-backoff retry (`tenacity`) on rate limits
- Upserted to **Pinecone** cloud vector index; skips re-indexing if vectors already exist

### 3. Hybrid Retrieval — `pipeline/retrieval.py`

The retrieval score is computed by **Reciprocal Rank Fusion (RRF)**:

```
score(doc) = Σ   1 / (60 + rank_in_list)
              over each list doc appears in
```

- **Dense**: Pinecone similarity search using Cohere embeddings (semantic understanding)
- **Sparse**: BM25Okapi over all document chunks with a `score > 0` filter (keyword precision)
- **Fusion**: RRF (constant k=60, from the original paper) promotes documents that appear in both result sets

> **Why not `BM25Retriever` from `langchain-community`?**  
> `BM25Retriever` always returns exactly `k` results regardless of BM25 score, removing the meaningful `score > 0` quality filter. The custom `BM25Index` wrapper is kept intentionally.

### 4. LangGraph Workflow — `pipeline/graph.py`

| Node | Responsibility |
|---|---|
| `check_retrieval_needed` | Intent router — greetings skip retrieval entirely via fast-path lookup (0 ms LLM latency) |
| `understand_query` | Rewrites the user question into a keyword-dense search query; resolves coreferences from chat history |
| `retrieve` | Runs hybrid retrieval (Pinecone dense + BM25 sparse + RRF fusion) |
| `verify_evidence` | Conservative structured LLM verifier — only passes if documents contain **concrete facts** |
| `improve_query` | Iteratively refines the query when evidence is insufficient (up to `MAX_ITERATIONS`) |
| `web_search` | Tavily web search fallback after all local retrieval iterations are exhausted |
| `generate_answer` | Grounded answer generation — LLM is explicitly forbidden from using prior knowledge |
| `direct_chat` | Conversational response with full chat history (no retrieval overhead) |
| `handle_insufficient_evidence` | Graceful failure with source citations; confidence = 0.1 |

### 5. Evidence Verification — `pipeline/verification.py`
- Uses `llm.with_structured_output(EvidenceVerification)` — Pydantic schema enforces typed JSON response
- Conservative threshold: evidence must contain concrete facts, not just keyword overlap
- Returns `EvidenceVerification(enough_evidence: bool, reason: str)`

### 6. Answer Generation — `pipeline/generation.py`
- Context-grounded only — LLM system prompt explicitly prohibits use of prior knowledge
- Structured output: `AnswerOutput(answer: str, confidence: float, warning: str)`
- Confidence clamped to `[0.0, 1.0]`; warning field always populated

### 7. Persistence — `database.py`
- SQLAlchemy ORM with two tables: `conversations` and `conversation_turns`
- Stores: session ID, original question, optimized query, retrieved sources, answer, confidence, warning, iteration count
- Runs in **offline mode** gracefully when `DATABASE_URL` is not configured

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- [Pinecone](https://app.pinecone.io/) account (free tier works)
- [Google AI Studio](https://aistudio.google.com/) API key (Gemini)
- [Cohere](https://cohere.com/) API key
- [Supabase](https://supabase.com/) / PostgreSQL (optional — for conversation persistence)
- [Tavily](https://app.tavily.com/) API key (optional — for web search fallback)

### 1. Clone & Create Virtual Environment

```bash
git clone <your-repo-url>
cd RAG

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
# ── LLM & Embedding ────────────────────────────────────────
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite

COHERE_API_KEY=your_cohere_api_key_here
EMBEDDING_MODEL=embed-english-v3.0

# ── Pinecone Vector Store ───────────────────────────────────
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=rag-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# ── Web Search Fallback (optional) ─────────────────────────
TAVILY_API_KEY=your_tavily_api_key_here

# ── Retrieval Tuning ────────────────────────────────────────
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=5
MAX_ITERATIONS=3

# ── Persistence (optional) ──────────────────────────────────
DATABASE_URL=postgresql://user:password@host:5432/dbname

# ── Directories ─────────────────────────────────────────────
DOCUMENTS_DIR=documents
```

### 4. Add Documents

Place your PDF, TXT, or DOCX files in the `documents/` directory:

```
documents/
├── annual_report_2024.pdf
├── policy_manual.docx
└── research_notes.txt
```

The first run will automatically chunk and embed all documents into Pinecone.

### 5. Run

**Streamlit Web App:**
```bash
streamlit run app.py
```

**Terminal CLI:**
```bash
python main.py
```

---

## 🚀 End-to-End Example

### Happy Path (Evidence Found Locally)

```
User: "What was India's GDP growth rate in 2023?"

1. check_retrieval_needed  →  needs_retrieval = True  (factual question)
2. understand_query        →  "India GDP growth rate 2023 economic data"
3. retrieve                →  Pinecone top-5 + BM25 top-5 → RRF → top-5 fused
4. verify_evidence         →  enough_evidence = True  (GDP data found in context)
5. generate_answer         →  "India's GDP grew by 7.2% in FY2023..."
                               confidence = 0.92
                               sources = ["india_economy.pdf, page 14"]
```

### Fallback Path (Evidence Not Found → Web Search)

```
User: "What is the latest WHO recommendation on mpox vaccines?"

1. check_retrieval_needed  →  needs_retrieval = True
2. understand_query        →  "WHO mpox vaccine recommendation 2024"
3. retrieve (iter 1)       →  local docs retrieved, no mpox coverage
4. verify_evidence         →  enough_evidence = False
5. improve_query           →  "monkeypox mpox WHO vaccination guidelines"
6. retrieve (iter 2)       →  still no relevant local coverage
7. verify_evidence         →  enough_evidence = False
8. retrieve (iter 3)       →  max_iterations reached
9. web_search              →  Tavily fetches 5 live web results
10. generate_answer        →  answer from web results + notice banner
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `langchain-core` | ≥0.3.0 | Base primitives: Documents, Messages, `StrOutputParser` |
| `langchain-community` | ≥0.3.0 | Document loaders (PyPDF, TextLoader, Docx2txt) |
| `langchain-google-genai` | ≥2.0.0 | Gemini LLM and embedding integration |
| `langchain-cohere` | ≥0.3.0 | Cohere embedding integration |
| `langchain-text-splitters` | ≥0.1.0 | `RecursiveCharacterTextSplitter` |
| `langgraph` | ≥0.2.0 | `StateGraph` workflow engine |
| `pinecone` | ≥10.0.0 | Cloud vector database |
| `rank-bm25` | ≥0.2.2 | `BM25Okapi` sparse keyword index |
| `cohere` | ≥5.0.0 | Cohere API client |
| `tavily-python` | ≥0.3.0 | Web search fallback |
| `pydantic` | ≥2.0.0 | Structured output schemas |
| `pydantic-settings` | ≥2.0.0 | Typed settings from `.env` |
| `sqlalchemy` | ≥2.0.0 | PostgreSQL ORM |
| `psycopg2-binary` | ≥2.9.10 | PostgreSQL driver |
| `streamlit` | ≥1.30.0 | Web UI |
| `tenacity` | ≥8.0.0 | Retry logic for rate-limited API calls |
| `pypdf` | ≥4.0.0 | PDF parsing |
| `python-docx` | ≥1.1.0 | DOCX parsing |
| `python-dotenv` | ≥1.0.0 | `.env` file loading (transitional) |

---

## 🔧 Configuration Reference

All settings are defined in [`config.py`](config.py) and loaded from `.env` via `pydantic-settings`:

| Variable | Type | Default | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | `str` | — | Gemini API key (**required**) |
| `COHERE_API_KEY` | `str` | — | Cohere API key (**required**) |
| `PINECONE_API_KEY` | `str` | — | Pinecone API key (**required**) |
| `TAVILY_API_KEY` | `str` | — | Tavily key (optional, enables web fallback) |
| `GEMINI_MODEL` | `str` | `gemini-3.5-flash-lite` | Gemini model name |
| `EMBEDDING_MODEL` | `str` | `embed-english-v3.0` | Cohere embedding model |
| `PINECONE_INDEX_NAME` | `str` | `rag-index` | Pinecone index name |
| `PINECONE_CLOUD` | `str` | `aws` | Pinecone cloud provider |
| `PINECONE_REGION` | `str` | `us-east-1` | Pinecone cloud region |
| `CHUNK_SIZE` | `int` | `1000` | Token size per document chunk |
| `CHUNK_OVERLAP` | `int` | `200` | Overlap between consecutive chunks |
| `RETRIEVAL_K` | `int` | `5` | Documents retrieved per search |
| `MAX_ITERATIONS` | `int` | `3` | Max query-rewrite iterations before web fallback |
| `DOCUMENTS_DIR` | `str` | `documents` | Directory containing source documents |
| `DATABASE_URL` | `str` | — | PostgreSQL connection URL (optional) |

---

## 🗃️ Database Schema

When `DATABASE_URL` is configured, two tables are created automatically:

```
conversations
├── session_id      TEXT   (PK, UUID-based)
├── title           TEXT   (optional session label)
├── created_at      TIMESTAMP WITH TIME ZONE
└── updated_at      TIMESTAMP WITH TIME ZONE

conversation_turns
├── id              INTEGER  (PK, auto-increment)
├── session_id      TEXT     (FK → conversations.session_id)
├── question        TEXT     (user's original question)
├── search_query    TEXT     (optimized retrieval query)
├── answer          TEXT     (generated answer)
├── confidence      FLOAT    (0.0 – 1.0)
├── warning         TEXT     (hallucination risk assessment)
├── sources         JSON     (list of document citations)
├── iterations      INTEGER  (query-rewrite count)
└── created_at      TIMESTAMP WITH TIME ZONE
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes while preserving all existing RAG business logic
4. Run the functional smoke test: `python -m tests.smoke_test`
5. Submit a pull request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
