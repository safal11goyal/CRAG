# 🧠 Corrective & Adaptive RAG (CRAG) with LangGraph

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-FF6F00.svg)](https://github.com/langchain-ai/langgraph)
[![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-4285F4.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![Cohere](https://img.shields.io/badge/Embeddings-Cohere%20v3-39594C.svg)](https://cohere.com/)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-00599C.svg)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-336791.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade, modular **Corrective and Adaptive Retrieval-Augmented Generation (CRAG)** system built with **LangGraph**, **Google Gemini**, **Cohere Embeddings**, local **FAISS + BM25 Hybrid Retrieval**, automated **Tavily Web Search** fallback, and **PostgreSQL** conversation telemetry persistence.

Available both as a **Streamlit Web Application** with live execution tracing and a rich **Terminal CLI**.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Architecture & Workflow](#-architecture--workflow)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Configuration (.env)](#-configuration-env)
- [Running the Application](#-running-the-application)
  - [1. Streamlit Web Interface](#1-streamlit-web-interface-recommended)
  - [2. Terminal CLI Chatbot](#2-terminal-cli-chatbot)
- [Database & Telemetry Persistence](#-database--telemetry-persistence)
- [How CRAG Works Under the Hood](#-how-crag-works-under-the-hood)
- [License](#-license)

---

## 🔭 Overview

Traditional RAG systems fail when:
1. **Query Mismatch**: User prompts are ambiguous or poorly phrased for vector search.
2. **Missing Information**: The internal knowledge base does not contain the answer, leading to confident hallucinations.
3. **Noisy Retrieval**: Irrelevant chunks dilute the context, degrading generation quality.

**CRAG solves these failure modes** through a self-correcting evaluation cycle:
- Evaluates retrieved evidence with a **conservative LLM judge**.
- If evidence is weak, **refines the query** with expanded terminology and retrieves again.
- If repeated internal retrieval fails, **autonomously searches the live web via Tavily**.
- Synthesizes answers strictly grounded in verified context, providing **confidence scores**, **deduplicated citations**, and **hallucination risk indicators**.

---

## 📐 Architecture & Workflow

The LangGraph state machine orchestrates the query understanding, hybrid retrieval, verification, query expansion, web search fallback, and response generation:

```mermaid
flowchart TD
     subgraph STARTUP["🚀 Startup & Ingestion"]
        A([Start]) --> B[Load .env / config.py]
        B --> C[Init PostgreSQL\ndatabase.init_db]
        C --> D[Init CohereEmbeddings\n+ ChatGoogleGenerativeAI LLM]
        D --> E{FAISS index\nexists on disk?}
        E -- Yes --> F[FAISS.load_local]
        E -- No --> G[Load Documents\nPyPDFLoader / TextLoader\nDocx2txtLoader]
        G --> H[RecursiveCharacterTextSplitter\nchunk_size=1000, overlap=200]
        H --> I[CohereEmbeddings\nbatch embed with tenacity retry]
        I --> J[FAISS.save_local]
        F --> K[Build BM25Index\nfrom chunks]
        J --> K
        K --> L[build_rag_graph\ncompile LangGraph]
        L --> M([System Ready])
    end
    subgraph GRAPH["⚡ CRAG LangGraph Runtime — Per Query"]
        N([User Question]) --> O[check_retrieval_needed]
        O --> P{Fast-path\ngreeting check}
        P -- Greeting/Conversational --> Q[direct_chat_node\nLLM with history]
        Q --> R([Answer + END])
        P -- Needs checking --> S[LLM: YES or NO?]
        S -- NO --> Q
        S -- YES --> T[understand_query\nLLM → search_query]
        T --> U[hybrid_retrieval\nFAISS similarity_search\n+ BM25Okapi.get_scores\n→ deduplicated docs]
        U --> V[verify_evidence\nLLM with_structured_output\n→ EvidenceVerification]
        V --> W{enough_evidence?}
        W -- True --> X[generate_answer\nLLM with_structured_output\n→ AnswerOutput]
        X --> Y[format_sources\ndeduplicate citations]
        Y --> Z([Answer + Sources + Confidence + END])
        W -- False\niterations < max_iterations --> AA[improve_query\nLLM → refined_query]
        AA --> U
        W -- False\niterations >= max_iterations --> AB[web_search\nTavilyClient.search]
        AB --> AC{Web results\nfound?}
        AC -- Yes --> X
        AC -- No --> AD[handle_insufficient_evidence\nlow-confidence fallback]
        AD --> Z
    end
    M --> N
    subgraph PERSIST["💾 Persistence — After Every Turn"]
        Z --> AE[database.save_conversation_turn\nPostgreSQL via SQLAlchemy]
        R --> AE
    end
```

---

## ✨ Key Features

- 🔍 **Hybrid Retrieval Engine**:
  - **Dense Vector Search**: Powered by **Cohere** (`embed-english-v3.0`) and local **FAISS**.
  - **Lexical Keyword Search**: Powered by **BM25 Okapi** for exact keyword, acronym, and entity matching.
  - **Score Normalization & Deduplication**: Reciprocal-rank and min-max score blending to rank the most relevant excerpts.
- 🔄 **Adaptive Query Refinement & Correction**:
  - Automatically translates raw conversational prompts into optimized search queries.
  - Iteratively broadens search terms and explores synonyms when retrieved context is insufficient.
- ⚖️ **Conservative Evidence Verification**:
  - Evaluates whether retrieved excerpts contain concrete facts directly answering the question.
  - Rejects incomplete or ambiguous evidence to protect against hallucinations.
- 🌐 **Tavily Web Search Fallback**:
  - If internal documents do not satisfy evidence requirements after `MAX_ITERATIONS` (default: 3), queries Tavily to retrieve live web results.
- 🛡️ **Strict Grounded Answer Synthesis**:
  - Responses strictly cite source documents (filenames, page numbers, or web URLs).
  - Outputs a **confidence score (0–100%)**, evidence support metric, and hallucination warning flags.
- 💻 **Modern Streamlit Web App (`app.py`)**:
  - Real-time pipeline status trackers showing each step (Query Understanding, Hybrid Retrieval, Verification, Web Search fallback, Generation).
  - Expandable source citation drawers with extracted passage snippets and relevance scores.
  - Multi-session conversation management with sidebar history.
- 🗄️ **PostgreSQL / Supabase Telemetry (`database.py`)**:
  - Automatically logs every conversation session, user prompt, system answer, retrieved sources, iteration count, confidence score, and fallback status via SQLAlchemy ORM.
  - Works with Supabase, Neon, Docker, local PostgreSQL, or gracefully falls back to in-memory mode if disabled.
- ⚡ **Local Vector Store Persistence**:
  - Caches FAISS embeddings in `vectorstore/` to eliminate redundant embedding API calls on startup.

---

## 📂 Project Structure

```
CRAG/
├── .streamlit/
│   └── config.toml          # Custom dark UI theme settings for Streamlit
├── documents/               # Internal knowledge base (PDF, TXT, DOCX)
│   ├── WHO_TB_2022.pdf
│   └── WHO_TB_2024.pdf
├── pipeline/
│   ├── __init__.py
│   ├── ingestion.py         # PDF parsing, chunking, Cohere embeddings & FAISS indexer
│   ├── retrieval.py         # FAISS dense + BM25 sparse hybrid retrieval
│   ├── query.py             # Query understanding & corrective query refinement
│   ├── verification.py      # Conservative LLM evidence verification
│   ├── generation.py        # Context-grounded response generation & citation parser
│   ├── web_search.py        # Tavily web search fallback integration
│   └── graph.py             # LangGraph state graph & conditional routing logic
├── .env.example             # Environment variable template
├── .gitignore               # Git ignore rules (secrets, venv, vectorstore)
├── app.py                   # Streamlit web application with real-time telemetry
├── config.py                # Centralized project configuration & settings
├── database.py              # PostgreSQL database models & session logging
├── main.py                  # Interactive terminal CLI chatbot
└── requirements.txt         # Python package dependencies
```

---

## 📦 Prerequisites

- **Python**: 3.10, 3.11, or 3.12
- **API Keys**:
  - [Google AI Studio](https://aistudio.google.com/) (`GOOGLE_API_KEY`) for Gemini models.
  - [Cohere Dashboard](https://dashboard.cohere.com/) (`COHERE_API_KEY`) for `embed-english-v3.0`.
  - [Tavily AI](https://tavily.com/) (`TAVILY_API_KEY`) for automated web search fallback.
  - *(Optional)* [PostgreSQL / Supabase](https://supabase.com/) (`DATABASE_URL`) for conversation persistence.

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/safal11goyal/CRAG.git
cd CRAG
```

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Place Your Documents

Add your knowledge base files (`.pdf`, `.docx`, `.txt`) into the `documents/` directory.

---

## ⚙️ Configuration (.env)

Copy `.env.example` to create your active `.env` file:

```bash
cp .env.example .env
```

Edit `.env` with your API keys and configuration:

```ini
# ==========================================
# LLM & Embedding API Keys
# ==========================================
GOOGLE_API_KEY=AIzaSy...your_gemini_api_key
GEMINI_MODEL=gemini-3.1-flash-lite

COHERE_API_KEY=your_cohere_api_key
EMBEDDING_MODEL=embed-english-v3.0

# ==========================================
# Web Search Fallback (Tavily)
# ==========================================
TAVILY_API_KEY=tvly-your_tavily_api_key

# ==========================================
# PostgreSQL Database (Optional - Supabase / Neon / Local)
# ==========================================
# Format: postgresql+psycopg2://user:password@host:port/dbname
DATABASE_URL=postgresql+psycopg2://postgres:your_password@localhost:5432/rag_db

# Or configure individual parameters:
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=
POSTGRES_DB=rag_db

# ==========================================
# Retrieval & Chunking Parameters
# ==========================================
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=4
MAX_ITERATIONS=3
DOCUMENTS_DIR=documents
VECTORSTORE_DIR=vectorstore
```

---

## 🖥️ Running the Application

### 1. Streamlit Web Interface (Recommended)

Launch the interactive web application:

```bash
streamlit run app.py
```

- Access the UI in your browser at `http://localhost:8501`.
- **Features in Web UI**:
  - **Live Execution Stepper**: Watch the agent rewrite queries, perform hybrid retrieval, verify evidence, trigger web search fallback, and generate the final answer in real time.
  - **Expandable Source Citations**: Inspect retrieved chunk excerpts, similarity scores, document filenames, and page numbers.
  - **Confidence Badges**: Instant visual indicators for evidence support, confidence level, and iteration count.
  - **Session Manager**: View, switch between, and create new conversation sessions in the sidebar.

### 2. Terminal CLI Chatbot

For command-line testing and lightweight interactive chats:

```bash
python main.py
```

- Features ANSI colored output displaying real-time step progressions, verification decisions, and final grounded answers.
- Type `exit` or `quit` to end the session.

---

## 🗄️ Database & Telemetry Persistence

The system includes built-in persistence using **SQLAlchemy** in `database.py`.

### Supported Providers:
- **Supabase**: Direct connection string via `DATABASE_URL`.
- **Neon / AWS RDS / DigitalOcean / Heroku Postgres**.
- **Local PostgreSQL** (Docker or native installation).
- **Graceful Fallback**: If no database credentials are provided, the application runs normally without database persistence.

### Relational Schema:
- **`conversation_sessions`**: Stores `session_id`, `created_at`, and `title`.
- **`conversation_turns`**: Stores `turn_id`, `session_id`, `user_query`, `refined_query`, `final_answer`, `confidence_score`, `evidence_support`, `iterations_used`, `fallback_used`, `retrieved_sources` (JSON), and `timestamp`.

---

## 🔬 How CRAG Works Under the Hood

| Node | Module | Description |
| :--- | :--- | :--- |
| **`understand_query`** | `pipeline/query.py` | Extracts key intent, removes conversational filler, and constructs targeted search phrases. |
| **`retrieve`** | `pipeline/retrieval.py` | Performs hybrid search combining FAISS cosine vector distance and BM25 token frequencies. |
| **`verify_evidence`** | `pipeline/verification.py` | Uses structured Gemini prompting to conservatively grade whether the context answers the user query. |
| **`refine_query`** | `pipeline/query.py` | Triggered if evidence is incomplete and `iteration < MAX_ITERATIONS`. Expands keywords and queries alternative angles. |
| **`web_search_fallback`**| `pipeline/web_search.py`| Triggered if `iteration >= MAX_ITERATIONS` without sufficient evidence. Fetches real-time web results via Tavily. |
| **`generate`** | `pipeline/generation.py` | Synthesizes a factual, grounded answer strictly from verified context with citations and confidence metrics. |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
