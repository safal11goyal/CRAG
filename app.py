"""Streamlit Web UI for Modular Corrective/Adaptive RAG Chatbot.

Features:
- Left sidebar with "+ New Conversation" button and previous conversations from PostgreSQL (Supabase)
- Real-time response streaming into chat bubbles
- Real-time pipeline status reasoning steps (st.status)
- Expandable source citations and evidence confidence metrics
- Persistent session storage in PostgreSQL
"""

import logging
import warnings

# Suppress noisy watcher warnings for optional third-party packages
warnings.filterwarnings("ignore")
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import time
import uuid
import streamlit as st

import config
import database
from langchain_cohere import CohereEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pipeline.graph import build_rag_graph
from pipeline.ingestion import load_or_create_vectorstore
from pipeline.retrieval import BM25Index

# ============================================================================
# PAGE CONFIG & MODERN CSS
# ============================================================================

st.set_page_config(
    page_title="RAG Intelligence Chatbot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for high-contrast, crystal-clear readability
st.markdown(
    """
    <style>
    /* Global Base & Dark Layout */
    html, body, [class*="css"], .stApp {
        background-color: #0b0f19 !important;
        background: radial-gradient(circle at top right, #111827 0%, #0b0f19 80%) !important;
        color: #f8fafc !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Headings & Text Clarity */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    p, span, label, div {
        color: #f1f5f9;
    }
    .stCaption, [data-testid="stCaptionContainer"] p {
        color: #94a3b8 !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b !important;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span {
        color: #cbd5e1 !important;
    }

    /* Primary 'New Conversation' Button */
    [data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 8px !important;
        padding: 0.65rem 1rem !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
        width: 100% !important;
        transition: all 0.2s ease-in-out !important;
    }
    [data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6) !important;
        transform: translateY(-1px) !important;
    }

    /* Previous Conversation Session Buttons */
    [data-testid="stSidebar"] div.stButton > button[kind="secondary"] {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 0.6rem 0.8rem !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        width: 100% !important;
        margin-bottom: 0.35rem !important;
        transition: all 0.15s ease-in-out !important;
    }
    [data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {
        background-color: #2563eb !important;
        border-color: #60a5fa !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
    }

    /* Chat Messages - High Contrast Text & Distinct Cards */
    [data-testid="stChatMessage"] {
        background-color: #172033 !important;
        border: 1px solid #2e3d5b !important;
        border-radius: 12px !important;
        padding: 1.25rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
        color: #f8fafc !important;
        font-size: 1.05rem !important;
        line-height: 1.7 !important;
    }

    /* Expanders & Sources Accordion */
    [data-testid="stExpander"] {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        margin-top: 0.75rem !important;
    }
    [data-testid="stExpander"] summary {
        color: #60a5fa !important;
        font-weight: 600 !important;
    }
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {
        color: #60a5fa !important;
        font-weight: 600 !important;
    }
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] li {
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
    }

    /* Bottom Container & Chat Input Bar (Eliminating white footer) */
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"],
    footer {
        background-color: #0b0f19 !important;
        background: #0b0f19 !important;
        border-top: 1px solid #1e293b !important;
    }
    [data-testid="stChatInput"] {
        background-color: #1e293b !important;
        border: 1.5px solid #3b82f6 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #ffffff !important;
        background-color: transparent !important;
        font-size: 1rem !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #94a3b8 !important;
    }
    /* Chat Input Send Button - Crisp white arrow & modern styling */
    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.4) !important;
        transition: all 0.2s ease-in-out !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    [data-testid="stChatInput"] button:hover {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.6) !important;
        transform: scale(1.05) !important;
    }
    [data-testid="stChatInput"] button svg {
        fill: #ffffff !important;
        color: #ffffff !important;
        stroke: #ffffff !important;
        width: 18px !important;
        height: 18px !important;
    }
    [data-testid="stChatInput"] button:disabled {
        background: rgba(255, 255, 255, 0.08) !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
    }
    [data-testid="stChatInput"] button:disabled svg {
        fill: #64748b !important;
        color: #64748b !important;
        stroke: #64748b !important;
    }

    /* Status Container */
    [data-testid="stStatusWidget"] {
        background-color: #1e293b !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 10px !important;
    }
    [data-testid="stStatusWidget"] span,
    [data-testid="stStatusWidget"] div {
        color: #ffffff !important;
    }

    /* Badges */
    .badge-connected {
        display: inline-flex;
        align-items: center;
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399 !important;
        border: 1px solid rgba(52, 211, 153, 0.4);
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-offline {
        display: inline-flex;
        align-items: center;
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171 !important;
        border: 1px solid rgba(248, 113, 113, 0.4);
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Code & Metrics */
    code {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border: 1px solid #334155 !important;
        padding: 0.15rem 0.4rem !important;
        border-radius: 4px !important;
    }
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# CACHED PIPELINE INITIALIZATION
# ============================================================================

@st.cache_resource(show_spinner="⚡ Initializing RAG Knowledge Base & Embeddings...")
def load_rag_app():
    """Load Cohere embeddings, Gemini LLM, FAISS vectorstore, BM25, and compile LangGraph."""
    # Ensure database is connected
    database.init_db()

    # 1. Embeddings & LLM
    embeddings = CohereEmbeddings(
        model=config.EMBEDDING_MODEL,
        cohere_api_key=config.COHERE_API_KEY,
        max_retries=3,
        request_timeout=60,
    )
    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        max_retries=3,
        timeout=60,
    )

    # 2. Vectorstore & Ingestion
    vectorstore, chunks = load_or_create_vectorstore(
        documents_dir=config.DOCUMENTS_DIR,
        vectorstore_dir=config.VECTORSTORE_DIR,
        embeddings=embeddings,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    # 3. BM25 Index
    bm25_index = BM25Index(chunks)

    # 4. Build LangGraph Workflow
    rag_app = build_rag_graph(
        vectorstore=vectorstore,
        bm25_index=bm25_index,
        llm=llm,
        k=config.RETRIEVAL_K,
        max_iterations=config.MAX_ITERATIONS,
    )

    return rag_app


# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "session_title" not in st.session_state:
    st.session_state.session_title = "New Conversation"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize DB connection on every session run if needed
if not database.is_db_connected():
    database.init_db()


def start_new_conversation():
    """Reset session state to begin a new chat conversation."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_title = "New Conversation"
    st.session_state.messages = []
    st.rerun()


def load_conversation(session_id: str, title: str):
    """Load past conversation turns from PostgreSQL into session state."""
    turns = database.get_conversation_history(session_id)
    messages = []
    for turn in turns:
        messages.append({
            "role": "user",
            "content": turn["user_query"],
        })
        messages.append({
            "role": "assistant",
            "content": turn["answer"],
            "sources": turn.get("sources", []),
            "confidence": turn.get("confidence", 0.0),
            "warning": turn.get("warning", ""),
            "enough_evidence": turn.get("enough_evidence", False),
        })

    st.session_state.session_id = session_id
    st.session_state.session_title = title
    st.session_state.messages = messages
    st.rerun()


# ============================================================================
# LEFT SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### ⚡ RAG Chatbot")
    st.caption("Modular Corrective/Adaptive RAG Engine")

    # + New Conversation Button
    if st.button("➕ New Conversation", type="primary", use_container_width=True):
        start_new_conversation()

    st.markdown("---")

    # Previous Conversations List from PostgreSQL
    st.markdown("#### 💬 Previous Conversations")

    if database.is_db_connected():
        recent_sessions = database.list_conversations(limit=20)
        if not recent_sessions:
            st.caption("No conversations yet. Start a chat!")
        else:
            for conv in recent_sessions:
                s_id = conv["session_id"]
                title = conv["title"] or "Untitled Chat"
                turn_count = conv.get("turns_count", 0)
                is_active = (s_id == st.session_state.session_id)

                # Distinct icon and style for active session vs previous sessions
                if is_active:
                    label = f"📌 {title[:26]} ({turn_count})"
                else:
                    label = f"🗨️ {title[:26]} ({turn_count})"

                if st.button(
                    label,
                    key=f"session_btn_{s_id}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True,
                    help=f"Session: {s_id}\nTurns: {turn_count}",
                ):
                    if not is_active:
                        load_conversation(s_id, title)
    else:
        st.info("PostgreSQL is currently offline. Sessions are stored in memory for this session.")

    st.markdown("---")

    # System Status Badges
    st.markdown("#### ⚙️ System Status")
    if database.is_db_connected():
        st.markdown('Database: <span class="badge-connected">● Supabase Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('Database: <span class="badge-offline">○ Offline Mode</span>', unsafe_allow_html=True)

    st.caption(f"**LLM:** `{config.GEMINI_MODEL}`")
    st.caption(f"**Embeddings:** `{config.EMBEDDING_MODEL}`")
    st.caption(f"**Retrieval Top-K:** `{config.RETRIEVAL_K}`")


# ============================================================================
# MAIN CHAT VIEW
# ============================================================================

# Main Header
st.title("⚡ RAG Assistant")
st.caption(f"Active Session: `{st.session_state.session_title}`")

# Initialize RAG Pipeline (cached)
rag_pipeline = load_rag_app()

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Only render citations & grounding assessment if documents were retrieved & cited
        sources = message.get("sources", [])
        if message["role"] == "assistant" and sources:
            confidence = message.get("confidence", 0.0)
            warning = message.get("warning", "")

            with st.expander("📚 Sources & Grounding Assessment", expanded=False):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Evidence Grounding", f"{int(confidence * 100)}%")
                with col2:
                    if "Low risk" in warning:
                        st.success(warning)
                    else:
                        st.warning(warning)

                st.markdown("**Cited Sources:**")
                for src in sources:
                    st.markdown(f"- 📄 `{src}`")


# Chat Input
user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    # 1. Display User Message immediately
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Ensure parent session exists in PostgreSQL with question as title
    if database.is_db_connected():
        if st.session_state.session_title == "New Conversation":
            title = (user_query[:35] + "...") if len(user_query) > 35 else user_query
            st.session_state.session_title = title
            database.create_conversation(session_id=st.session_state.session_id, title=title)

    # 3. Assistant Response Container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("...")

        # Convert session history to LangChain BaseMessage objects for chat context
        chat_history_messages = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_history_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history_messages.append(AIMessage(content=msg["content"]))

        initial_state = {
            "messages": chat_history_messages,
            "question": user_query,
            "search_query": user_query,
            "needs_retrieval": True,
            "documents": [],
            "enough_evidence": False,
            "verification_reason": "",
            "answer": "",
            "confidence": 0.0,
            "warning": "",
            "sources": [],
            "iterations": 0,
        }

        try:
            final_state = rag_pipeline.invoke(initial_state)
        except Exception as e:
            message_placeholder.empty()
            st.error(f"⚠️ A network connection issue occurred while communicating with the AI service: {e}. Please try sending your message again.")
            st.stop()

        needs_retrieval = final_state.get("needs_retrieval", True)
        enough_evidence = final_state.get("enough_evidence", False)
        iterations = final_state.get("iterations", 0)

        # Extract answer & metadata
        answer = final_state.get("answer", "I could not generate an answer.")
        sources = final_state.get("sources", [])
        confidence = final_state.get("confidence", 0.0)
        warning = final_state.get("warning", "")
        verification_reason = final_state.get("verification_reason", "")
        search_query = final_state.get("search_query", user_query)

        # Streaming generator to simulate live token generation
        def stream_generator():
            words = answer.split(" ")
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
                time.sleep(0.015)

        # Stream answer into chat UI, replacing the three dots
        full_response = message_placeholder.write_stream(stream_generator)

        # Render Grounding Accordion ONLY if documents were retrieved & cited
        if sources:
            with st.expander("📚 Sources & Grounding Assessment", expanded=False):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Evidence Grounding", f"{int(confidence * 100)}%")
                with col2:
                    if "Low risk" in warning:
                        st.success(warning)
                    else:
                        st.warning(warning)

                st.markdown("**Cited Sources:**")
                for src in sources:
                    st.markdown(f"- 📄 `{src}`")

        # Save turn to PostgreSQL
        database.save_conversation_turn(
            session_id=st.session_state.session_id,
            user_query=user_query,
            search_query=search_query,
            answer=answer,
            sources=sources,
            enough_evidence=enough_evidence,
            verification_reason=verification_reason,
            confidence=confidence,
            warning=warning,
            iterations=iterations,
        )

        # Store in session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "confidence": confidence,
            "warning": warning,
            "enough_evidence": enough_evidence,
        })

        # Trigger quick rerun to refresh sidebar conversation count/title
        st.rerun()
