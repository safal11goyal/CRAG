import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

import config
import database
from pipeline.retrieval import BM25Index, _rrf_rank
from pipeline.query import check_retrieval_needed, understand_query, improve_query
from pipeline.verification import verify_evidence, EvidenceVerification
from pipeline.generation import format_sources, generate_answer, AnswerOutput
from pipeline.ingestion import chunk_documents
from pipeline.graph import build_rag_graph


# ==============================================================================
# TEST 1: BM25 Tokenization and Keyword Retrieval
# ==============================================================================
def test_bm25_index_search_and_score_filtering():
    """Verify BM25 tokenization, exact keyword matching, and score > 0 filtering."""
    doc1 = Document(
        page_content="Machine learning models require clean training datasets.",
        metadata={"source": "ml.txt", "page": 1},
    )
    doc2 = Document(
        page_content="Photosynthesis is the process used by plants to convert light energy.",
        metadata={"source": "bio.txt", "page": 2},
    )
    doc3 = Document(
        page_content="Deep learning and neural networks are subfields of machine learning.",
        metadata={"source": "dl.txt", "page": 3},
    )

    bm25 = BM25Index([doc1, doc2, doc3])

    # Query matching machine learning documents
    results = bm25.search("machine learning", k=2)
    assert len(results) == 2
    sources = [doc.metadata["source"] for doc in results]
    assert "ml.txt" in sources
    assert "dl.txt" in sources
    assert "bio.txt" not in sources

    # Query with non-matching keywords should return empty list (score > 0 filter)
    unrelated_results = bm25.search("astronomy telescope galaxy", k=2)
    assert len(unrelated_results) == 0

    # Query with empty or purely punctuation string
    assert bm25.search("", k=2) == []
    assert bm25.search("!@#$%^", k=2) == []


# ==============================================================================
# TEST 2: Reciprocal Rank Fusion (RRF) Ranking and Deduplication
# ==============================================================================
def test_rrf_rank_fusion_and_deduplication():
    """Verify Reciprocal Rank Fusion promotes dual-ranked items and deduplicates identical chunks."""
    doc_shared = Document(
        page_content="Corrective RAG refines ambiguous user search queries.",
        metadata={"source": "crag.pdf", "page": 4},
    )
    doc_dense_only = Document(
        page_content="Vector embeddings capture semantic similarity in high dimensional space.",
        metadata={"source": "vectors.pdf", "page": 1},
    )
    doc_sparse_only = Document(
        page_content="BM25 relies on inverted index frequency of exact token occurrences.",
        metadata={"source": "bm25.pdf", "page": 2},
    )

    # doc_shared appears in BOTH dense and sparse retrieval results
    vector_docs = [doc_shared, doc_dense_only]
    keyword_docs = [doc_shared, doc_sparse_only]

    fused_docs = _rrf_rank(vector_docs, keyword_docs, k=3, rrf_k=60)

    # 1. Total deduplicated results should be 3
    assert len(fused_docs) == 3

    # 2. doc_shared should have the highest combined RRF score and be ranked #1
    assert fused_docs[0].metadata["source"] == "crag.pdf"
    assert fused_docs[0].page_content == doc_shared.page_content

    # 3. Both remaining individual documents must be included without duplicate entries
    returned_sources = [d.metadata["source"] for d in fused_docs]
    assert len(set(returned_sources)) == 3
    assert "vectors.pdf" in returned_sources
    assert "bm25.pdf" in returned_sources


# ==============================================================================
# TEST 3: Query Intent Router Fast Paths (Conversational Greetings & Identity)
# ==============================================================================
def test_check_retrieval_needed_fast_paths():
    """Verify instant fast-path resolution for common greetings and personal identity questions."""
    mock_llm = MagicMock()

    # Greetings and pleasantries that should return False immediately (0ms, no LLM call)
    greetings = [
        "hi",
        "Hello!",
        "Good morning",
        "How are you doing?",
        "thank you very much",
        "bye",
    ]
    for greeting in greetings:
        assert check_retrieval_needed(greeting, mock_llm) is False

    # Identity and memory queries that should return False immediately
    identity_queries = [
        "What is my name?",
        "what's my name",
        "who am I?",
        "do you remember my name",
        "my name is Alex",
    ]
    for query in identity_queries:
        assert check_retrieval_needed(query, mock_llm) is False

    # Assert that mock_llm was never invoked for any of the fast-path queries
    assert mock_llm.invoke.call_count == 0


# ==============================================================================
# TEST 4: Query Intent Router LLM Classification & Error Fallback
# ==============================================================================
def test_check_retrieval_needed_llm_and_fallback():
    """Verify intent router invokes LLM for factual questions and falls back safely on error."""
    mock_llm = MagicMock()

    # Case A: Factual question where LLM responds with 'YES' -> needs retrieval
    mock_llm.invoke.return_value = AIMessage(content="YES")
    result_factual = check_retrieval_needed("What is the company revenue for Q3 2024?", mock_llm)
    assert result_factual is True
    assert mock_llm.invoke.call_count == 1

    # Case B: LLM classifies non-factual query as conversational 'NO' -> no retrieval
    mock_llm.invoke.return_value = AIMessage(content="NO")
    result_conversational = check_retrieval_needed("Tell me a creative joke about a robot.", mock_llm)
    assert result_conversational is False

    # Case C: If LLM raises an unexpected exception, safe fallback defaults to True (proceed with retrieval)
    mock_llm.invoke.side_effect = RuntimeError("API timeout")
    fallback_result = check_retrieval_needed("Explain the architecture flow in section 2.", mock_llm)
    assert fallback_result is True


# ==============================================================================
# TEST 5: Query Understanding and Coreference Resolution
# ==============================================================================
def test_understand_query_with_history():
    """Verify understand_query incorporates recent chat history and strips quotes from response."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content='"Apollo 11 spacecraft lunar landing mission launch year"')

    history = [
        HumanMessage(content="Tell me about Apollo 11 mission."),
        AIMessage(content="Apollo 11 was the American spaceflight that first landed humans on the Moon."),
    ]

    question = "When was it launched?"
    result_query = understand_query(question, mock_llm, chat_history=history)

    # 1. Output string should have surrounding quotes stripped
    assert result_query == "Apollo 11 spacecraft lunar landing mission launch year"

    # 2. Verify messages sent to LLM included conversation history context
    call_args = mock_llm.invoke.call_args[0][0]
    human_msg = [m for m in call_args if isinstance(m, HumanMessage)][0]
    assert "Recent Conversation Context:" in human_msg.content
    assert "Apollo 11" in human_msg.content
    assert "Current Question: When was it launched?" in human_msg.content


# ==============================================================================
# TEST 6: Evidence Verification Structured Output & Empty Context
# ==============================================================================
def test_verify_evidence():
    """Verify evidence verification fast-path on empty documents and structured LLM parsing."""
    mock_llm = MagicMock()

    # Case A: Empty documents list -> fast-path returns False immediately without invoking LLM
    empty_result = verify_evidence("What is the refund policy?", [], mock_llm)
    assert empty_result.enough_evidence is False
    assert "No documents were retrieved" in empty_result.reason
    assert mock_llm.with_structured_output.call_count == 0

    # Case B: Documents provided -> calls structured LLM
    docs = [
        Document(
            page_content="Customers can request a refund within 30 days of initial purchase.",
            metadata={"source": "policy.pdf", "page": 5},
        )
    ]
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = EvidenceVerification(
        enough_evidence=True,
        reason="Context contains explicit 30-day refund window.",
    )
    mock_llm.with_structured_output.return_value = mock_structured

    verification = verify_evidence("What is the refund policy?", docs, mock_llm)
    assert verification.enough_evidence is True
    assert "30-day refund window" in verification.reason

    # Case C: Exception during verification -> graceful fallback with enough_evidence=False
    mock_structured.invoke.side_effect = Exception("Model service error")
    failed_verification = verify_evidence("What is the refund policy?", docs, mock_llm)
    assert failed_verification.enough_evidence is False
    assert "encountered an error" in failed_verification.reason


# ==============================================================================
# TEST 7: Source Formatting and Grounded Answer Generation
# ==============================================================================
def test_generation_and_source_formatting():
    """Verify citation deduplication with page metadata and grounded answer confidence bounds."""
    docs = [
        Document(page_content="Section 1 details.", metadata={"source": "handbook.pdf", "page": 10}),
        Document(page_content="Section 2 details.", metadata={"source": "handbook.pdf", "page": 10}),  # duplicate citation
        Document(page_content="Overview notes.", metadata={"source": "notes.txt"}),  # no page
    ]

    # Test source formatting and deduplication
    sources = format_sources(docs)
    assert len(sources) == 2
    assert sources[0] == "handbook.pdf, page 10"
    assert sources[1] == "notes.txt"

    # Test generate_answer with confidence clipping
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    # Mock LLM output that sets an unclipped confidence (> 1.0) and empty warning
    mock_structured.invoke.return_value = AnswerOutput(
        answer="Eligible within 30 days.",
        confidence=1.45,
        warning="",
    )
    mock_llm.with_structured_output.return_value = mock_structured

    answer_out = generate_answer("What is the return period?", docs, mock_llm)
    assert answer_out.answer == "Eligible within 30 days."
    # Confidence must be clipped to <= 1.0
    assert answer_out.confidence == 1.0
    # Warning must be assigned a non-empty risk default
    assert "Low risk" in answer_out.warning


# ==============================================================================
# TEST 8: Document Chunking and Metadata Preservation
# ==============================================================================
def test_chunk_documents_metadata_preservation():
    """Verify document chunking splits large text while preserving source and page metadata."""
    sample_text = (
        "Artificial Intelligence (AI) is transforming software development.\n\n"
        "Machine Learning algorithms learn patterns from extensive data.\n\n"
        "Natural Language Processing enables computers to understand human language.\n\n"
        "Retrieval-Augmented Generation grounds generative language models on authoritative external documents.\n\n"
        "Vector databases index dense embeddings for low-latency semantic search."
    )
    original_doc = Document(
        page_content=sample_text,
        metadata={"source": "ai_overview.pdf", "page": 3, "author": "Research Team"},
    )

    # Chunk with small chunk_size to create multiple chunks
    chunks = chunk_documents([original_doc], chunk_size=120, chunk_overlap=30)

    assert len(chunks) > 1
    for chunk in chunks:
        # All chunks must retain the parent document metadata
        assert chunk.metadata["source"] == "ai_overview.pdf"
        assert chunk.metadata["page"] == 3
        assert chunk.metadata["author"] == "Research Team"
        assert len(chunk.page_content.strip()) > 0


# ==============================================================================
# TEST 9: Database Engine Lifecycle and Conversation History Persistence
# ==============================================================================
def test_database_crud_lifecycle():
    """Verify database initialization, conversation creation, turn persistence, and history retrieval."""
    # Use isolated in-memory SQLite database
    db_ok = database.init_db("sqlite://")
    assert db_ok is True
    assert database.is_db_connected() is True

    try:
        session_id = "test-session-uuid-001"
        created_sid = database.create_conversation(session_id=session_id, title="Unit Test Session")
        assert created_sid == session_id

        # Persist a conversation turn
        save_success = database.save_conversation_turn(
            session_id=session_id,
            user_query="What is CRAG?",
            search_query="Corrective RAG architecture",
            answer="Corrective RAG verifies evidence before generating answers.",
            sources=["paper.pdf, page 1"],
            enough_evidence=True,
            verification_reason="Found architecture section",
            confidence=0.95,
            warning="Low risk",
            iterations=1,
        )
        assert save_success is True

        # Retrieve history and verify persisted fields
        history = database.get_conversation_history(session_id)
        assert len(history) == 1
        turn = history[0]
        assert turn["user_query"] == "What is CRAG?"
        assert turn["search_query"] == "Corrective RAG architecture"
        assert turn["enough_evidence"] is True
        assert turn["confidence"] == 0.95
        assert turn["sources"] == ["paper.pdf, page 1"]

        # Verify conversation listing
        conv_list = database.list_conversations(limit=5)
        assert len(conv_list) >= 1
        assert any(c["session_id"] == session_id for c in conv_list)

    finally:
        database.close_db()
        assert database.is_db_connected() is False


# ==============================================================================
# TEST 10: LangGraph State Machine Execution Flow
# ==============================================================================
def test_rag_graph_execution_flows():
    """Verify compiled LangGraph state machine handles both direct chat and full RAG retrieval flows."""
    docs = [
        Document(
            page_content="Project Titan budget is 1.5 million dollars for fiscal year 2025.",
            metadata={"source": "budget.docx", "page": 1},
        )
    ]
    bm25 = BM25Index(docs)
    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search.return_value = docs

    # Scenario A: Conversational greeting -> routes to direct_chat_node without retrieval
    mock_llm_conversational = MagicMock()
    mock_llm_conversational.invoke.return_value = AIMessage(content="Hello! How can I assist you with your documents?")
    graph_chat = build_rag_graph(mock_vectorstore, bm25, mock_llm_conversational)

    chat_res = graph_chat.invoke({"question": "Good morning!"})
    assert "Hello!" in chat_res["answer"]
    assert chat_res["confidence"] == 1.0
    assert chat_res["sources"] == []
    # Vectorstore similarity search should NOT be called for direct chat
    assert mock_vectorstore.similarity_search.call_count == 0

    # Scenario B: Factual question -> routes through full RAG pipeline
    mock_llm_rag = MagicMock()
    # LLM invoke returns 'YES' for router, then search query string for understand_query
    mock_llm_rag.invoke.side_effect = [
        AIMessage(content="YES"),
        AIMessage(content="Project Titan budget fiscal year 2025"),
    ]
    mock_structured = MagicMock()
    mock_llm_rag.with_structured_output.side_effect = lambda schema: mock_structured
    mock_structured.invoke.side_effect = [
        EvidenceVerification(enough_evidence=True, reason="Direct budget figures found in doc."),
        AnswerOutput(
            answer="The budget for Project Titan is $1.5M for FY 2025.",
            confidence=0.98,
            warning="Low risk — answer is directly supported by retrieved documents.",
        ),
    ]

    graph_rag = build_rag_graph(mock_vectorstore, bm25, mock_llm_rag)
    rag_res = graph_rag.invoke({"question": "What is the budget for Project Titan?"})

    assert rag_res["enough_evidence"] is True
    assert "$1.5M" in rag_res["answer"]
    assert rag_res["confidence"] == 0.98
    assert "budget.docx, page 1" in rag_res["sources"]
    assert rag_res["search_query"] == "Project Titan budget fiscal year 2025"
    assert mock_vectorstore.similarity_search.call_count == 1
