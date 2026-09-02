from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from pipeline.retrieval import BM25Index, hybrid_retrieval
from pipeline.query import understand_query, improve_query
from pipeline.verification import verify_evidence
from pipeline.generation import generate_answer, format_sources
from pipeline.web_search import tavily_search
import config


class RAGState(TypedDict):
    question: str
    search_query: str
    documents: List[Document]
    enough_evidence: bool
    verification_reason: str
    answer: str
    confidence: float
    warning: str
    sources: List[str]
    iterations: int


def build_rag_graph(
    vectorstore: FAISS,
    bm25_index: BM25Index,
    llm: ChatGoogleGenerativeAI,
    k: int = config.RETRIEVAL_K,
    max_iterations: int = config.MAX_ITERATIONS,
):
    """Build and compile the Corrective/Adaptive RAG LangGraph workflow with Tavily fallback."""

    def understand_query_node(state: RAGState) -> dict:
        query = understand_query(state["question"], llm)
        return {
            "search_query": query,
            "iterations": state.get("iterations", 0),
        }

    def retrieve_node(state: RAGState) -> dict:
        query = state.get("search_query", state["question"])
        docs = hybrid_retrieval(vectorstore, bm25_index, query, k=k)
        current_iter = state.get("iterations", 0) + 1
        return {
            "documents": docs,
            "iterations": current_iter,
        }

    def verify_evidence_node(state: RAGState) -> dict:
        docs = state.get("documents", [])
        verification = verify_evidence(state["question"], docs, llm)
        return {
            "enough_evidence": verification.enough_evidence,
            "verification_reason": verification.reason,
        }

    def improve_query_node(state: RAGState) -> dict:
        refined_query = improve_query(
            question=state["question"],
            previous_query=state.get("search_query", state["question"]),
            reason=state.get("verification_reason", "Insufficient supporting evidence"),
            llm=llm,
        )
        return {
            "search_query": refined_query,
        }

    def web_search_node(state: RAGState) -> dict:
        query = state.get("search_query", state["question"])
        web_docs = tavily_search(query)
        return {
            "documents": web_docs,
        }

    def generate_answer_node(state: RAGState) -> dict:
        docs = state.get("documents", [])
        output = generate_answer(state["question"], docs, llm)
        sources = format_sources(docs)

        warning = output.warning
        # If answering from web search fallback, clarify in warning
        if any(doc.metadata.get("is_web") for doc in docs):
            warning = "Notice — answer was retrieved via Tavily Web Search fallback because local documents lacked sufficient coverage."

        return {
            "answer": output.answer,
            "confidence": output.confidence,
            "warning": warning,
            "sources": sources,
        }

    def handle_insufficient_evidence_node(state: RAGState) -> dict:
        docs = state.get("documents", [])
        sources = format_sources(docs) if docs else []
        return {
            "answer": "I could not find sufficient evidence in the provided documents or web search to answer this question reliably.",
            "confidence": 0.1,
            "warning": "High risk — insufficient supporting evidence was found after maximum retrieval attempts and web search.",
            "sources": sources,
        }

    def decide_next_step(state: RAGState) -> str:
        if state.get("enough_evidence", False):
            return "generate_answer"
        if state.get("iterations", 0) < max_iterations:
            return "improve_query"
        return "web_search"

    def decide_after_web_search(state: RAGState) -> str:
        docs = state.get("documents", [])
        if docs:
            return "generate_answer"
        return "handle_insufficient_evidence"

    workflow = StateGraph(RAGState)

    workflow.add_node("understand_query", understand_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("verify_evidence", verify_evidence_node)
    workflow.add_node("improve_query", improve_query_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("handle_insufficient_evidence", handle_insufficient_evidence_node)

    workflow.add_edge(START, "understand_query")
    workflow.add_edge("understand_query", "retrieve")
    workflow.add_edge("retrieve", "verify_evidence")

    workflow.add_conditional_edges(
        "verify_evidence",
        decide_next_step,
        {
            "generate_answer": "generate_answer",
            "improve_query": "improve_query",
            "web_search": "web_search",
        },
    )

    workflow.add_edge("improve_query", "retrieve")

    workflow.add_conditional_edges(
        "web_search",
        decide_after_web_search,
        {
            "generate_answer": "generate_answer",
            "handle_insufficient_evidence": "handle_insufficient_evidence",
        },
    )

    workflow.add_edge("generate_answer", END)
    workflow.add_edge("handle_insufficient_evidence", END)

    return workflow.compile()
