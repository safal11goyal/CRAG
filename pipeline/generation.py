from typing import List
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage


class AnswerOutput(BaseModel):
    answer: str = Field(
        description="The detailed answer based strictly and solely on the provided context."
    )
    confidence: float = Field(
        default=0.9,
        description="An evidence-support confidence score between 0.0 and 1.0 reflecting how directly the context supports the answer.",
    )
    warning: str = Field(
        default="Low risk — answer is directly supported by retrieved documents.",
        description="A concise hallucination risk assessment string.",
    )


def format_sources(documents: List[Document]) -> List[str]:
    """
    Extract and deduplicate source citations from document metadata.
    Preserves exact page numbers when present; never invents page numbers.
    """
    sources: List[str] = []
    seen = set()

    for doc in documents:
        source_name = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")
        if page is not None:
            citation = f"{source_name}, page {page}"
        else:
            citation = source_name

        if citation not in seen:
            seen.add(citation)
            sources.append(citation)

    return sources


def generate_answer(
    question: str,
    documents: List[Document],
    llm: ChatGoogleGenerativeAI,
) -> AnswerOutput:
    """Generate a grounded answer strictly from the retrieved document context."""
    formatted_context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')}"
        + (f", Page: {doc.metadata['page']}" if "page" in doc.metadata else "")
        + f"]\n{doc.page_content}"
        for doc in documents
    )

    structured_llm = llm.with_structured_output(AnswerOutput)

    messages = [
        SystemMessage(
            content=(
                "You are a precise RAG assistant. Answer the question using ONLY the provided context.\n\n"
                "Rules:\n"
                "1. If the context does not explicitly support a claim, do NOT make the claim.\n"
                "2. Do NOT use outside or prior knowledge.\n"
                "3. Provide a confidence score between 0.0 and 1.0 reflecting the factual evidence strength.\n"
                "4. In the warning field, always provide a concise risk assessment statement "
                "(e.g., 'Low risk — answer is directly supported by retrieved documents.'). Never leave it empty."
            )
        ),
        HumanMessage(
            content=(
                f"Question:\n{question}\n\n"
                f"Retrieved Context:\n{formatted_context}\n\n"
                "Generate the grounded response:"
            )
        ),
    ]

    try:
        result = structured_llm.invoke(messages)
        if isinstance(result, AnswerOutput):
            result.confidence = max(0.0, min(1.0, float(result.confidence)))
            if not result.warning or not result.warning.strip():
                result.warning = "Low risk — answer is directly supported by retrieved documents."
            return result
        if isinstance(result, dict):
            return AnswerOutput(
                answer=result.get("answer", ""),
                confidence=max(0.0, min(1.0, float(result.get("confidence", 0.85)))),
                warning=result.get("warning") or "Low risk — answer is supported by retrieved documents.",
            )
        return AnswerOutput(
            answer="Unable to parse structured response.",
            confidence=0.0,
            warning="High risk — answer could not be parsed.",
        )
    except Exception as e:
        return AnswerOutput(
            answer=f"Error generating answer: {e}",
            confidence=0.0,
            warning="High risk — generation encountered an error.",
        )
