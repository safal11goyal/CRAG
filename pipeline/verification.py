from typing import List
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage


class EvidenceVerification(BaseModel):
    enough_evidence: bool = Field(
        description="True ONLY if the retrieved documents contain sufficient factual evidence to answer the question; False otherwise."
    )
    reason: str = Field(
        description="Concise justification explaining why the evidence is sufficient or insufficient."
    )


def verify_evidence(
    question: str,
    documents: List[Document],
    llm: ChatGoogleGenerativeAI,
) -> EvidenceVerification:
    """
    Conservatively evaluate whether retrieved documents contain enough evidence
    to answer the user's question accurately.
    """
    if not documents:
        return EvidenceVerification(
            enough_evidence=False,
            reason="No documents were retrieved.",
        )

    formatted_context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')}"
        + (f", Page: {doc.metadata['page']}" if "page" in doc.metadata else "")
        + f"]\n{doc.page_content}"
        for doc in documents
    )

    structured_llm = llm.with_structured_output(EvidenceVerification)

    messages = [
        SystemMessage(
            content=(
                "You are a conservative evidence verifier in a RAG system.\n"
                "Evaluate whether the retrieved context contains sufficient, factual evidence "
                "to answer the user's question accurately.\n\n"
                "Guidelines:\n"
                "- Be conservative. If the context merely touches on keywords or is incomplete, mark enough_evidence as false.\n"
                "- Only mark enough_evidence as true if the retrieved text contains the concrete facts needed to answer the question.\n"
                "- Provide a concise reason for your decision."
            )
        ),
        HumanMessage(
            content=(
                f"Question:\n{question}\n\n"
                f"Retrieved Context:\n{formatted_context}\n\n"
                "Perform evidence verification:"
            )
        ),
    ]

    try:
        result = structured_llm.invoke(messages)
        if isinstance(result, EvidenceVerification):
            return result
        if isinstance(result, dict):
            return EvidenceVerification(**result)
        return EvidenceVerification(
            enough_evidence=False,
            reason="Structured output could not be parsed.",
        )
    except Exception as e:
        return EvidenceVerification(
            enough_evidence=False,
            reason=f"Evidence verification encountered an error: {e}",
        )
