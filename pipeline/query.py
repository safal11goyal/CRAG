from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage


def _extract_text(content) -> str:
    """Extract plain text from LLM response content."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
        return "".join(text_parts).strip()
    return str(content).strip()


def understand_query(question: str, llm: ChatGoogleGenerativeAI) -> str:
    """
    Transform the user's natural language question into a concise,
    keyword-dense search query for hybrid retrieval.
    """
    messages = [
        SystemMessage(
            content=(
                "You are a search query optimizer for a RAG system.\n"
                "Transform the user's input question into a concise, keyword-focused search query.\n"
                "Rules:\n"
                "- Do NOT answer the question.\n"
                "- Do NOT include explanations, markdown, or punctuation wrappers.\n"
                "- Output ONLY the plain search query string."
            )
        ),
        HumanMessage(content=f"Question: {question}"),
    ]
    response = llm.invoke(messages)
    query = _extract_text(response.content).strip('"').strip("'")
    return query if query else question


def improve_query(
    question: str,
    previous_query: str,
    reason: str,
    llm: ChatGoogleGenerativeAI,
) -> str:
    """
    Refine and improve the search query when prior evidence verification fails.
    Preserves original intent, introduces relevant technical keywords/synonyms,
    and removes noise words.
    """
    messages = [
        SystemMessage(
            content=(
                "You are an expert query refinement optimizer for a RAG system.\n"
                "The previous search query failed to retrieve sufficient evidence to answer the question.\n"
                "Refine the query by:\n"
                "1. Preserving the user's core intent.\n"
                "2. Adding technical synonyms, related terms, acronyms, or specific concepts.\n"
                "3. Removing generic or conversational filler words.\n"
                "Rules:\n"
                "- Do NOT answer the question.\n"
                "- Do NOT include explanations or punctuation wrappers.\n"
                "- Output ONLY the refined search query string."
            )
        ),
        HumanMessage(
            content=(
                f"Original Question: {question}\n"
                f"Previous Query: {previous_query}\n"
                f"Verification Failure Reason: {reason}\n"
                "Generate improved search query:"
            )
        ),
    ]
    response = llm.invoke(messages)
    refined = _extract_text(response.content).strip('"').strip("'")
    return refined if refined else previous_query
