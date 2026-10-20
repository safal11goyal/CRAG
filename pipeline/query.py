from typing import List, Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

# Shared parser instance — StrOutputParser handles str, AIMessage, and
# AIMessage with list-of-dict content blocks (Gemini provider format).
_str_parser = StrOutputParser()


def _parse_llm_response(response) -> str:
    """Extract plain text from an LLM response object using StrOutputParser.

    StrOutputParser correctly handles all response formats from the configured
    LLM providers (Gemini string content and list-of-dict content blocks).
    The result is stripped of leading/trailing whitespace.
    """
    return _str_parser.invoke(response).strip()


def understand_query(
    question: str,
    llm: ChatGoogleGenerativeAI,
    chat_history: Optional[List[BaseMessage]] = None,
) -> str:
    """
    Transform the user's natural language question into a concise,
    keyword-dense search query for hybrid retrieval, resolving any coreferences
    from recent chat history.
    """
    history_context = ""
    if chat_history and len(chat_history) > 1:
        # Take prior turns before current question
        recent = chat_history[-5:-1]
        if recent:
            history_lines = []
            for m in recent:
                role = "User" if isinstance(m, HumanMessage) else "Assistant"
                history_lines.append(f"{role}: {m.content}")
            history_context = "Recent Conversation Context:\n" + "\n".join(history_lines) + "\n\n"

    messages = [
        SystemMessage(
            content=(
                "You are a search query optimizer for a RAG system.\n"
                "Transform the user's input question into a concise, keyword-focused search query.\n"
                "Resolve any coreferences or pronouns (e.g., 'it', 'they', 'that') using the recent conversation context if relevant.\n"
                "Rules:\n"
                "- Do NOT answer the question.\n"
                "- Do NOT include explanations, markdown, or punctuation wrappers.\n"
                "- Output ONLY the plain search query string."
            )
        ),
        HumanMessage(content=f"{history_context}Current Question: {question}"),
    ]
    response = llm.invoke(messages)
    query = _parse_llm_response(response).strip('"').strip("'")
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
    refined = _parse_llm_response(response).strip('"').strip("'")
    return refined if refined else previous_query


def check_retrieval_needed(
    question: str,
    llm: ChatGoogleGenerativeAI,
    chat_history: Optional[List[BaseMessage]] = None,
) -> bool:
    """
    Determine if user input requires document retrieval (factual, specific questions from documents)
    or is conversational (greetings, pleasantries, small talk, questions about the user or ongoing conversation).

    Returns:
        True if retrieval is required, False if it can be answered directly.
    """
    cleaned = question.strip().lower().rstrip("!?.,")

    # Fast path for common conversational greetings and pleasantries (0ms latency)
    common_greetings = {
        "hi", "hello", "hey", "hey there", "greetings", "good morning",
        "good afternoon", "good evening", "how are you", "how are you doing",
        "how's it going", "who are you", "what can you do", "what are you",
        "help", "thanks", "thank you", "thank you very much", "bye", "goodbye", "see you", "ok",
        "okay", "cool", "nice", "great", "welcome",
    }
    if cleaned in common_greetings:
        return False

    # Check for very short 1-word greetings
    words = cleaned.split()
    if len(words) == 1 and words[0] in {"hi", "hello", "hey", "hola", "yo", "sup", "thanks", "thx"}:
        return False

    # Fast path for personal identity, introductions, and chat memory queries
    conversational_phrases = {
        "what is my name", "what is me name", "what's my name", "who am i",
        "do you know my name", "do you remember my name", "tell me my name",
        "what did i say", "what did i ask", "what was my last question",
        "my name is", "my name i", "i am", "call me",
    }
    if any(cleaned == phrase or cleaned.startswith(phrase + " ") for phrase in conversational_phrases):
        return False

    history_context = ""
    if chat_history and len(chat_history) > 1:
        recent = chat_history[-5:-1]
        if recent:
            history_lines = []
            for m in recent:
                role = "User" if isinstance(m, HumanMessage) else "Assistant"
                history_lines.append(f"{role}: {m.content}")
            history_context = "Recent Conversation History:\n" + "\n".join(history_lines) + "\n\n"

    messages = [
        SystemMessage(
            content=(
                "You are an intent router for a Retrieval-Augmented Generation (RAG) assistant.\n"
                "Analyze the user's latest input and decide whether answering it requires document retrieval from a knowledge base, or if it is purely conversational or refers to the ongoing conversation/user identity.\n\n"
                "Criteria:\n"
                "- Output 'NO' if the input is a greeting, polite pleasantry, small talk, expression of gratitude, farewell, asking what you can do, or asking about the user/chat history (e.g., 'what is my name', 'who am I', 'what did I say earlier').\n"
                "- Output 'YES' if the input asks for specific facts, explanations, data, policies, summaries, or questions from documents.\n\n"
                "Rules:\n"
                "- Output ONLY 'YES' or 'NO' with no extra words or punctuation."
            )
        ),
        HumanMessage(content=f"{history_context}User input: {question}"),
    ]
    try:
        response = llm.invoke(messages)
        decision = _parse_llm_response(response).upper()
        return "NO" not in decision
    except Exception:
        # Safe fallback: if router encounters an error, proceed with retrieval
        return True
