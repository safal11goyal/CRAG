import os
import sys
import uuid
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import CohereEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import config
import database
from pipeline.ingestion import load_or_create_vectorstore
from pipeline.retrieval import BM25Index
from pipeline.graph import build_rag_graph


def main():
    config.validate_config()

    if not os.path.exists(config.DOCUMENTS_DIR) or not os.listdir(config.DOCUMENTS_DIR):
        print(f"[ERROR] No documents found in '{config.DOCUMENTS_DIR}' directory.")
        print("Please place PDF, TXT, or DOCX files into the 'documents/' folder.")
        sys.exit(1)

    # Initialize PostgreSQL Database
    db_connected = database.init_db()

    # Generate a unique session ID for this conversation run
    session_id = str(uuid.uuid4())
    if db_connected:
        database.create_conversation(session_id=session_id, title="CLI Chat Session")

    print("=" * 60)
    print("      Modular Corrective/Adaptive RAG Chatbot (Terminal)      ")
    print("=" * 60)
    print(f"LLM Model:        {config.GEMINI_MODEL}")
    print(f"Embedding Model:  {config.EMBEDDING_MODEL} (Cohere)")
    print(f"Documents Folder: {config.DOCUMENTS_DIR}")
    print(f"Vector Store:     {config.VECTORSTORE_DIR}")
    print(f"PostgreSQL DB:    {'Connected' if db_connected else 'Offline / Disabled'}")
    print(f"Session ID:       {session_id}")
    print("=" * 60)

    # Initialize Cohere embeddings
    print(f"[INFO] Initializing Cohere embedding model '{config.EMBEDDING_MODEL}'...")
    embeddings = CohereEmbeddings(
        model=config.EMBEDDING_MODEL,
        cohere_api_key=config.COHERE_API_KEY,
        max_retries=3,
        request_timeout=60,
    )

    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        max_retries=3,
        timeout=60,
    )

    # Ingestion / Load Vectorstore & BM25 Index
    try:
        vectorstore, chunks = load_or_create_vectorstore(
            documents_dir=config.DOCUMENTS_DIR,
            vectorstore_dir=config.VECTORSTORE_DIR,
            embeddings=embeddings,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
    except Exception as e:
        print(f"[ERROR] Failed during document ingestion / vectorstore setup: {e}")
        sys.exit(1)

    print("[INFO] Initializing BM25 keyword index...")
    bm25_index = BM25Index(chunks)

    print("[INFO] Compiling LangGraph workflow...")
    rag_app = build_rag_graph(
        vectorstore=vectorstore,
        bm25_index=bm25_index,
        llm=llm,
        k=config.RETRIEVAL_K,
        max_iterations=config.MAX_ITERATIONS,
    )

    print("\nSystem ready! Type your question below.")
    print("Commands: '/history' to view session history, '/sessions' to list sessions, 'exit' to quit.\n")

    chat_messages: list[BaseMessage] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye.")
            break

        # Handle history / session inspection commands
        if user_input.lower() == "/history":
            history = database.get_conversation_history(session_id)
            if not history:
                print("\n[INFO] No conversation history recorded yet in this session.\n")
            else:
                print(f"\n--- Conversation History ({len(history)} turn(s)) ---")
                for idx, turn in enumerate(history, 1):
                    print(f"[{idx}] User:   {turn['user_query']}")
                    print(f"    Answer: {turn['answer']}")
                    print(f"    Confidence: {turn['confidence']} | Evidence: {turn['enough_evidence']}")
                print("--------------------------------------------------\n")
            continue

        if user_input.lower() == "/sessions":
            sessions = database.list_conversations(limit=10)
            if not sessions:
                print("\n[INFO] No stored sessions found or database is offline.\n")
            else:
                print(f"\n--- Recent Sessions ({len(sessions)}) ---")
                for s in sessions:
                    active = " [Active]" if s["session_id"] == session_id else ""
                    print(f"- Session ID: {s['session_id']}{active}")
                    print(f"  Title: {s['title']} | Turns: {s['turns_count']} | Updated: {s['updated_at']}")
                print("----------------------------------------\n")
            continue

        chat_messages.append(HumanMessage(content=user_input))

        initial_state = {
            "messages": chat_messages,
            "question": user_input,
            "search_query": user_input,
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
            final_state = rag_app.invoke(initial_state)

            answer = final_state.get("answer", "No answer could be generated.")
            confidence = final_state.get("confidence", 0.0)
            sources = final_state.get("sources", [])
            warning = final_state.get("warning", "No warning information provided.")
            enough_evidence = final_state.get("enough_evidence", False)
            verification_reason = final_state.get("verification_reason", "")
            iterations = final_state.get("iterations", 0)

            chat_messages.append(AIMessage(content=answer))

            print("\nAnswer:")
            print(answer)

            print("\nSources:")
            if sources:
                for src in sources:
                    print(f"- {src}")
            else:
                print("- None")

            print("\nHallucination Warning:")
            print(warning)
            print("\n" + "-" * 60 + "\n")

            # Persist the turn to PostgreSQL database
            database.save_conversation_turn(
                session_id=session_id,
                user_query=user_input,
                search_query=final_state.get("search_query", user_input),
                answer=answer,
                sources=sources,
                enough_evidence=enough_evidence,
                verification_reason=verification_reason,
                confidence=confidence,
                warning=warning,
                iterations=iterations,
            )

        except Exception as e:
            print(f"\n[ERROR] An error occurred while running the pipeline: {e}\n")


if __name__ == "__main__":
    main()
