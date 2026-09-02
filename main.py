import os
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import CohereEmbeddings
import config
from pipeline.ingestion import load_or_create_vectorstore
from pipeline.retrieval import BM25Index
from pipeline.graph import build_rag_graph


def main():
    config.validate_config()

    if not os.path.exists(config.DOCUMENTS_DIR) or not os.listdir(config.DOCUMENTS_DIR):
        print(f"[ERROR] No documents found in '{config.DOCUMENTS_DIR}' directory.")
        print("Please place PDF, TXT, or DOCX files into the 'documents/' folder.")
        sys.exit(1)

    print("=" * 60)
    print("      Modular Corrective/Adaptive RAG Chatbot (Terminal)      ")
    print("=" * 60)
    print(f"LLM Model:        {config.GEMINI_MODEL}")
    print(f"Embedding Model:  {config.EMBEDDING_MODEL} (Cohere)")
    print(f"Documents Folder: {config.DOCUMENTS_DIR}")
    print(f"Vector Store:     {config.VECTORSTORE_DIR}")
    print("=" * 60)

    # Initialize Cohere embeddings
    print(f"[INFO] Initializing Cohere embedding model '{config.EMBEDDING_MODEL}'...")
    embeddings = CohereEmbeddings(
        model=config.EMBEDDING_MODEL,
        cohere_api_key=config.COHERE_API_KEY,
    )

    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
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

    print("\nSystem ready! Type your question below (type 'exit' or 'quit' to exit).\n")

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

        initial_state = {
            "question": user_input,
            "search_query": user_input,
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

        except Exception as e:
            print(f"\n[ERROR] An error occurred while running the pipeline: {e}\n")


if __name__ == "__main__":
    main()
