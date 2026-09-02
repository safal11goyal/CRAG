import os
import sys
import logging
import warnings
from dotenv import load_dotenv

# Suppress internal SDK / library warnings in terminal
warnings.filterwarnings("ignore")
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

try:
    from google.genai.models import Models
    Models._logged_afc_warning = True
except Exception:
    pass

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embed-english-v3.0")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))

DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "documents")
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore")


def validate_config() -> None:
    """Validate critical environment configurations."""
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_api_key_here":
        print("[ERROR] GOOGLE_API_KEY is not set or is a placeholder in .env.")
        print("Please configure your GOOGLE_API_KEY in .env before running.")
        sys.exit(1)

    if not COHERE_API_KEY or COHERE_API_KEY == "your_cohere_api_key_here":
        print("[ERROR] COHERE_API_KEY is not set or is a placeholder in .env.")
        print("Please configure your COHERE_API_KEY in .env before running.")
        sys.exit(1)

    if not TAVILY_API_KEY or TAVILY_API_KEY == "your_tavily_api_key_here":
        print("[WARN] TAVILY_API_KEY is not set. Web search fallback will be disabled.")
