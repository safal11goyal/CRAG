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

# Explicitly load .env from project root with override=True
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embed-english-v3.0")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE") or 1000)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP") or 200)
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K") or 4)
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS") or 3)

DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "documents")
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore")

# PostgreSQL Database Settings
DATABASE_URL = os.getenv("DATABASE_URL", "")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_db")


def get_database_url() -> str:
    """Return a normalized SQLAlchemy-compatible PostgreSQL connection URL."""
    url = DATABASE_URL.strip()
    if url:
        # Normalize standard postgresql:// scheme to postgresql+psycopg2:// for SQLAlchemy
        if url.startswith("postgresql://"):
            return "postgresql+psycopg2://" + url[len("postgresql://"):]
        elif url.startswith("postgres://"):
            return "postgresql+psycopg2://" + url[len("postgres://"):]
        return url

    # Assemble from individual environment variables if provided
    if POSTGRES_USER and POSTGRES_PASSWORD:
        return f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

    return ""


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

    db_url = get_database_url()
    if not db_url:
        print("[WARN] DATABASE_URL is not configured in .env. Conversation persistence will run in offline mode.")
