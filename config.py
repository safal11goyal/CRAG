import sys
import logging
import warnings
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from sqlalchemy.engine import make_url

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

_ENV_PATH = Path(__file__).resolve().parent / ".env"


class _Settings(BaseSettings):
    """
    Typed configuration loaded from environment variables and .env file.
    pydantic-settings handles type coercion, validation, and secret masking.
    """

    # API Keys
    GOOGLE_API_KEY: str = ""
    COHERE_API_KEY: str = ""
    TAVILY_API_KEY: str = ""
    PINECONE_API_KEY: str = ""

    # Model names
    GEMINI_MODEL: str = "gemini-3.1-flash-lite"
    EMBEDDING_MODEL: str = "embed-english-v3.0"

    # Chunking & Retrieval (integers auto-coerced by pydantic-settings)
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    RETRIEVAL_K: int = 4
    MAX_ITERATIONS: int = 3

    # Directories
    DOCUMENTS_DIR: str = "documents"
    VECTORSTORE_DIR: str = "vectorstore"

    # Pinecone
    PINECONE_INDEX_NAME: str = "rag-index"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    # PostgreSQL (full URL takes priority; individual vars are the fallback)
    DATABASE_URL: str = ""
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = "rag_db"

    model_config = {
        "env_file": str(_ENV_PATH),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton settings object
_settings = _Settings()

# ─── Module-level compatibility constants ──────────────────────────────────────
# All callers (app.py, main.py, database.py, pipeline/*) import these names
# directly, so we expose them as top-level module attributes exactly as before.
GOOGLE_API_KEY: str = _settings.GOOGLE_API_KEY
COHERE_API_KEY: str = _settings.COHERE_API_KEY
TAVILY_API_KEY: str = _settings.TAVILY_API_KEY
PINECONE_API_KEY: str = _settings.PINECONE_API_KEY

GEMINI_MODEL: str = _settings.GEMINI_MODEL
EMBEDDING_MODEL: str = _settings.EMBEDDING_MODEL

CHUNK_SIZE: int = _settings.CHUNK_SIZE
CHUNK_OVERLAP: int = _settings.CHUNK_OVERLAP
RETRIEVAL_K: int = _settings.RETRIEVAL_K
MAX_ITERATIONS: int = _settings.MAX_ITERATIONS

DOCUMENTS_DIR: str = _settings.DOCUMENTS_DIR
VECTORSTORE_DIR: str = _settings.VECTORSTORE_DIR

PINECONE_INDEX_NAME: str = _settings.PINECONE_INDEX_NAME
PINECONE_CLOUD: str = _settings.PINECONE_CLOUD
PINECONE_REGION: str = _settings.PINECONE_REGION

DATABASE_URL: str = _settings.DATABASE_URL
POSTGRES_HOST: str = _settings.POSTGRES_HOST
POSTGRES_PORT: str = _settings.POSTGRES_PORT
POSTGRES_USER: str = _settings.POSTGRES_USER
POSTGRES_PASSWORD: str = _settings.POSTGRES_PASSWORD
POSTGRES_DB: str = _settings.POSTGRES_DB


def get_database_url() -> str:
    """
    Return a normalized SQLAlchemy-compatible PostgreSQL connection URL.
    Uses SQLAlchemy's make_url to safely handle URL-encoded credentials and
    special characters in passwords without brittle string slicing.
    """
    raw = DATABASE_URL.strip()
    if raw:
        parsed = make_url(raw)
        # Normalize driver to postgresql+psycopg2 if it's a bare postgresql:// or postgres://
        if parsed.drivername in ("postgresql", "postgres"):
            parsed = parsed.set(drivername="postgresql+psycopg2")
        # Already normalized (postgresql+psycopg2://) → return as-is
        return parsed.render_as_string(hide_password=False)

    # Assemble from individual environment variables if provided
    if POSTGRES_USER and POSTGRES_PASSWORD:
        return (
            f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
            f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
        )

    return ""


def validate_config() -> None:
    """Validate critical environment configurations and exit on fatal errors."""
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_api_key_here":
        print("[ERROR] GOOGLE_API_KEY is not set or is a placeholder in .env.")
        print("Please configure your GOOGLE_API_KEY in .env before running.")
        sys.exit(1)

    if not COHERE_API_KEY or COHERE_API_KEY == "your_cohere_api_key_here":
        print("[ERROR] COHERE_API_KEY is not set or is a placeholder in .env.")
        print("Please configure your COHERE_API_KEY in .env before running.")
        sys.exit(1)

    if not PINECONE_API_KEY or PINECONE_API_KEY == "your_pinecone_api_key_here":
        print("[ERROR] PINECONE_API_KEY is not set or is a placeholder in .env.")
        print("Please configure your PINECONE_API_KEY in .env before running.")
        sys.exit(1)

    if not TAVILY_API_KEY or TAVILY_API_KEY == "your_tavily_api_key_here":
        print("[WARN] TAVILY_API_KEY is not set. Web search fallback will be disabled.")

    db_url = get_database_url()
    if not db_url:
        print("[WARN] DATABASE_URL is not configured in .env. Conversation persistence will run in offline mode.")
