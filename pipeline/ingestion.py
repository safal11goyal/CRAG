import time
from pathlib import Path
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
import config
from pipeline.pinecone_store import PineconeVectorStore, get_or_create_pinecone_index


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if the exception is a transient API rate-limit error."""
    msg = str(exc).lower()
    return any(
        k in msg
        for k in ("429", "rate limit", "too many requests", "resource_exhausted", "trial token")
    )


@retry(
    retry=retry_if_exception(_is_rate_limit_error),
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=10, min=10, max=80),
    before_sleep=lambda rs: print(
        f"[WARN] Rate limit reached. Waiting before retrying batch (attempt {rs.attempt_number}/8)...",
        flush=True,
    ),
    reraise=True,
)
def _embed_batch(vectorstore: PineconeVectorStore, batch: List[Document], embeddings: Embeddings):
    """Embed and upsert a single batch of document chunks into Pinecone, retrying on rate-limit errors."""
    vectorstore.add_documents(batch)
    return vectorstore


def load_documents(documents_dir: str) -> List[Document]:
    """Load PDF, TXT, and DOCX files from the given directory."""
    documents: List[Document] = []
    dir_path = Path(documents_dir)

    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        return documents

    # sorted() for reproducible ordering across platforms
    for filepath in sorted(dir_path.iterdir()):
        if not filepath.is_file():
            continue

        ext = filepath.suffix.lower().lstrip(".")
        filename = filepath.name

        if ext == "pdf":
            try:
                loader = PyPDFLoader(str(filepath))
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                    # PyPDFLoader uses 0-based page index; normalize to 1-based
                    if "page" in doc.metadata:
                        doc.metadata["page"] = doc.metadata["page"] + 1
                    if doc.page_content.strip():
                        documents.append(doc)
            except Exception as e:
                print(f"[WARN] Failed to load PDF {filename}: {e}", flush=True)

        elif ext == "txt":
            try:
                loader = TextLoader(str(filepath), encoding="utf-8", autodetect_encoding=True)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                    if doc.page_content.strip():
                        documents.append(doc)
            except Exception as e:
                print(f"[WARN] Failed to load TXT {filename}: {e}", flush=True)

        elif ext == "docx":
            try:
                loader = Docx2txtLoader(str(filepath))
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                    if doc.page_content.strip():
                        documents.append(doc)
            except Exception as e:
                print(f"[WARN] Failed to load DOCX {filename}: {e}", flush=True)

    return documents


def chunk_documents(
    documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    """Split documents into chunks while preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def load_or_create_vectorstore(
    documents_dir: str = config.DOCUMENTS_DIR,
    vectorstore_dir: Optional[str] = None,
    embeddings: Optional[Embeddings] = None,
    index_name: Optional[str] = None,
    api_key: Optional[str] = None,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    batch_size: int = 30,
    force_reindex: bool = False,
) -> Tuple[PineconeVectorStore, List[Document]]:
    """
    Connect to existing Pinecone vector index or create and populate a new index.
    Also returns document chunks for the BM25 keyword index.
    """
    if embeddings is None:
        raise ValueError("An Embeddings model instance must be provided.")

    target_index_name = index_name or config.PINECONE_INDEX_NAME
    target_api_key = api_key or config.PINECONE_API_KEY

    raw_docs = load_documents(documents_dir)
    chunks = chunk_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not target_api_key or target_api_key == "your_pinecone_api_key_here":
        raise ValueError("PINECONE_API_KEY is not configured in .env. Please configure your Pinecone API key.")

    print(f"[INFO] Initializing Pinecone index '{target_index_name}'...", flush=True)
    probe_vector = embeddings.embed_query("dimension_probe")
    dimension = len(probe_vector)

    pinecone_index = get_or_create_pinecone_index(
        api_key=target_api_key,
        index_name=target_index_name,
        dimension=dimension,
        cloud=config.PINECONE_CLOUD,
        region=config.PINECONE_REGION,
    )

    vectorstore = PineconeVectorStore(index=pinecone_index, embedding=embeddings)

    # Check vector count in index
    stats = pinecone_index.describe_index_stats()
    total_vectors = (
        getattr(stats, "total_vector_count", 0)
        if hasattr(stats, "total_vector_count")
        else (stats.get("total_vector_count", 0) if isinstance(stats, dict) else 0)
    )

    if total_vectors > 0 and not force_reindex:
        print(
            f"[INFO] Connected to existing Pinecone index '{target_index_name}' with {total_vectors} vectors.",
            flush=True,
        )
        return vectorstore, chunks

    print(
        f"[INFO] Indexing {len(chunks)} chunks into Pinecone index '{target_index_name}' in batches of {batch_size}...",
        flush=True,
    )
    if not chunks:
        raise ValueError(
            f"No valid documents found in '{documents_dir}'. Please add PDF/TXT/DOCX files."
        )

    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for idx, i in enumerate(range(0, len(chunks), batch_size), start=1):
        batch = chunks[i : i + batch_size]
        print(f"[INFO] Embedding and upserting batch {idx}/{total_batches} ({len(batch)} chunks)...", flush=True)
        _embed_batch(vectorstore, batch, embeddings)

        # Courtesy pause between batches to stay under Cohere's 100k TPM trial limit
        time.sleep(3)

    print(f"[INFO] Pinecone index '{target_index_name}' successfully populated.", flush=True)
    return vectorstore, chunks
