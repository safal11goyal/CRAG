import os
import time
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception


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
def _embed_batch(vectorstore, batch: List[Document], embeddings: Embeddings):
    """Embed a single batch of document chunks, retrying on rate-limit errors."""
    if vectorstore is None:
        return FAISS.from_documents(batch, embeddings)
    vectorstore.add_documents(batch)
    return vectorstore


def load_documents(documents_dir: str) -> List[Document]:
    """Load PDF, TXT, and DOCX files from the given directory."""
    documents: List[Document] = []

    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir, exist_ok=True)
        return documents

    for filename in sorted(os.listdir(documents_dir)):
        filepath = os.path.join(documents_dir, filename)
        if os.path.isdir(filepath):
            continue

        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

        if ext == "pdf":
            try:
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                    # PyPDFLoader uses 0-based page index; normalize to 1-based to match prior behavior
                    if "page" in doc.metadata:
                        doc.metadata["page"] = doc.metadata["page"] + 1
                    if doc.page_content.strip():
                        documents.append(doc)
            except Exception as e:
                print(f"[WARN] Failed to load PDF {filename}: {e}", flush=True)

        elif ext == "txt":
            try:
                loader = TextLoader(filepath, encoding="utf-8", autodetect_encoding=True)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                    if doc.page_content.strip():
                        documents.append(doc)
            except Exception as e:
                print(f"[WARN] Failed to load TXT {filename}: {e}", flush=True)

        elif ext == "docx":
            try:
                loader = Docx2txtLoader(filepath)
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
    documents_dir: str,
    vectorstore_dir: str,
    embeddings: Embeddings,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    batch_size: int = 30,
) -> Tuple[FAISS, List[Document]]:
    """Load existing FAISS index or create and persist a new one in rate-limited batches with retries."""
    raw_docs = load_documents(documents_dir)
    chunks = chunk_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    index_file = os.path.join(vectorstore_dir, "index.faiss")
    if os.path.exists(index_file):
        print(f"[INFO] Loading existing FAISS index from '{vectorstore_dir}'...", flush=True)
        vectorstore = FAISS.load_local(
            vectorstore_dir, embeddings, allow_dangerous_deserialization=True
        )
        return vectorstore, chunks

    print(f"[INFO] Creating new FAISS index from {len(chunks)} chunks in batches of {batch_size}...", flush=True)
    if not chunks:
        raise ValueError(
            f"No valid documents found in '{documents_dir}'. Please add PDF/TXT/DOCX files."
        )

    os.makedirs(vectorstore_dir, exist_ok=True)
    vectorstore = None
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for idx, i in enumerate(range(0, len(chunks), batch_size), start=1):
        batch = chunks[i : i + batch_size]
        print(f"[INFO] Embedding batch {idx}/{total_batches} ({len(batch)} chunks)...", flush=True)
        vectorstore = _embed_batch(vectorstore, batch, embeddings)

        # Courtesy pause between batches to stay under Cohere's 100k TPM trial limit
        time.sleep(3)

    if vectorstore is not None:
        vectorstore.save_local(vectorstore_dir)
        print(f"[INFO] FAISS index successfully created and saved to '{vectorstore_dir}'.", flush=True)

    return vectorstore, chunks
