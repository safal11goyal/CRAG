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


def _extract_shape_text(shape) -> List[str]:
    """Recursively extract text from a PowerPoint slide shape, including tables and group shapes."""
    texts: List[str] = []
    if shape.has_text_frame:
        for paragraph in shape.text_frame.paragraphs:
            line = paragraph.text.strip()
            if line:
                texts.append(line)
    elif shape.has_table:
        for row in shape.table.rows:
            row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_cells:
                texts.append(" | ".join(row_cells))
    elif hasattr(shape, "shapes"):  # Grouped shapes
        for child_shape in shape.shapes:
            texts.extend(_extract_shape_text(child_shape))
    return texts


def load_pptx(filepath: Path) -> List[Document]:
    """Load slides from a PowerPoint presentation (.pptx) as LangChain Documents."""
    try:
        import pptx
    except ImportError as e:
        raise ImportError(
            "python-pptx is required to load PPTX files. Install it using: pip install python-pptx"
        ) from e

    prs = pptx.Presentation(str(filepath))
    documents: List[Document] = []
    filename = filepath.name

    for slide_idx, slide in enumerate(prs.slides, start=1):
        texts: List[str] = []
        for shape in slide.shapes:
            texts.extend(_extract_shape_text(shape))

        # Check speaker notes if available
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                texts.append(f"Speaker Notes: {notes}")

        slide_content = "\n".join(texts).strip()
        if slide_content:
            documents.append(
                Document(
                    page_content=slide_content,
                    metadata={
                        "source": filename,
                        "page": slide_idx,
                    },
                )
            )

    return documents


def load_single_document(filepath: Path | str) -> List[Document]:
    """Load a single PDF, PPTX, PPT, TXT, or DOCX document into LangChain Documents."""
    file_path = Path(filepath)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.suffix.lower().lstrip(".")
    filename = file_path.name
    documents: List[Document] = []

    if ext == "pdf":
        try:
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
                # PyPDFLoader uses 0-based page index; normalize to 1-based
                if "page" in doc.metadata:
                    doc.metadata["page"] = doc.metadata["page"] + 1
                if doc.page_content.strip():
                    documents.append(doc)
        except Exception as e:
            raise RuntimeError(f"Failed to load PDF file '{filename}': {e}") from e

    elif ext in ("pptx", "ppt"):
        try:
            docs = load_pptx(file_path)
            documents.extend(docs)
        except Exception as e:
            if ext == "ppt":
                raise ValueError(
                    f"Failed to load '{filename}'. Legacy binary .ppt format cannot be parsed directly. "
                    f"Please convert or save it as modern .pptx format."
                ) from e
            raise RuntimeError(f"Failed to load PowerPoint file '{filename}': {e}") from e

    elif ext == "txt":
        try:
            loader = TextLoader(str(file_path), encoding="utf-8", autodetect_encoding=True)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
                if doc.page_content.strip():
                    documents.append(doc)
        except Exception as e:
            raise RuntimeError(f"Failed to load TXT file '{filename}': {e}") from e

    elif ext == "docx":
        try:
            loader = Docx2txtLoader(str(file_path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
                if doc.page_content.strip():
                    documents.append(doc)
        except Exception as e:
            raise RuntimeError(f"Failed to load DOCX file '{filename}': {e}") from e

    else:
        raise ValueError(
            f"Unsupported file format '.{ext}'. Supported formats are: PDF, PPTX, PPT, TXT, DOCX"
        )

    return documents


def load_documents(documents_dir: str) -> List[Document]:
    """Load PDF, PPTX, TXT, and DOCX files from the given directory."""
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
        if ext not in ("pdf", "pptx", "ppt", "txt", "docx"):
            continue

        try:
            docs = load_single_document(filepath)
            documents.extend(docs)
        except Exception as e:
            print(f"[WARN] Failed to load {filepath.name}: {e}", flush=True)

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


def ingest_uploaded_file(
    file_bytes: bytes,
    filename: str,
    documents_dir: str = config.DOCUMENTS_DIR,
    embeddings: Optional[Embeddings] = None,
    index_name: Optional[str] = None,
    api_key: Optional[str] = None,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    batch_size: int = 30,
    status_callback: Optional[Any] = None,
) -> Tuple[int, int]:
    """
    Save an uploaded file to documents_dir, load and chunk it,
    then embed and upsert its chunks into Pinecone.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Name of the file including extension.
        documents_dir: Target directory where document is persisted.
        embeddings: Optional Embeddings instance (initialized if not supplied).
        index_name: Optional Pinecone index name.
        api_key: Optional Pinecone API key.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        batch_size: Number of chunks per Pinecone upsert batch.
        status_callback: Optional callable for live progress messaging (e.g. in Streamlit).

    Returns:
        Tuple of (num_pages_or_slides, num_chunks).
    """
    dest_dir = Path(documents_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    if status_callback:
        status_callback(f"Saving '{filename}' to documents library...")

    with open(dest_path, "wb") as f:
        f.write(file_bytes)

    if status_callback:
        status_callback(f"Parsing content and structure from '{filename}'...")

    docs = load_single_document(dest_path)
    if not docs:
        return 0, 0

    if status_callback:
        status_callback(f"Splitting into chunks (parsed {len(docs)} pages/slides)...")

    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return len(docs), 0

    if embeddings is None:
        if status_callback:
            status_callback("Initializing Cohere embedding model...")
        from langchain_cohere import CohereEmbeddings
        embeddings = CohereEmbeddings(
            model=config.EMBEDDING_MODEL,
            cohere_api_key=config.COHERE_API_KEY,
            max_retries=3,
            request_timeout=60,
        )

    target_index_name = index_name or config.PINECONE_INDEX_NAME
    target_api_key = api_key or config.PINECONE_API_KEY

    if not target_api_key or target_api_key == "your_pinecone_api_key_here":
        raise ValueError("PINECONE_API_KEY is not configured in .env.")

    if status_callback:
        status_callback("Connecting to Pinecone vector store...")

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

    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for idx, i in enumerate(range(0, len(chunks), batch_size), start=1):
        batch = chunks[i : i + batch_size]
        if status_callback:
            status_callback(f"Embedding & upserting batch {idx}/{total_batches} ({len(batch)} chunks)...")
        _embed_batch(vectorstore, batch, embeddings)
        if idx < total_batches:
            time.sleep(3)

    if status_callback:
        status_callback(f"Successfully indexed {len(docs)} pages/slides into {len(chunks)} chunks!")

    return len(docs), len(chunks)

