import time
import hashlib
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pinecone import Pinecone, ServerlessSpec


def _clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure metadata values conform to Pinecone supported types:
    str, int, float, bool, or list of str.
    """
    cleaned: Dict[str, Any] = {}
    for k, v in metadata.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        elif isinstance(v, list) and all(isinstance(item, str) for item in v):
            cleaned[k] = v
        elif isinstance(v, (list, tuple)):
            cleaned[k] = [str(item) for item in v]
        else:
            cleaned[k] = str(v)
    return cleaned


def _generate_doc_id(doc: Document, fallback_idx: int) -> str:
    """Generate a deterministic hash ID for a document chunk to guarantee idempotent upserts."""
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page", "")
    content = doc.page_content.strip()
    raw_key = f"{source}_{page}_{content}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:32]


class PineconeVectorStore(VectorStore):
    """
    Production-grade VectorStore integration for Pinecone Database (v10 SDK).
    Supports dense vector similarity search, metadata filtering, and batch upserting.
    """

    def __init__(
        self,
        index: Any,
        embedding: Embeddings,
        text_key: str = "text",
        namespace: str = "",
    ):
        self._index = index
        self._embedding = embedding
        self._text_key = text_key
        self._namespace = namespace

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    @property
    def index(self) -> Any:
        return self._index

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 50,
        **kwargs: Any,
    ) -> List[str]:
        """Embed and upsert texts into Pinecone."""
        texts_list = list(texts)
        if not texts_list:
            return []

        embeddings = self._embedding.embed_documents(texts_list)
        generated_ids = ids if ids else [str(uuid.uuid4()) for _ in texts_list]
        metadatas = metadatas if metadatas else [{} for _ in texts_list]

        vectors = []
        for doc_id, text, emb, meta in zip(generated_ids, texts_list, embeddings, metadatas):
            clean_meta = _clean_metadata(meta)
            clean_meta[self._text_key] = text
            vectors.append({
                "id": str(doc_id),
                "values": emb,
                "metadata": clean_meta,
            })

        for i in range(0, len(vectors), batch_size):
            chunk = vectors[i : i + batch_size]
            self._index.upsert(vectors=chunk, namespace=self._namespace)

        return generated_ids

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        batch_size: int = 50,
        **kwargs: Any,
    ) -> List[str]:
        """Add LangChain Document instances to Pinecone with deterministic IDs."""
        if not documents:
            return []

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        doc_ids = ids or [_generate_doc_id(doc, i) for i, doc in enumerate(documents)]

        return self.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=doc_ids,
            batch_size=batch_size,
            **kwargs,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search Pinecone for documents similar to query and return with similarity score."""
        query_vector = self._embedding.embed_query(query)
        res = self._index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True,
            filter=filter,
            namespace=self._namespace,
        )

        results: List[Tuple[Document, float]] = []
        matches = getattr(res, "matches", []) or (res.get("matches", []) if isinstance(res, dict) else [])
        for match in matches:
            match_meta = getattr(match, "metadata", None)
            if match_meta is None and isinstance(match, dict):
                match_meta = match.get("metadata", {})
            meta = dict(match_meta or {})
            text = meta.pop(self._text_key, "")
            score = getattr(match, "score", 0.0) if hasattr(match, "score") else (match.get("score", 0.0) if isinstance(match, dict) else 0.0)
            doc = Document(page_content=text, metadata=meta)
            results.append((doc, float(score)))

        return results

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search Pinecone for documents similar to query."""
        docs_and_scores = self.similarity_search_with_score(query, k=k, filter=filter, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        index: Any = None,
        text_key: str = "text",
        namespace: str = "",
        batch_size: int = 50,
        **kwargs: Any,
    ) -> "PineconeVectorStore":
        store = cls(index=index, embedding=embedding, text_key=text_key, namespace=namespace)
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids, batch_size=batch_size, **kwargs)
        return store


def get_or_create_pinecone_index(
    api_key: str,
    index_name: str,
    dimension: int,
    cloud: str = "aws",
    region: str = "us-east-1",
) -> Any:
    """Connect to existing Pinecone index or create a new Serverless index if missing."""
    pc = Pinecone(api_key=api_key)

    if not pc.has_index(index_name):
        print(f"[INFO] Creating new Pinecone index '{index_name}' (dimension={dimension}, metric='cosine')...", flush=True)
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

        # Wait for index ready
        max_wait = 60
        waited = 0
        while waited < max_wait:
            desc = pc.describe_index(index_name)
            status = getattr(desc, "status", {})
            is_ready = getattr(status, "ready", False) if hasattr(status, "ready") else (status.get("ready", False) if isinstance(status, dict) else False)
            if is_ready:
                break
            time.sleep(2)
            waited += 2

    return pc.Index(index_name)
