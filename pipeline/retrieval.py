import re
from typing import Dict, List
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class BM25Index:
    """Simple BM25 index for keyword search over document chunks.

    Uses `rank_bm25.BM25Okapi` directly so we can preserve the existing
    score > 0 filter (which `langchain_community.retrievers.BM25Retriever`
    does NOT apply — it always returns k results regardless of score).
    """

    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.tokenized_corpus = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def search(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve top-k documents matching the query keywords (score > 0 only)."""
        if not self.bm25 or not self.documents:
            return []
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[::-1][:k]
        return [self.documents[i] for i in top_indices if scores[i] > 0]


def _rrf_rank(
    vector_docs: List[Document],
    keyword_docs: List[Document],
    k: int = 4,
    rrf_k: int = 60,
) -> List[Document]:
    """
    Reciprocal Rank Fusion (RRF) over dense and sparse result lists.

    Each document receives a score:
        score(d) = sum(1 / (rrf_k + rank_in_list))  for each list it appears in

    Documents appearing in BOTH lists are promoted; the relative ordering of
    each individual list is respected.  Only the top-k results are returned.

    Args:
        vector_docs:  Ranked list from dense vector search (index 0 = rank 1).
        keyword_docs: Ranked list from BM25 sparse search (index 0 = rank 1).
        k:            Number of results to return.
        rrf_k:        RRF constant (60 is the standard value from the original paper).

    Returns:
        List[Document] of at most k results, ordered by descending RRF score.
    """
    # Compute a stable document fingerprint for deduplication
    def _doc_key(doc: Document) -> tuple:
        return (
            doc.page_content.strip(),
            doc.metadata.get("source"),
            doc.metadata.get("page"),
        )

    scores: Dict[tuple, float] = {}
    docs_by_key: Dict[tuple, Document] = {}

    for rank, doc in enumerate(vector_docs, start=1):
        key = _doc_key(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        docs_by_key[key] = doc

    for rank, doc in enumerate(keyword_docs, start=1):
        key = _doc_key(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        docs_by_key[key] = doc

    sorted_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [docs_by_key[key] for key in sorted_keys[:k]]


def hybrid_retrieval(
    vectorstore: VectorStore,
    bm25_index: BM25Index,
    query: str,
    k: int = 4,
) -> List[Document]:
    """
    Perform hybrid retrieval combining Pinecone dense vector search and BM25
    sparse keyword search, merged via Reciprocal Rank Fusion (RRF).

    RRF promotes documents that appear in both result sets and respects the
    relative ranking within each list, which is more reliable than naive
    list concatenation that unconditionally gives dense results priority.
    """
    vector_docs = vectorstore.similarity_search(query, k=k)
    keyword_docs = bm25_index.search(query, k=k)
    return _rrf_rank(vector_docs, keyword_docs, k=k)
