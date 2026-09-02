import re
from typing import List
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


class BM25Index:
    """Simple BM25 index for keyword search over document chunks."""

    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.tokenized_corpus = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def search(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve top-k documents matching the query keywords."""
        if not self.bm25 or not self.documents:
            return []
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[::-1][:k]
        return [self.documents[i] for i in top_indices if scores[i] > 0]


def hybrid_retrieval(
    vectorstore: FAISS,
    bm25_index: BM25Index,
    query: str,
    k: int = 4,
) -> List[Document]:
    """
    Perform hybrid retrieval combining FAISS dense vector search and BM25 sparse keyword search.
    Deduplicates results while preserving ranking order.
    """
    # 1. FAISS vector search
    vector_docs = vectorstore.similarity_search(query, k=k)

    # 2. BM25 keyword search
    keyword_docs = bm25_index.search(query, k=k)

    # Combine and deduplicate
    combined_docs: List[Document] = []
    seen = set()

    for doc in vector_docs + keyword_docs:
        # Unique fingerprint per chunk
        doc_key = (
            doc.page_content.strip(),
            doc.metadata.get("source"),
            doc.metadata.get("page"),
        )
        if doc_key not in seen:
            seen.add(doc_key)
            combined_docs.append(doc)

    return combined_docs
