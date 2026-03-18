"""Hybrid retriever combining vector search and BM25 keyword search."""

from langchain_core.documents import Document
from loguru import logger

from src.config import VECTOR_K, BM25_K, HYBRID_WEIGHTS, RERANK_INITIAL_K
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever


class HybridRetriever:
    """Combines vector similarity and BM25 keyword search with score fusion."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        vector_k: int = VECTOR_K,
        bm25_k: int = BM25_K,
        vector_weight: float = HYBRID_WEIGHTS["vector"],
        bm25_weight: float = HYBRID_WEIGHTS["bm25"],
    ) -> None:
        """Initialize hybrid retriever with both search backends.

        Args:
            vector_store: VectorStore for embedding-based search.
            bm25_retriever: BM25Retriever for keyword-based search.
            vector_k: Number of candidates from vector search.
            bm25_k: Number of candidates from BM25 search.
            vector_weight: Weight for vector search scores in fusion.
            bm25_weight: Weight for BM25 scores in fusion.
        """
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.vector_k = vector_k
        self.bm25_k = bm25_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        logger.info(
            f"Initialized hybrid retriever: "
            f"vector_k={vector_k}, bm25_k={bm25_k}, "
            f"weights=(vector={vector_weight}, bm25={bm25_weight})"
        )

    def _normalize_scores(
        self, scored_docs: list[tuple[Document, float]]
    ) -> list[tuple[Document, float]]:
        """Min-max normalize scores to [0, 1] range."""
        if not scored_docs:
            return []
        scores = [s for _, s in scored_docs]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [(doc, 1.0) for doc, _ in scored_docs]
        return [(doc, (s - min_s) / (max_s - min_s)) for doc, s in scored_docs]

    def _doc_id(self, doc: Document) -> str:
        """Create a unique identifier for a document chunk."""
        source = doc.metadata.get("source", "")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        return f"{source}::{chunk_idx}"

    def retrieve(self, query: str, k: int = RERANK_INITIAL_K) -> list[Document]:
        """Retrieve candidates using reciprocal rank fusion of both methods.

        Args:
            query: The user's question.
            k: Number of fused results to return.

        Returns:
            List of Document chunks sorted by fused score.
        """
        # Get candidates from both sources
        vector_results = self.vector_store.similarity_search(query, k=self.vector_k)
        bm25_results = self.bm25_retriever.search(query, k=self.bm25_k)

        # Convert vector results to scored tuples (reverse index as proxy score)
        vector_scored = [
            (doc, (self.vector_k - i) / self.vector_k)
            for i, doc in enumerate(vector_results)
        ]
        vector_normed = self._normalize_scores(vector_scored)
        bm25_normed = self._normalize_scores(bm25_results)

        # Fuse scores by document identity
        fused: dict[str, tuple[Document, float]] = {}

        for doc, score in vector_normed:
            doc_id = self._doc_id(doc)
            fused[doc_id] = (doc, score * self.vector_weight)

        for doc, score in bm25_normed:
            doc_id = self._doc_id(doc)
            if doc_id in fused:
                existing_doc, existing_score = fused[doc_id]
                fused[doc_id] = (
                    existing_doc,
                    existing_score + score * self.bm25_weight,
                )
            else:
                fused[doc_id] = (doc, score * self.bm25_weight)

        # Sort by fused score descending
        ranked = sorted(fused.values(), key=lambda x: x[1], reverse=True)
        top_k = [doc for doc, _ in ranked[:k]]

        logger.info(
            f"Hybrid retrieval: {len(vector_results)} vector + "
            f"{len(bm25_results)} BM25 = {len(fused)} unique candidates, "
            f"returning top {len(top_k)}"
        )
        return top_k
