"""Tests for hybrid retriever."""

from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.retrieval.hybrid_retriever import HybridRetriever


def _make_doc(source: str, chunk_index: int, content: str) -> Document:
    """Helper to create a test Document."""
    return Document(
        page_content=content,
        metadata={"source": source, "chunk_index": chunk_index, "total_chunks": 5},
    )


class TestHybridRetriever:
    """Tests for the HybridRetriever class."""

    def test_combines_results_from_both_sources(self) -> None:
        """Should return documents from both vector and BM25 search."""
        vec_doc = _make_doc("vec_only.mdx", 0, "Vector result.")
        bm25_doc = _make_doc("bm25_only.mdx", 0, "BM25 result.")

        mock_vector = MagicMock()
        mock_vector.similarity_search.return_value = [vec_doc]

        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = [(bm25_doc, 5.0)]

        hybrid = HybridRetriever(
            vector_store=mock_vector,
            bm25_retriever=mock_bm25,
            vector_k=5,
            bm25_k=5,
        )
        results = hybrid.retrieve("test query", k=10)

        sources = [d.metadata["source"] for d in results]
        assert "vec_only.mdx" in sources
        assert "bm25_only.mdx" in sources

    def test_shared_documents_get_boosted_score(self) -> None:
        """Documents appearing in both results should rank higher."""
        shared_doc = _make_doc("shared.mdx", 0, "Shared content.")
        vec_only_doc = _make_doc("vec_only.mdx", 0, "Vector only.")

        mock_vector = MagicMock()
        mock_vector.similarity_search.return_value = [shared_doc, vec_only_doc]

        mock_bm25 = MagicMock()
        # shared_doc also appears in BM25 with high score
        mock_bm25.search.return_value = [(shared_doc, 10.0)]

        hybrid = HybridRetriever(
            vector_store=mock_vector,
            bm25_retriever=mock_bm25,
            vector_k=5,
            bm25_k=5,
        )
        results = hybrid.retrieve("test query", k=5)

        # Shared doc should be ranked first due to combined scores
        assert results[0].metadata["source"] == "shared.mdx"

    def test_respects_k_limit(self) -> None:
        """Should return at most k results."""
        docs = [_make_doc(f"doc{i}.mdx", 0, f"Content {i}") for i in range(10)]

        mock_vector = MagicMock()
        mock_vector.similarity_search.return_value = docs[:5]

        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = [(d, 5.0 - i) for i, d in enumerate(docs[5:])]

        hybrid = HybridRetriever(
            vector_store=mock_vector,
            bm25_retriever=mock_bm25,
            vector_k=5,
            bm25_k=5,
        )
        results = hybrid.retrieve("test", k=3)

        assert len(results) == 3

    def test_empty_results_handled(self) -> None:
        """Should handle empty results from one or both sources."""
        mock_vector = MagicMock()
        mock_vector.similarity_search.return_value = []

        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = []

        hybrid = HybridRetriever(
            vector_store=mock_vector,
            bm25_retriever=mock_bm25,
            vector_k=5,
            bm25_k=5,
        )
        results = hybrid.retrieve("test", k=5)

        assert results == []
