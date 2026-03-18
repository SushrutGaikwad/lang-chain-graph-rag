"""Tests for cross-encoder reranker."""

from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from src.retrieval.reranker import Reranker


def _make_docs(n: int = 5) -> list[Document]:
    """Create sample documents."""
    return [
        Document(
            page_content=f"Content about topic {i} with varying relevance.",
            metadata={
                "source": f"doc{i}.mdx",
                "chunk_index": 0,
                "total_chunks": 1,
            },
        )
        for i in range(n)
    ]


class TestReranker:
    """Tests for the Reranker class."""

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_returns_final_k(self, mock_ce_cls: MagicMock) -> None:
        """Should return exactly final_k documents."""
        mock_model = MagicMock()
        # Scores: doc0=0.1, doc1=0.9, doc2=0.5, doc3=0.3, doc4=0.7
        mock_model.predict.return_value = [0.1, 0.9, 0.5, 0.3, 0.7]
        mock_ce_cls.return_value = mock_model

        reranker = Reranker(final_k=3)
        docs = _make_docs(5)
        results = reranker.rerank("test query", docs)

        assert len(results) == 3

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_sorts_by_score_descending(self, mock_ce_cls: MagicMock) -> None:
        """Results should be sorted by cross-encoder score, highest first."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5, 0.3, 0.7]
        mock_ce_cls.return_value = mock_model

        reranker = Reranker(final_k=5)
        docs = _make_docs(5)
        results = reranker.rerank("test query", docs)

        # doc1 (0.9), doc4 (0.7), doc2 (0.5), doc3 (0.3), doc0 (0.1)
        assert results[0].metadata["source"] == "doc1.mdx"
        assert results[1].metadata["source"] == "doc4.mdx"
        assert results[2].metadata["source"] == "doc2.mdx"

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_passes_correct_pairs(self, mock_ce_cls: MagicMock) -> None:
        """Should pass (query, content) pairs to the cross-encoder."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.8]
        mock_ce_cls.return_value = mock_model

        reranker = Reranker(final_k=2)
        docs = _make_docs(2)
        reranker.rerank("my question", docs)

        called_pairs = mock_model.predict.call_args[0][0]
        assert len(called_pairs) == 2
        assert called_pairs[0][0] == "my question"
        assert called_pairs[1][0] == "my question"

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_empty_input(self, mock_ce_cls: MagicMock) -> None:
        """Should handle empty document list gracefully."""
        mock_ce_cls.return_value = MagicMock()
        reranker = Reranker(final_k=5)
        results = reranker.rerank("test", [])

        assert results == []

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_fewer_than_final_k(self, mock_ce_cls: MagicMock) -> None:
        """Should return all docs if fewer than final_k are provided."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8, 0.3]
        mock_ce_cls.return_value = mock_model

        reranker = Reranker(final_k=10)
        docs = _make_docs(2)
        results = reranker.rerank("test", docs)

        assert len(results) == 2
