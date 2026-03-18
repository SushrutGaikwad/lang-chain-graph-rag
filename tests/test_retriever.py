"""Tests for the retriever module."""

from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.retrieval.retriever import Retriever


def _make_mock_store(docs: list[Document]) -> MagicMock:
    """Create a mock VectorStore that returns the given docs."""
    mock = MagicMock()
    mock.similarity_search.return_value = docs
    return mock


def _sample_docs(n: int = 3) -> list[Document]:
    """Create sample Document objects."""
    return [
        Document(
            page_content=f"Content about topic {i}.",
            metadata={
                "source": f"langchain/doc{i}.mdx",
                "library": "langchain",
                "chunk_index": 0,
                "total_chunks": 1,
                "chunk_char_count": 30,
            },
        )
        for i in range(n)
    ]


class TestRetriever:
    """Tests for the Retriever class."""

    def test_retrieve_returns_documents(self) -> None:
        """Should return documents from the vector store."""
        docs = _sample_docs(3)
        mock_store = _make_mock_store(docs)
        retriever = Retriever(vector_store=mock_store, top_k=3)

        results = retriever.retrieve("test query")

        assert len(results) == 3
        mock_store.similarity_search.assert_called_once_with("test query", k=3)

    def test_format_context_includes_sources(self) -> None:
        """Formatted context should include source paths and chunk info."""
        docs = _sample_docs(2)
        mock_store = _make_mock_store(docs)
        retriever = Retriever(vector_store=mock_store)

        context = retriever.format_context(docs)

        assert "langchain/doc0.mdx" in context
        assert "langchain/doc1.mdx" in context
        assert "Content about topic 0" in context
        assert "Content about topic 1" in context

    def test_format_context_numbers_sources(self) -> None:
        """Each source should be numbered sequentially."""
        docs = _sample_docs(3)
        mock_store = _make_mock_store(docs)
        retriever = Retriever(vector_store=mock_store)

        context = retriever.format_context(docs)

        assert "[Source 1:" in context
        assert "[Source 2:" in context
        assert "[Source 3:" in context

    def test_format_context_empty_list(self) -> None:
        """Formatting an empty list should return an empty string."""
        mock_store = _make_mock_store([])
        retriever = Retriever(vector_store=mock_store)

        context = retriever.format_context([])
        assert context == ""
