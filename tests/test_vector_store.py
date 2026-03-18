"""Tests for vector store (uses mocked embeddings to avoid API calls)."""

import pytest
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from src.retrieval.vector_store import VectorStore


@pytest.fixture
def temp_chroma_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for ChromaDB."""
    chroma_dir = tmp_path / "test_chroma"
    chroma_dir.mkdir()
    return chroma_dir


@pytest.fixture
def mock_embeddings():
    """Mock OpenAIEmbeddings to avoid API calls in tests."""
    with patch("src.retrieval.vector_store.OpenAIEmbeddings") as mock_cls:
        mock_instance = MagicMock()
        # Return one vector per input document
        mock_instance.embed_documents.side_effect = lambda texts: [
            [0.1] * 1536 for _ in texts
        ]
        mock_instance.embed_query.return_value = [0.1] * 1536
        mock_cls.return_value = mock_instance
        yield mock_instance


def _make_chunks(n: int = 5) -> list[Document]:
    """Helper to create test chunk Documents."""
    return [
        Document(
            page_content=f"Test content for chunk {i} about LangGraph agents.",
            metadata={
                "source": f"langgraph/doc{i}.mdx",
                "library": "langgraph",
                "char_count": 50,
                "chunk_index": 0,
                "total_chunks": 1,
                "chunk_char_count": 50,
            },
        )
        for i in range(n)
    ]


class TestVectorStore:
    """Tests for the VectorStore class."""

    def test_init_creates_store(
        self, temp_chroma_dir: Path, mock_embeddings: MagicMock
    ) -> None:
        """VectorStore should initialize without errors."""
        store = VectorStore(
            persist_dir=temp_chroma_dir,
            collection_name="test_collection",
        )
        assert store.get_collection_count() == 0

    def test_add_documents_increases_count(
        self, temp_chroma_dir: Path, mock_embeddings: MagicMock
    ) -> None:
        """Adding documents should increase the collection count."""
        store = VectorStore(
            persist_dir=temp_chroma_dir,
            collection_name="test_collection",
        )
        chunks = _make_chunks(5)
        added = store.add_documents(chunks)

        assert added == 5
        assert store.get_collection_count() == 5

    def test_add_documents_batching(
        self, temp_chroma_dir: Path, mock_embeddings: MagicMock
    ) -> None:
        """Documents should be added in batches of the specified size."""
        store = VectorStore(
            persist_dir=temp_chroma_dir,
            collection_name="test_collection",
        )
        chunks = _make_chunks(10)
        added = store.add_documents(chunks, batch_size=3)

        assert added == 10
        assert store.get_collection_count() == 10

    def test_reset_clears_collection(
        self, temp_chroma_dir: Path, mock_embeddings: MagicMock
    ) -> None:
        """Reset should remove all documents from the collection."""
        store = VectorStore(
            persist_dir=temp_chroma_dir,
            collection_name="test_collection",
        )
        store.add_documents(_make_chunks(5))
        assert store.get_collection_count() == 5

        store.reset()
        assert store.get_collection_count() == 0

    def test_similarity_search_returns_documents(
        self, temp_chroma_dir: Path, mock_embeddings: MagicMock
    ) -> None:
        """Similarity search should return Document objects with metadata."""
        store = VectorStore(
            persist_dir=temp_chroma_dir,
            collection_name="test_collection",
        )
        store.add_documents(_make_chunks(5))

        results = store.similarity_search("LangGraph agents", k=3)

        assert len(results) <= 3
        for doc in results:
            assert isinstance(doc, Document)
            assert "source" in doc.metadata
            assert "library" in doc.metadata
