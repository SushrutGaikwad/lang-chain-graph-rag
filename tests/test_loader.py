"""Tests for document loader."""

import pytest
from langchain_core.documents import Document

from src.ingestion.loader import DocLoader
from src.config import DOCS_ROOT


class TestDocLoader:
    """Tests for the DocLoader class."""

    def test_load_all_returns_documents(self) -> None:
        """Loader should return a non-empty list of Document objects."""
        loader = DocLoader()
        docs = loader.load_all()

        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_documents_have_required_metadata(self) -> None:
        """Each document should have source, library, and char_count metadata."""
        loader = DocLoader()
        docs = loader.load_all()

        required_keys = {"source", "library", "char_count"}
        for doc in docs:
            assert required_keys.issubset(doc.metadata.keys()), (
                f"Missing metadata keys in {doc.metadata.get('source', 'unknown')}"
            )

    def test_source_paths_use_forward_slashes(self) -> None:
        """Source paths should use forward slashes, not backslashes."""
        loader = DocLoader()
        docs = loader.load_all()

        for doc in docs:
            assert "\\" not in doc.metadata["source"], (
                f"Backslash found in source: {doc.metadata['source']}"
            )

    def test_library_detection(self) -> None:
        """Each document should belong to a known library."""
        loader = DocLoader()
        docs = loader.load_all()

        known_libraries = {"langchain", "langgraph", "concepts"}
        for doc in docs:
            assert doc.metadata["library"] in known_libraries, (
                f"Unknown library: {doc.metadata['library']}"
            )

    def test_no_empty_content(self) -> None:
        """No document should have empty page_content."""
        loader = DocLoader()
        docs = loader.load_all()

        for doc in docs:
            assert len(doc.page_content.strip()) > 0, (
                f"Empty content in {doc.metadata['source']}"
            )

    def test_invalid_root_raises_error(self, tmp_path: pytest.TempPathFactory) -> None:
        """Loader should raise FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            DocLoader(docs_root=tmp_path / "nonexistent")
