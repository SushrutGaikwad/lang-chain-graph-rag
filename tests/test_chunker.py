"""Tests for document chunker."""

from langchain_core.documents import Document

from src.ingestion.chunker import DocChunker
from src.config import CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS


def _make_doc(content: str, source: str = "test/doc.mdx") -> Document:
    """Helper to create a test Document."""
    return Document(
        page_content=content,
        metadata={"source": source, "library": "test", "char_count": len(content)},
    )


class TestDocChunker:
    """Tests for the DocChunker class."""

    def test_short_doc_produces_single_chunk(self) -> None:
        """A document shorter than chunk_size should produce exactly one chunk."""
        doc = _make_doc("This is a short document.")
        chunker = DocChunker()
        chunks = chunker.chunk_documents([doc])

        assert len(chunks) == 1
        assert chunks[0].page_content == "This is a short document."

    def test_long_doc_produces_multiple_chunks(self) -> None:
        """A document longer than chunk_size should produce multiple chunks."""
        content = "LangGraph is great. " * 500  # ~10000 chars
        doc = _make_doc(content)
        chunker = DocChunker()
        chunks = chunker.chunk_documents([doc])

        assert len(chunks) > 1

    def test_metadata_preserved_in_chunks(self) -> None:
        """Original metadata should be preserved in each chunk."""
        doc = _make_doc("Some content here.", source="langchain/agents.mdx")
        chunker = DocChunker()
        chunks = chunker.chunk_documents([doc])

        for chunk in chunks:
            assert chunk.metadata["source"] == "langchain/agents.mdx"
            assert chunk.metadata["library"] == "test"

    def test_chunk_metadata_enriched(self) -> None:
        """Chunks should have chunk_index, total_chunks, and chunk_char_count."""
        content = "LangGraph is great. " * 500
        doc = _make_doc(content)
        chunker = DocChunker()
        chunks = chunker.chunk_documents([doc])

        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["total_chunks"] == len(chunks)
            assert chunk.metadata["chunk_char_count"] == len(chunk.page_content)

    def test_chunk_sizes_within_limit(self) -> None:
        """No chunk should exceed the configured chunk size."""
        content = "LangGraph is great. " * 500
        doc = _make_doc(content)
        chunker = DocChunker()
        chunks = chunker.chunk_documents([doc])

        for chunk in chunks:
            assert chunk.metadata["chunk_char_count"] <= CHUNK_SIZE_CHARS, (
                f"Chunk exceeds size limit: {chunk.metadata['chunk_char_count']} > {CHUNK_SIZE_CHARS}"
            )

    def test_empty_input_returns_empty(self) -> None:
        """Chunking an empty list should return an empty list."""
        chunker = DocChunker()
        chunks = chunker.chunk_documents([])

        assert chunks == []

    def test_multiple_docs_chunked_independently(self) -> None:
        """Chunk indices should reset per document."""
        doc1 = _make_doc("Short doc one.", source="a.mdx")
        doc2 = _make_doc("Short doc two.", source="b.mdx")
        chunker = DocChunker()
        chunks = chunker.chunk_documents([doc1, doc2])

        assert len(chunks) == 2
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].metadata["source"] == "a.mdx"
        assert chunks[1].metadata["chunk_index"] == 0
        assert chunks[1].metadata["source"] == "b.mdx"
