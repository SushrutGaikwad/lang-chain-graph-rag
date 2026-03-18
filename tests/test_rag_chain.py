"""Tests for the RAG pipeline."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.pipeline.rag_chain import RAGPipeline, RAGResult


def _sample_docs() -> list[Document]:
    """Create sample retrieved documents."""
    return [
        Document(
            page_content="LangGraph uses stateful graphs.",
            metadata={
                "source": "langgraph/overview.mdx",
                "library": "langgraph",
                "chunk_index": 0,
                "total_chunks": 5,
                "chunk_char_count": 35,
            },
        ),
        Document(
            page_content="Memory is added via checkpointers.",
            metadata={
                "source": "langgraph/add-memory.mdx",
                "library": "langgraph",
                "chunk_index": 2,
                "total_chunks": 10,
                "chunk_char_count": 38,
            },
        ),
    ]


class TestRAGResult:
    """Tests for the RAGResult dataclass."""

    def test_sources_deduplicates(self) -> None:
        """sources property should return unique source paths."""
        docs = _sample_docs()
        # Add a duplicate source
        docs.append(
            Document(
                page_content="More about graphs.",
                metadata={"source": "langgraph/overview.mdx"},
            )
        )
        result = RAGResult(
            question="test",
            answer="test answer",
            source_documents=docs,
            context="test context",
            prompt_version="v1",
        )
        assert result.sources == [
            "langgraph/overview.mdx",
            "langgraph/add-memory.mdx",
        ]

    def test_sources_preserves_order(self) -> None:
        """sources should preserve the order of first appearance."""
        docs = _sample_docs()
        result = RAGResult(
            question="test",
            answer="test answer",
            source_documents=docs,
            context="test context",
            prompt_version="v1",
        )
        assert result.sources[0] == "langgraph/overview.mdx"
        assert result.sources[1] == "langgraph/add-memory.mdx"


class TestRAGPipeline:
    """Tests for the RAGPipeline class."""

    def test_query_returns_rag_result(self) -> None:
        """Pipeline query should return a RAGResult with all fields populated."""
        docs = _sample_docs()

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = docs
        mock_retriever.format_context.return_value = "formatted context"
        mock_retriever.top_k = 5

        mock_generator = MagicMock()
        mock_generator.generate.return_value = "Generated answer about memory."
        mock_generator.prompt_loader.version = "v1"
        mock_generator.llm.model = "gemini-2.5-flash"

        pipeline = RAGPipeline(
            retriever=mock_retriever,
            generator=mock_generator,
        )

        result = pipeline.query("How do I add memory?")

        assert isinstance(result, RAGResult)
        assert result.question == "How do I add memory?"
        assert result.answer == "Generated answer about memory."
        assert result.prompt_version == "v1"
        assert len(result.source_documents) == 2
        mock_retriever.retrieve.assert_called_once_with("How do I add memory?")
        mock_generator.generate.assert_called_once()

    def test_query_passes_context_to_generator(self) -> None:
        """Generator should receive the formatted context from retriever."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = _sample_docs()
        mock_retriever.format_context.return_value = "specific context string"
        mock_retriever.top_k = 5

        mock_generator = MagicMock()
        mock_generator.generate.return_value = "answer"
        mock_generator.prompt_loader.version = "v1"
        mock_generator.llm.model = "gemini-2.5-flash"

        pipeline = RAGPipeline(
            retriever=mock_retriever,
            generator=mock_generator,
        )
        pipeline.query("test question")

        mock_generator.generate.assert_called_once_with(
            context="specific context string",
            question="test question",
        )

    def test_query_empty_retrieval(self) -> None:
        """Pipeline should handle zero retrieved documents gracefully."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_retriever.format_context.return_value = ""
        mock_retriever.top_k = 5

        mock_generator = MagicMock()
        mock_generator.generate.return_value = "I don't have enough information."
        mock_generator.prompt_loader.version = "v1"
        mock_generator.llm.model = "gemini-2.5-flash"

        pipeline = RAGPipeline(
            retriever=mock_retriever,
            generator=mock_generator,
        )

        result = pipeline.query("Something obscure?")

        assert result.answer == "I don't have enough information."
        assert result.sources == []
