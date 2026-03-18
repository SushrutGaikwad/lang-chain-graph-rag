"""Tests for the production RAG pipeline v2."""

from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.pipeline.rag_chain_v2 import (
    RAGPipelineV2,
    RAGResult,
    INSUFFICIENT_CONTEXT_PREFIX,
)


def _sample_docs() -> list[Document]:
    """Create sample retrieved documents."""
    return [
        Document(
            page_content="LangGraph uses stateful graphs for agent workflows.",
            metadata={
                "source": "langgraph/overview.mdx",
                "chunk_index": 0,
                "total_chunks": 5,
            },
        ),
        Document(
            page_content="Memory is added via checkpointers in LangGraph.",
            metadata={
                "source": "langgraph/add-memory.mdx",
                "chunk_index": 2,
                "total_chunks": 10,
            },
        ),
    ]


def _build_mock_pipeline(answer: str = "Mocked answer.") -> RAGPipelineV2:
    """Build a pipeline with all components mocked."""
    mock_hybrid = MagicMock()
    mock_hybrid.retrieve.return_value = _sample_docs()

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = _sample_docs()

    mock_generator = MagicMock()
    mock_generator.generate.return_value = answer
    mock_generator.prompt_loader.version = "v2"
    mock_generator.llm.model = "gemini-2.5-flash"

    pipeline = RAGPipelineV2.__new__(RAGPipelineV2)
    pipeline.hybrid = mock_hybrid
    pipeline.reranker = mock_reranker
    pipeline.generator = mock_generator
    pipeline.vector_store = MagicMock()
    pipeline.bm25_retriever = MagicMock()

    return pipeline


class TestRAGResult:
    """Tests for the RAGResult dataclass."""

    def test_declined_true_when_insufficient(self) -> None:
        """declined should be True when answer starts with INSUFFICIENT_CONTEXT."""
        result = RAGResult(
            question="test",
            answer=f"{INSUFFICIENT_CONTEXT_PREFIX} Not enough info.",
            source_documents=[],
            context="",
            prompt_version="v2",
            declined=True,
        )
        assert result.declined is True

    def test_declined_false_for_normal_answer(self) -> None:
        """declined should be False for a normal answer."""
        result = RAGResult(
            question="test",
            answer="LangGraph uses graphs for workflow orchestration.",
            source_documents=_sample_docs(),
            context="some context",
            prompt_version="v2",
            declined=False,
        )
        assert result.declined is False

    def test_sources_deduplicated(self) -> None:
        """sources property should return unique source paths."""
        docs = _sample_docs()
        docs.append(
            Document(
                page_content="Duplicate source.",
                metadata={"source": "langgraph/overview.mdx"},
            )
        )
        result = RAGResult(
            question="test",
            answer="answer",
            source_documents=docs,
            context="ctx",
            prompt_version="v2",
            declined=False,
        )
        assert len(result.sources) == 2


class TestRAGPipelineV2:
    """Tests for the RAGPipelineV2 class."""

    def test_query_returns_rag_result(self) -> None:
        """Pipeline should return a properly structured RAGResult."""
        pipeline = _build_mock_pipeline("Answer about LangGraph [Source 1].")
        result = pipeline.query("What is LangGraph?")

        assert isinstance(result, RAGResult)
        assert result.question == "What is LangGraph?"
        assert result.answer == "Answer about LangGraph [Source 1]."
        assert result.prompt_version == "v2"
        assert result.declined is False

    def test_query_calls_hybrid_then_reranker(self) -> None:
        """Pipeline should call hybrid retrieval then reranker in order."""
        pipeline = _build_mock_pipeline()
        pipeline.query("test question")

        pipeline.hybrid.retrieve.assert_called_once_with("test question")
        pipeline.reranker.rerank.assert_called_once()

    def test_query_detects_declined_answer(self) -> None:
        """Pipeline should detect INSUFFICIENT_CONTEXT prefix."""
        pipeline = _build_mock_pipeline(
            f"{INSUFFICIENT_CONTEXT_PREFIX} The sources do not cover this topic."
        )
        result = pipeline.query("What is quantum computing?")

        assert result.declined is True

    def test_query_passes_reranked_docs_to_context(self) -> None:
        """Generator should receive context built from reranked documents."""
        pipeline = _build_mock_pipeline()
        result = pipeline.query("test")

        # Context should include content from the sample docs
        assert "langgraph/overview.mdx" in result.context
        assert "langgraph/add-memory.mdx" in result.context
