"""Tests for BM25 keyword retriever."""

from langchain_core.documents import Document

from src.retrieval.bm25_retriever import BM25Retriever, _tokenize


def _make_docs() -> list[Document]:
    """Create sample docs with distinct keywords."""
    return [
        Document(
            page_content="LangGraph uses StateGraph for stateful agent workflows.",
            metadata={
                "source": "langgraph/overview.mdx",
                "chunk_index": 0,
                "total_chunks": 1,
            },
        ),
        Document(
            page_content="GRAPH_RECURSION_LIMIT error is raised when max steps exceeded.",
            metadata={
                "source": "langgraph/errors.mdx",
                "chunk_index": 0,
                "total_chunks": 1,
            },
        ),
        Document(
            page_content="ChromaDB is a vector store for embedding-based retrieval.",
            metadata={
                "source": "langchain/retrieval.mdx",
                "chunk_index": 0,
                "total_chunks": 1,
            },
        ),
    ]


class TestTokenize:
    """Tests for the tokenizer."""

    def test_lowercases(self) -> None:
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self) -> None:
        assert _tokenize("What's up?") == ["what", "s", "up"]

    def test_preserves_underscores(self) -> None:
        assert "graph_recursion_limit" in _tokenize("GRAPH_RECURSION_LIMIT")


class TestBM25Retriever:
    """Tests for the BM25Retriever class."""

    def test_search_returns_ranked_results(self) -> None:
        """Results should be sorted by relevance."""
        docs = _make_docs()
        retriever = BM25Retriever(docs)
        results = retriever.search("StateGraph stateful agent", k=3)

        assert len(results) == 3
        # First result should be the StateGraph doc
        assert results[0][0].metadata["source"] == "langgraph/overview.mdx"

    def test_search_scores_are_descending(self) -> None:
        """Scores should be in descending order."""
        docs = _make_docs()
        retriever = BM25Retriever(docs)
        results = retriever.search("GRAPH_RECURSION_LIMIT error", k=3)

        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_exact_keyword_match_ranks_high(self) -> None:
        """Exact keyword matches should rank at the top."""
        docs = _make_docs()
        retriever = BM25Retriever(docs)
        results = retriever.search("GRAPH_RECURSION_LIMIT", k=3)

        assert results[0][0].metadata["source"] == "langgraph/errors.mdx"

    def test_k_limits_results(self) -> None:
        """Should return at most k results."""
        docs = _make_docs()
        retriever = BM25Retriever(docs)
        results = retriever.search("agent", k=2)

        assert len(results) == 2
