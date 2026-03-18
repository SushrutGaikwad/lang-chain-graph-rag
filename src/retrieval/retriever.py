"""Retriever wrapper for vector store search."""

from langchain_core.documents import Document
from loguru import logger

from src.config import TOP_K
from src.retrieval.vector_store import VectorStore


class Retriever:
    """Retrieves relevant document chunks for a given query."""

    def __init__(
        self, vector_store: VectorStore | None = None, top_k: int = TOP_K
    ) -> None:
        """Initialize retriever with a vector store.

        Args:
            vector_store: VectorStore instance. Creates a new one if not provided.
            top_k: Number of results to retrieve.
        """
        self.vector_store = vector_store or VectorStore()
        self.top_k = top_k
        logger.info(f"Initialized retriever with top_k={self.top_k}")

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve the top-k most relevant chunks for a query.

        Args:
            query: The user's question.

        Returns:
            List of relevant Document chunks with metadata.
        """
        results = self.vector_store.similarity_search(query, k=self.top_k)
        logger.info(f"Retrieved {len(results)} chunks for query: '{query[:80]}...'")
        return results

    def format_context(self, documents: list[Document]) -> str:
        """Format retrieved documents into a context string for the LLM.

        Each chunk is wrapped with its source metadata for citation.

        Args:
            documents: List of retrieved Document chunks.

        Returns:
            Formatted context string.
        """
        context_parts: list[str] = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "unknown")
            chunk_idx = doc.metadata.get("chunk_index", "?")
            total = doc.metadata.get("total_chunks", "?")
            context_parts.append(
                f"[Source {i}: {source} (chunk {chunk_idx}/{total})]\n"
                f"{doc.page_content}\n"
            )
        return "\n---\n".join(context_parts)
