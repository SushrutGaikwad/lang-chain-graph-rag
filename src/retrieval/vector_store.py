"""ChromaDB vector store with OpenAI embeddings."""

from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger
from dotenv import load_dotenv

from src.config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
)

load_dotenv()


class VectorStore:
    """Manages ChromaDB vector store with OpenAI embeddings."""

    def __init__(
        self,
        persist_dir: Path = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        """Initialize embeddings and ChromaDB client.

        Args:
            persist_dir: Directory to persist ChromaDB data.
            collection_name: Name of the ChromaDB collection.
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        logger.info(
            f"Initialized embeddings: model={EMBEDDING_MODEL}, "
            f"dimensions={EMBEDDING_DIMENSIONS}"
        )

        self.store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )
        logger.info(
            f"Connected to ChromaDB: collection='{self.collection_name}', "
            f"persist_dir='{self.persist_dir}'"
        )

    def add_documents(self, documents: list[Document], batch_size: int = 100) -> int:
        """Add documents to the vector store in batches.

        Args:
            documents: List of LangChain Document objects to embed and store.
            batch_size: Number of documents per batch to avoid API limits.

        Returns:
            Total number of documents added.
        """
        total = len(documents)
        added = 0

        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            try:
                self.store.add_documents(batch)
                added += len(batch)
                logger.info(
                    f"Added batch {i // batch_size + 1}: {added}/{total} chunks"
                )
            except Exception as e:
                logger.error(f"Failed to add batch starting at index {i}: {e}")
                raise

        logger.info(f"Finished adding {added} chunks to vector store")
        return added

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """Search for similar documents by query.

        Args:
            query: The search query string.
            k: Number of results to return.

        Returns:
            List of matching Document objects.
        """
        results = self.store.similarity_search(query, k=k)
        logger.debug(f"Query: '{query[:80]}...' returned {len(results)} results")
        return results

    def get_collection_count(self) -> int:
        """Return the number of documents in the collection."""
        return self.store._collection.count()

    def reset(self) -> None:
        """Delete all documents from the collection."""
        self.store.delete_collection()
        self.store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )
        logger.warning(f"Reset collection '{self.collection_name}'")
