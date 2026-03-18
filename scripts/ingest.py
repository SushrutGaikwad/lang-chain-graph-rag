"""Ingest documentation into the vector store."""

from loguru import logger

from src.ingestion.loader import DocLoader
from src.ingestion.chunker import DocChunker
from src.retrieval.vector_store import VectorStore


def main() -> None:
    """Run the full ingestion pipeline: load, chunk, embed, store."""
    # Step 1: Load documents
    logger.info("=== Step 1: Loading documents ===")
    loader = DocLoader()
    docs = loader.load_all()

    # Step 2: Chunk documents
    logger.info("=== Step 2: Chunking documents ===")
    chunker = DocChunker()
    chunks = chunker.chunk_documents(docs)

    # Step 3: Embed and store
    logger.info("=== Step 3: Embedding and storing ===")
    store = VectorStore()

    # Reset if collection already exists with data
    existing_count = store.get_collection_count()
    if existing_count > 0:
        logger.warning(f"Collection already has {existing_count} chunks, resetting...")
        store.reset()

    store.add_documents(chunks)

    # Verify
    final_count = store.get_collection_count()
    logger.info(f"Ingestion complete. Collection has {final_count} chunks.")


if __name__ == "__main__":
    main()
