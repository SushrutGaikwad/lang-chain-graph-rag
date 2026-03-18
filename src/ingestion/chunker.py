"""Split documents into overlapping chunks with metadata preservation."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.config import CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS


class DocChunker:
    """Splits documents into chunks using recursive character splitting."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE_CHARS,
        chunk_overlap: int = CHUNK_OVERLAP_CHARS,
    ) -> None:
        """Initialize the text splitter.

        Args:
            chunk_size: Maximum chunk size in characters.
            chunk_overlap: Overlap between consecutive chunks in characters.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
            length_function=len,
        )
        logger.info(
            f"Initialized chunker: size={chunk_size} chars, overlap={chunk_overlap} chars"
        )

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Split a list of documents into chunks, preserving and enriching metadata.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            List of chunked Document objects with added chunk metadata.
        """
        all_chunks: list[Document] = []

        for doc in documents:
            splits = self.splitter.split_text(doc.page_content)

            for i, chunk_text in enumerate(splits):
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(splits),
                        "chunk_char_count": len(chunk_text),
                    },
                )
                all_chunks.append(chunk_doc)

        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks")

        # Log chunk size statistics
        chunk_sizes = [c.metadata["chunk_char_count"] for c in all_chunks]
        avg_size = sum(chunk_sizes) // len(chunk_sizes) if chunk_sizes else 0
        logger.info(
            f"Chunk sizes: min={min(chunk_sizes, default=0)}, "
            f"max={max(chunk_sizes, default=0)}, avg={avg_size}"
        )

        return all_chunks
