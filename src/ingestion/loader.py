"""Load markdown documentation files from disk into LangChain Document objects."""

from pathlib import Path

from langchain_core.documents import Document
from loguru import logger

from src.config import DOCS_ROOT, SUPPORTED_EXTENSIONS, MIN_DOC_LENGTH_CHARS


class DocLoader:
    """Recursively loads markdown files and returns LangChain Document objects."""

    def __init__(self, docs_root: Path = DOCS_ROOT) -> None:
        """Initialize with path to documentation root."""
        self.docs_root = docs_root
        self._validate_root()

    def _validate_root(self) -> None:
        """Check that the docs root exists."""
        if not self.docs_root.exists():
            raise FileNotFoundError(f"Docs root not found: {self.docs_root.resolve()}")
        subdirs = [d.name for d in self.docs_root.iterdir() if d.is_dir()]
        logger.info(f"Found subdirectories in docs root: {subdirs}")

    def _detect_library(self, file_path: Path) -> str:
        """Determine which library a file belongs to based on its path."""
        relative = file_path.relative_to(self.docs_root)
        return relative.parts[0] if relative.parts else "unknown"

    def load_all(self) -> list[Document]:
        """Load all valid documentation files recursively.

        Returns:
            List of LangChain Document objects with metadata.
        """
        documents: list[Document] = []
        skipped_small = 0
        skipped_extension = 0

        for file_path in sorted(self.docs_root.rglob("*")):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                skipped_extension += 1
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                continue

            if len(content) < MIN_DOC_LENGTH_CHARS:
                skipped_small += 1
                continue

            relative_path = str(file_path.relative_to(self.docs_root)).replace(
                "\\", "/"
            )
            library = self._detect_library(file_path)

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": relative_path,
                        "library": library,
                        "char_count": len(content),
                    },
                )
            )

        logger.info(
            f"Loaded {len(documents)} documents "
            f"(skipped {skipped_small} too-small, {skipped_extension} non-markdown)"
        )

        libs: dict[str, int] = {}
        for doc in documents:
            lib = doc.metadata["library"]
            libs[lib] = libs.get(lib, 0) + 1
        for lib, count in sorted(libs.items()):
            logger.info(f"  {lib}: {count} files")

        return documents
