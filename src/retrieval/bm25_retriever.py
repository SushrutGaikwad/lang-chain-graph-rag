"""BM25 keyword-based retriever for lexical matching."""

import re

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from loguru import logger


def _tokenize(text: str) -> list[str]:
    """Simple whitespace and punctuation tokenizer with lowercasing."""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9_]+", text)
    return tokens


class BM25Retriever:
    """BM25-based keyword retriever over document chunks."""

    def __init__(self, documents: list[Document]) -> None:
        """Build the BM25 index from a list of documents.

        Args:
            documents: List of chunked Document objects.
        """
        self.documents = documents
        self._corpus = [_tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self._corpus)
        logger.info(f"Built BM25 index over {len(documents)} documents")

    def search(self, query: str, k: int = 20) -> list[tuple[Document, float]]:
        """Search for documents matching the query by keyword relevance.

        Args:
            query: The search query string.
            k: Number of results to return.

        Returns:
            List of (Document, score) tuples sorted by BM25 score descending.
        """
        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Pair documents with scores and sort descending
        scored_docs = list(zip(self.documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        top_k = scored_docs[:k]
        logger.debug(
            f"BM25 query: '{query[:60]}...' | "
            f"top score={top_k[0][1]:.2f}, min score={top_k[-1][1]:.2f}"
        )
        return top_k
