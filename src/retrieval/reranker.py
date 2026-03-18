"""Cross-encoder reranker for rescoring retrieved candidates."""

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from loguru import logger

from src.config import RERANK_FINAL_K


class Reranker:
    """Reranks candidate documents using a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        final_k: int = RERANK_FINAL_K,
    ) -> None:
        """Initialize the cross-encoder model.

        Args:
            model_name: HuggingFace model name for the cross-encoder.
            final_k: Number of documents to return after reranking.
        """
        self.model = CrossEncoder(model_name)
        self.final_k = final_k
        self.model_name = model_name
        logger.info(f"Initialized reranker: model={model_name}, final_k={final_k}")

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """Rerank documents by cross-encoder relevance score.

        Args:
            query: The user's question.
            documents: List of candidate Document chunks to rerank.

        Returns:
            Top final_k documents sorted by cross-encoder score descending.
        """
        if not documents:
            return []

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, doc.page_content) for doc in documents]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Pair documents with scores, sort descending
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        top_k = scored_docs[: self.final_k]

        logger.info(
            f"Reranked {len(documents)} candidates -> top {len(top_k)} | "
            f"best={top_k[0][1]:.4f}, worst={top_k[-1][1]:.4f}"
        )
        logger.debug(
            "Reranked sources: "
            + ", ".join(
                f"{doc.metadata.get('source', '?')}({score:.3f})"
                for doc, score in top_k
            )
        )

        return [doc for doc, _ in top_k]
