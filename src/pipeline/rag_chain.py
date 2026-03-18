"""End-to-end RAG pipeline combining retrieval and generation."""

from dataclasses import dataclass, field

from langchain_core.documents import Document
from loguru import logger

from src.retrieval.retriever import Retriever
from src.generation.generator import AnswerGenerator
from src.generation.prompt_templates import PromptLoader


@dataclass
class RAGResult:
    """Container for a RAG pipeline result."""

    question: str
    answer: str
    source_documents: list[Document]
    context: str
    prompt_version: str

    @property
    def sources(self) -> list[str]:
        """Return deduplicated list of source file paths."""
        seen: set[str] = set()
        result: list[str] = []
        for doc in self.source_documents:
            source = doc.metadata.get("source", "unknown")
            if source not in seen:
                seen.add(source)
                result.append(source)
        return result


class RAGPipeline:
    """Orchestrates the full retrieve-then-generate pipeline."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        generator: AnswerGenerator | None = None,
    ) -> None:
        """Initialize the pipeline components.

        Args:
            retriever: Retriever instance. Creates default if not provided.
            generator: AnswerGenerator instance. Creates default if not provided.
        """
        self.retriever = retriever or Retriever()
        self.generator = generator or AnswerGenerator()
        logger.info(
            f"Initialized RAG pipeline: "
            f"top_k={self.retriever.top_k}, "
            f"model={self.generator.llm.model}, "
            f"prompt={self.generator.prompt_loader.version}"
        )

    def query(self, question: str) -> RAGResult:
        """Run the full RAG pipeline for a question.

        Args:
            question: The user's question.

        Returns:
            RAGResult containing the answer, sources, and context.
        """
        logger.info(f"Processing query: '{question[:80]}...'")

        # Step 1: Retrieve
        documents = self.retriever.retrieve(question)

        # Step 2: Format context
        context = self.retriever.format_context(documents)

        # Step 3: Generate answer
        answer = self.generator.generate(context=context, question=question)

        result = RAGResult(
            question=question,
            answer=answer,
            source_documents=documents,
            context=context,
            prompt_version=self.generator.prompt_loader.version,
        )

        logger.info(
            f"Pipeline complete: {len(documents)} chunks retrieved, "
            f"{len(result.sources)} unique sources, "
            f"answer={len(answer)} chars"
        )
        return result
