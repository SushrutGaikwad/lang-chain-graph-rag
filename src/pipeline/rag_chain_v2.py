"""Production RAG pipeline with hybrid retrieval, reranking, and citation enforcement."""

from dataclasses import dataclass

from langchain_core.documents import Document
from loguru import logger

from src.ingestion.loader import DocLoader
from src.ingestion.chunker import DocChunker
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever
from src.generation.generator import AnswerGenerator
from src.generation.prompt_templates import PromptLoader

INSUFFICIENT_CONTEXT_PREFIX = "INSUFFICIENT_CONTEXT:"


@dataclass
class RAGResult:
    """Container for a RAG pipeline result."""

    question: str
    answer: str
    source_documents: list[Document]
    context: str
    prompt_version: str
    declined: bool

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


class RAGPipelineV2:
    """Production RAG pipeline with hybrid retrieval, reranking, and citation enforcement."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        bm25_retriever: BM25Retriever | None = None,
        reranker: Reranker | None = None,
        generator: AnswerGenerator | None = None,
    ) -> None:
        """Initialize all pipeline components.

        Args:
            vector_store: VectorStore instance. Creates default if not provided.
            bm25_retriever: BM25Retriever instance. Builds from docs if not provided.
            reranker: Reranker instance. Creates default if not provided.
            generator: AnswerGenerator instance. Creates default if not provided.
        """
        # Vector store
        self.vector_store = vector_store or VectorStore()

        # BM25 (requires loading and chunking documents if not provided)
        if bm25_retriever is None:
            logger.info("Building BM25 index from documents...")
            loader = DocLoader()
            docs = loader.load_all()
            chunker = DocChunker()
            chunks = chunker.chunk_documents(docs)
            self.bm25_retriever = BM25Retriever(chunks)
        else:
            self.bm25_retriever = bm25_retriever

        # Hybrid retriever
        self.hybrid = HybridRetriever(
            vector_store=self.vector_store,
            bm25_retriever=self.bm25_retriever,
        )

        # Reranker
        self.reranker = reranker or Reranker()

        # Context formatter (reuse from Retriever)
        self._formatter = Retriever.__new__(Retriever)

        # Generator
        self.generator = generator or AnswerGenerator()

        logger.info(
            f"Initialized RAG pipeline v2: "
            f"prompt={self.generator.prompt_loader.version}, "
            f"model={self.generator.llm.model}"
        )

    def query(self, question: str) -> RAGResult:
        """Run the full production RAG pipeline.

        Args:
            question: The user's question.

        Returns:
            RAGResult with answer, sources, and citation metadata.
        """
        logger.info(f"[v2] Processing query: '{question[:80]}...'")

        # Step 1: Hybrid retrieval (broad candidate pool)
        candidates = self.hybrid.retrieve(question)
        logger.info(f"[v2] Hybrid retrieval returned {len(candidates)} candidates")

        # Step 2: Cross-encoder reranking (narrow to top-k)
        reranked = self.reranker.rerank(question, candidates)
        logger.info(f"[v2] Reranked to {len(reranked)} documents")

        # Step 3: Format context
        context = self._format_context(reranked)

        # Step 4: Generate answer
        answer = self.generator.generate(context=context, question=question)

        # Step 5: Check for declined answer
        declined = answer.strip().startswith(INSUFFICIENT_CONTEXT_PREFIX)

        result = RAGResult(
            question=question,
            answer=answer,
            source_documents=reranked,
            context=context,
            prompt_version=self.generator.prompt_loader.version,
            declined=declined,
        )

        logger.info(
            f"[v2] Pipeline complete: {len(reranked)} sources, "
            f"declined={declined}, answer={len(answer)} chars"
        )
        return result

    def _format_context(self, documents: list[Document]) -> str:
        """Format retrieved documents into a context string with source labels."""
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
