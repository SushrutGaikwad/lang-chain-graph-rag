"""Interactive query script for the RAG pipeline."""

from loguru import logger

from src.pipeline.rag_chain import RAGPipeline


def main() -> None:
    """Run interactive query loop."""
    logger.info("Initializing RAG pipeline...")
    pipeline = RAGPipeline()
    logger.info("Ready. Type a question (or 'quit' to exit).\n")

    while True:
        question = input("Question: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        result = pipeline.query(question)

        print(f"\nAnswer:\n{result.answer}")
        print(f"\nSources ({len(result.sources)}):")
        for src in result.sources:
            print(f"  - {src}")
        print(f"\nPrompt version: {result.prompt_version}")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
