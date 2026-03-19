"""Gradio chatbot interface for the RAG pipeline."""

import gradio as gr
from loguru import logger

from src.pipeline.rag_chain_v2 import RAGPipelineV2


def build_pipeline() -> RAGPipelineV2:
    """Initialize the RAG pipeline (runs once at startup)."""
    logger.info("Initializing RAG pipeline for chatbot...")
    pipeline = RAGPipelineV2()
    logger.info("Pipeline ready.")
    return pipeline


pipeline = build_pipeline()


def format_sources(sources: list[str]) -> str:
    """Format source file paths into a readable markdown list."""
    if not sources:
        return "*No sources retrieved.*"
    lines = [f"- `{src}`" for src in sources]
    return "\n".join(lines)


def respond(message: str, history: list[dict]) -> str:
    """Handle a user message and return the RAG response.

    Args:
        message: The user's question.
        history: Conversation history (managed by Gradio).

    Returns:
        Formatted response with answer and sources, yielded character by character.
    """
    if not message.strip():
        yield "Please enter a question about LangChain or LangGraph."
        return

    try:
        result = pipeline.query(message)

        if result.declined:
            full_response = (
                "I don't have enough information in the documentation to answer "
                "this question.\n\n"
                f"**Prompt version:** `{result.prompt_version}`"
            )
        else:
            sources_md = format_sources(result.sources)
            full_response = (
                f"{result.answer}\n\n"
                f"---\n\n"
                f"**Sources ({len(result.sources)}):**\n\n"
                f"{sources_md}\n\n"
                f"**Prompt version:** `{result.prompt_version}` · "
                f"**Chunks retrieved:** {len(result.source_documents)}"
            )

        # Stream character by character for typing animation
        for i in range(len(full_response)):
            yield full_response[: i + 1]

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        yield f"An error occurred while processing your question: {e}"


demo = gr.ChatInterface(
    fn=respond,
    title="LangChain & LangGraph Documentation Assistant",
    description=(
        "Ask questions about LangChain and LangGraph. "
        "Answers are grounded in the official documentation with source citations. "
        "Powered by hybrid retrieval (BM25 + vector search), cross-encoder reranking, "
        "and Google Gemini 2.5 Flash."
    ),
    examples=[
        "How do I add memory to a LangGraph agent?",
        "What is the difference between short-term and long-term memory in LangGraph?",
        "How do I set up a SQL agent in LangChain?",
        "What happens when a graph hits the recursion limit?",
        "How do I add human-in-the-loop approval to an agent?",
    ],
)


if __name__ == "__main__":
    demo.launch()
