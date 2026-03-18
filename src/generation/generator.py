"""Answer generation using Google Gemini 2.5 Flash."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from loguru import logger
from dotenv import load_dotenv

from src.config import GENERATION_MODEL, GENERATION_TEMPERATURE, MAX_OUTPUT_TOKENS
from src.generation.prompt_templates import PromptLoader

load_dotenv()


class AnswerGenerator:
    """Generates answers using Gemini 2.5 Flash with versioned prompts."""

    def __init__(self, prompt_loader: PromptLoader | None = None) -> None:
        """Initialize the LLM and prompt loader.

        Args:
            prompt_loader: PromptLoader instance. Creates default if not provided.
        """
        self.prompt_loader = prompt_loader or PromptLoader()
        self.llm = ChatGoogleGenerativeAI(
            model=GENERATION_MODEL,
            temperature=GENERATION_TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        logger.info(
            f"Initialized generator: model={GENERATION_MODEL}, "
            f"prompt_version={self.prompt_loader.version}"
        )

    def generate(self, context: str, question: str) -> str:
        """Generate an answer given context and a question.

        Args:
            context: Formatted context string from retrieved chunks.
            question: The user's question.

        Returns:
            The generated answer string.
        """
        prompt = self.prompt_loader.format(context=context, question=question)

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            answer = response.content
            logger.info(
                f"Generated answer ({len(answer)} chars) for: '{question[:80]}...'"
            )
            return answer
        except Exception as e:
            logger.error(f"Generation failed for query: '{question[:80]}...': {e}")
            raise
