"""Load and manage versioned prompt templates from YAML files."""

from pathlib import Path

import yaml
from loguru import logger

from src.config import ACTIVE_PROMPT_PATH


class PromptLoader:
    """Loads versioned prompt templates from YAML configuration files."""

    def __init__(self, prompt_path: Path = ACTIVE_PROMPT_PATH) -> None:
        """Initialize with path to a prompt YAML file.

        Args:
            prompt_path: Path to the YAML prompt config.
        """
        self.prompt_path = prompt_path
        self._config = self._load_config()
        logger.info(
            f"Loaded prompt version '{self._config['version']}': "
            f"{self._config['description']}"
        )

    def _load_config(self) -> dict:
        """Load and validate the YAML prompt config."""
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_path}")

        with open(self.prompt_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        required_keys = {"version", "description", "template"}
        missing = required_keys - set(config.keys())
        if missing:
            raise ValueError(f"Prompt config missing keys: {missing}")

        return config

    @property
    def version(self) -> str:
        """Return the prompt version string."""
        return self._config["version"]

    @property
    def description(self) -> str:
        """Return the prompt description."""
        return self._config["description"]

    @property
    def template(self) -> str:
        """Return the raw prompt template string."""
        return self._config["template"]

    def format(self, context: str, question: str) -> str:
        """Format the prompt template with context and question.

        Args:
            context: The formatted context string from retrieved chunks.
            question: The user's question.

        Returns:
            The fully formatted prompt ready for the LLM.
        """
        return self.template.format(context=context, question=question)
