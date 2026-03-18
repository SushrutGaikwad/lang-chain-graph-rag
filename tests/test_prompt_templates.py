"""Tests for prompt template loading."""

import pytest
from pathlib import Path

from src.generation.prompt_templates import PromptLoader
from src.config import ACTIVE_PROMPT_PATH


class TestPromptLoader:
    """Tests for the PromptLoader class."""

    def test_loads_default_prompt(self) -> None:
        """Should load the active prompt version without errors."""
        loader = PromptLoader()
        assert loader.version == "v1"
        assert len(loader.template) > 0

    def test_template_has_placeholders(self) -> None:
        """Template should contain {context} and {question} placeholders."""
        loader = PromptLoader()
        assert "{context}" in loader.template
        assert "{question}" in loader.template

    def test_format_replaces_placeholders(self) -> None:
        """format() should replace both placeholders with provided values."""
        loader = PromptLoader()
        result = loader.format(
            context="Some context here.", question="What is LangGraph?"
        )
        assert "Some context here." in result
        assert "What is LangGraph?" in result
        assert "{context}" not in result
        assert "{question}" not in result

    def test_missing_file_raises_error(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError for non-existent prompt file."""
        with pytest.raises(FileNotFoundError):
            PromptLoader(prompt_path=tmp_path / "nonexistent.yaml")

    def test_invalid_config_raises_error(self, tmp_path: Path) -> None:
        """Should raise ValueError if required keys are missing."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("version: v1\n")
        with pytest.raises(ValueError, match="missing keys"):
            PromptLoader(prompt_path=bad_yaml)
