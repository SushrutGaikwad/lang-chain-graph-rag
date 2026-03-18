"""Tests for the evaluation module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.evaluator import RAGEvaluator
from src.pipeline.rag_chain_v2 import RAGResult


@pytest.fixture
def sample_golden_dataset(tmp_path: Path) -> Path:
    """Create a temporary golden dataset file."""
    dataset = {
        "metadata": {"total_pairs": 2},
        "qa_pairs": [
            {
                "question": "What is LangGraph?",
                "answer": "LangGraph is a framework for building stateful agents.",
                "source_files": ["langgraph/overview.mdx"],
                "supporting_passages": ["LangGraph is a framework..."],
                "question_type": "conceptual",
            },
            {
                "question": "How do I add memory?",
                "answer": "Use a checkpointer when creating the agent.",
                "source_files": ["langgraph/add-memory.mdx"],
                "supporting_passages": ["Use a checkpointer..."],
                "question_type": "procedural",
            },
        ],
    }
    path = tmp_path / "golden_dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    return path


class TestRAGEvaluator:
    """Tests for the RAGEvaluator class."""

    def test_load_golden_dataset(self, sample_golden_dataset: Path) -> None:
        """Should load QA pairs from JSON file."""
        with patch.object(RAGEvaluator, "__init__", lambda self, **kwargs: None):
            evaluator = RAGEvaluator()
            evaluator.golden_dataset_path = sample_golden_dataset
            pairs = evaluator.load_golden_dataset()

        assert len(pairs) == 2
        assert pairs[0]["question"] == "What is LangGraph?"
        assert pairs[1]["question_type"] == "procedural"

    def test_build_ragas_dataset_skips_declined(self) -> None:
        """Should exclude declined answers from the RAGAS dataset."""
        with patch.object(RAGEvaluator, "__init__", lambda self, **kwargs: None):
            evaluator = RAGEvaluator()

        results = [
            {
                "question": "Q1",
                "golden_answer": "A1",
                "generated_answer": "Generated A1",
                "contexts": ["context"],
                "source_files": ["file.mdx"],
                "question_type": "factual",
                "declined": False,
                "prompt_version": "v2",
            },
            {
                "question": "Q2",
                "golden_answer": "A2",
                "generated_answer": "INSUFFICIENT_CONTEXT: ...",
                "contexts": [],
                "source_files": [],
                "question_type": "factual",
                "declined": True,
                "prompt_version": "v2",
            },
        ]
        dataset = evaluator.build_ragas_dataset(results)

        assert len(dataset.samples) == 1
        assert dataset.samples[0].user_input == "Q1"

    def test_save_results(self, tmp_path: Path) -> None:
        """Should save a valid JSON report."""
        with patch.object(RAGEvaluator, "__init__", lambda self, **kwargs: None):
            evaluator = RAGEvaluator()

        results = [
            {
                "question": "Q1",
                "golden_answer": "A1",
                "generated_answer": "Gen A1",
                "contexts": ["ctx"],
                "source_files": ["f.mdx"],
                "question_type": "factual",
                "declined": False,
                "prompt_version": "v2",
            }
        ]
        scores = {"faithfulness": 0.85, "answer_relevancy": 0.90}
        output_path = tmp_path / "report.json"

        evaluator.save_results(results, scores, output_path=output_path)

        assert output_path.exists()
        report = json.loads(output_path.read_text())
        assert report["average_scores"]["faithfulness"] == 0.85
        assert report["metadata"]["total_questions"] == 1
        assert report["metadata"]["declined_count"] == 0

    def test_generate_answers_handles_errors(self) -> None:
        """Should catch exceptions and record them as declined."""
        mock_pipeline = MagicMock()
        mock_pipeline.query.side_effect = Exception("API error")

        with patch.object(RAGEvaluator, "__init__", lambda self, **kwargs: None):
            evaluator = RAGEvaluator()
            evaluator.pipeline = mock_pipeline

        pairs = [{"question": "Q1", "answer": "A1", "question_type": "factual"}]
        results = evaluator.generate_answers(pairs)

        assert len(results) == 1
        assert results[0]["declined"] is True
        assert "ERROR" in results[0]["generated_answer"]
