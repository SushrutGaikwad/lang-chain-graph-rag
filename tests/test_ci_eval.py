"""Tests for the CI evaluation script."""

from unittest.mock import patch, MagicMock

from scripts.ci_eval import main


class TestCIEval:
    """Tests for the CI evaluation script."""

    @patch("scripts.ci_eval.RAGEvaluator")
    def test_passes_when_above_threshold(self, mock_eval_cls: MagicMock) -> None:
        """CI should return 0 when all metrics are above threshold."""
        mock_evaluator = MagicMock()
        mock_eval_cls.return_value = mock_evaluator

        mock_evaluator.load_golden_dataset.return_value = [
            {"question": "Q1", "answer": "A1", "question_type": "factual"}
        ]
        mock_evaluator.generate_answers.return_value = [
            {"question": "Q1", "declined": False}
        ]

        mock_dataset = MagicMock()
        mock_dataset.samples = [MagicMock()]
        mock_evaluator.build_ragas_dataset.return_value = mock_dataset

        mock_evaluator.run_evaluation.return_value = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.90,
            "context_precision": 0.80,
            "context_recall": 0.75,
        }

        exit_code = main()
        assert exit_code == 0

    @patch("scripts.ci_eval.RAGEvaluator")
    def test_fails_when_below_threshold(self, mock_eval_cls: MagicMock) -> None:
        """CI should return 1 when any metric is below threshold."""
        mock_evaluator = MagicMock()
        mock_eval_cls.return_value = mock_evaluator

        mock_evaluator.load_golden_dataset.return_value = [
            {"question": "Q1", "answer": "A1", "question_type": "factual"}
        ]
        mock_evaluator.generate_answers.return_value = [
            {"question": "Q1", "declined": False}
        ]

        mock_dataset = MagicMock()
        mock_dataset.samples = [MagicMock()]
        mock_evaluator.build_ragas_dataset.return_value = mock_dataset

        mock_evaluator.run_evaluation.return_value = {
            "faithfulness": 0.50,  # below 0.7 threshold
            "answer_relevancy": 0.90,
            "context_precision": 0.80,
            "context_recall": 0.75,
        }

        exit_code = main()
        assert exit_code == 1

    @patch("scripts.ci_eval.RAGEvaluator")
    def test_fails_when_all_declined(self, mock_eval_cls: MagicMock) -> None:
        """CI should return 1 when all answers are declined."""
        mock_evaluator = MagicMock()
        mock_eval_cls.return_value = mock_evaluator

        mock_evaluator.load_golden_dataset.return_value = [
            {"question": "Q1", "answer": "A1", "question_type": "factual"}
        ]
        mock_evaluator.generate_answers.return_value = [
            {"question": "Q1", "declined": True}
        ]

        mock_dataset = MagicMock()
        mock_dataset.samples = []
        mock_evaluator.build_ragas_dataset.return_value = mock_dataset

        exit_code = main()
        assert exit_code == 1
