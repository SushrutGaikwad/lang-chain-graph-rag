"""Evaluation pipeline using RAGAS with Claude Opus 4.6 as the evaluator LLM."""

import json
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas import EvaluationDataset, SingleTurnSample
from loguru import logger
from dotenv import load_dotenv

from src.config import (
    GOLDEN_DATASET_PATH,
    EVALUATOR_MODEL,
    EVALUATOR_TEMPERATURE,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    FAITHFULNESS_THRESHOLD,
)
from src.pipeline.rag_chain_v2 import RAGPipelineV2, RAGResult

load_dotenv()


class RAGEvaluator:
    """Evaluates RAG pipeline outputs against the golden dataset using RAGAS."""

    def __init__(
        self,
        pipeline: RAGPipelineV2 | None = None,
        golden_dataset_path: Path = GOLDEN_DATASET_PATH,
    ) -> None:
        """Initialize evaluator with pipeline and golden dataset.

        Args:
            pipeline: The RAG pipeline to evaluate. Creates default if not provided.
            golden_dataset_path: Path to the golden dataset JSON.
        """
        self.pipeline = pipeline or RAGPipelineV2()
        self.golden_dataset_path = golden_dataset_path

        # Evaluator LLM (Claude Opus 4.6)
        self.evaluator_llm = ChatAnthropic(
            model=EVALUATOR_MODEL,
            temperature=EVALUATOR_TEMPERATURE,
        )
        logger.info(f"Initialized evaluator LLM: {EVALUATOR_MODEL}")

        # Embeddings for RAGAS metrics that need them
        self.evaluator_embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
        )

    def load_golden_dataset(self) -> list[dict]:
        """Load the golden dataset from JSON.

        Returns:
            List of QA pair dictionaries.
        """
        with open(self.golden_dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        pairs = dataset["qa_pairs"]
        logger.info(
            f"Loaded {len(pairs)} golden QA pairs from {self.golden_dataset_path}"
        )
        return pairs

    def generate_answers(self, qa_pairs: list[dict]) -> list[dict]:
        """Run the RAG pipeline on each golden question and collect results.

        Args:
            qa_pairs: List of golden QA pair dictionaries.

        Returns:
            List of result dictionaries with pipeline outputs.
        """
        results: list[dict] = []

        for i, pair in enumerate(qa_pairs):
            question = pair["question"]
            logger.info(
                f"Generating answer {i + 1}/{len(qa_pairs)}: '{question[:60]}...'"
            )

            try:
                rag_result: RAGResult = self.pipeline.query(question)
                results.append(
                    {
                        "question": question,
                        "golden_answer": pair["answer"],
                        "generated_answer": rag_result.answer,
                        "contexts": [
                            doc.page_content for doc in rag_result.source_documents
                        ],
                        "source_files": rag_result.sources,
                        "question_type": pair.get("question_type", "unknown"),
                        "declined": rag_result.declined,
                        "prompt_version": rag_result.prompt_version,
                    }
                )
            except Exception as e:
                logger.error(f"Failed on question {i + 1}: {e}")
                results.append(
                    {
                        "question": question,
                        "golden_answer": pair["answer"],
                        "generated_answer": f"ERROR: {e}",
                        "contexts": [],
                        "source_files": [],
                        "question_type": pair.get("question_type", "unknown"),
                        "declined": True,
                        "prompt_version": "error",
                    }
                )

        logger.info(
            f"Generated {len(results)} answers "
            f"({sum(1 for r in results if r['declined'])} declined)"
        )
        return results

    def build_ragas_dataset(self, results: list[dict]) -> EvaluationDataset:
        """Convert pipeline results into a RAGAS EvaluationDataset.

        Args:
            results: List of result dictionaries from generate_answers.

        Returns:
            RAGAS EvaluationDataset.
        """
        samples = []
        for r in results:
            if r["declined"]:
                logger.debug(f"Skipping declined question: '{r['question'][:60]}...'")
                continue

            samples.append(
                SingleTurnSample(
                    user_input=r["question"],
                    response=r["generated_answer"],
                    retrieved_contexts=r["contexts"],
                    reference=r["golden_answer"],
                )
            )

        logger.info(
            f"Built RAGAS dataset with {len(samples)} samples "
            f"(skipped {len(results) - len(samples)} declined)"
        )
        return EvaluationDataset(samples=samples)

    def run_evaluation(self, ragas_dataset: EvaluationDataset) -> dict:
        """Run RAGAS evaluation metrics.

        Args:
            ragas_dataset: The RAGAS EvaluationDataset.

        Returns:
            Dictionary of metric scores.
        """
        logger.info("Running RAGAS evaluation with Claude Opus 4.6...")

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        result = evaluate(
            dataset=ragas_dataset,
            metrics=metrics,
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embeddings,
        )

        scores = result.to_pandas()
        avg_scores = {
            "faithfulness": float(scores["faithfulness"].mean()),
            "answer_relevancy": float(scores["answer_relevancy"].mean()),
            "context_precision": float(scores["context_precision"].mean()),
            "context_recall": float(scores["context_recall"].mean()),
        }

        logger.info("RAGAS evaluation results:")
        for metric, score in avg_scores.items():
            status = "PASS" if score >= FAITHFULNESS_THRESHOLD else "FAIL"
            logger.info(f"  {metric}: {score:.4f} [{status}]")

        return avg_scores

    def evaluate_by_question_type(
        self, results: list[dict], ragas_dataset: EvaluationDataset
    ) -> dict[str, dict]:
        """Break down RAGAS scores by question type.

        Args:
            results: The original pipeline results with question_type.
            ragas_dataset: The evaluated RAGAS dataset.

        Returns:
            Dictionary mapping question_type to average metric scores.
        """
        scores_df = evaluate(
            dataset=ragas_dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embeddings,
        ).to_pandas()

        # Map back to question types (only non-declined results are in the dataset)
        non_declined = [r for r in results if not r["declined"]]
        scores_df["question_type"] = [r["question_type"] for r in non_declined]

        type_scores: dict[str, dict] = {}
        for qtype in scores_df["question_type"].unique():
            subset = scores_df[scores_df["question_type"] == qtype]
            type_scores[qtype] = {
                "count": len(subset),
                "faithfulness": float(subset["faithfulness"].mean()),
                "answer_relevancy": float(subset["answer_relevancy"].mean()),
            }

        logger.info("Scores by question type:")
        for qtype, metrics in sorted(type_scores.items()):
            logger.info(
                f"  {qtype}: faithfulness={metrics['faithfulness']:.4f}, "
                f"relevancy={metrics['answer_relevancy']:.4f} (n={metrics['count']})"
            )

        return type_scores

    def save_results(
        self,
        results: list[dict],
        scores: dict,
        output_path: Path | None = None,
    ) -> Path:
        """Save evaluation results and scores to JSON.

        Args:
            results: Pipeline result dictionaries.
            scores: Average RAGAS metric scores.
            output_path: Path to save the report. Defaults to data/eval/eval_report.json.

        Returns:
            Path to the saved report.
        """
        output_path = output_path or (GOLDEN_DATASET_PATH.parent / "eval_report.json")

        report = {
            "metadata": {
                "evaluator_model": EVALUATOR_MODEL,
                "faithfulness_threshold": FAITHFULNESS_THRESHOLD,
                "total_questions": len(results),
                "declined_count": sum(1 for r in results if r["declined"]),
            },
            "average_scores": scores,
            "results": results,
        }

        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"Evaluation report saved to {output_path}")
        return output_path
