"""Evaluation pipeline using RAGAS with Claude Opus 4.6 as the evaluator LLM."""

import json
from pathlib import Path

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from loguru import logger
from dotenv import load_dotenv

from src.config import (
    GOLDEN_DATASET_PATH,
    EVALUATOR_MODEL,
    EVALUATOR_TEMPERATURE,
    EMBEDDING_MODEL,
    FAITHFULNESS_THRESHOLD,
)
from src.pipeline.rag_chain_v2 import RAGPipelineV2, RAGResult

load_dotenv()


def _build_evaluator_llm():
    """Build a RAGAS-compatible async LLM for Claude Opus 4.6.

    Patches model_args to remove top_p, which conflicts with
    temperature on the Anthropic API.
    """
    client = AsyncAnthropic()
    llm = llm_factory(
        EVALUATOR_MODEL,
        provider="anthropic",
        client=client,
    )
    # Anthropic API rejects requests with both temperature and top_p
    llm.model_args.pop("top_p", None)
    llm.model_args["temperature"] = EVALUATOR_TEMPERATURE
    llm.model_args["max_tokens"] = 4096
    return llm


def _build_evaluator_embeddings():
    """Build RAGAS-compatible async embeddings for answer relevancy metric."""
    client = AsyncOpenAI()
    return embedding_factory(
        "openai",
        model=EMBEDDING_MODEL,
        client=client,
    )


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

        self.evaluator_llm = _build_evaluator_llm()
        logger.info(f"Initialized evaluator LLM: {EVALUATOR_MODEL}")

        self.evaluator_embeddings = _build_evaluator_embeddings()
        logger.info("Initialized evaluator embeddings")

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

    def _build_metrics(self) -> list:
        """Build RAGAS metric instances with the evaluator LLM."""
        return [
            Faithfulness(llm=self.evaluator_llm),
            AnswerRelevancy(
                llm=self.evaluator_llm, embeddings=self.evaluator_embeddings
            ),
            ContextPrecision(llm=self.evaluator_llm),
            ContextRecall(llm=self.evaluator_llm),
        ]

    def run_evaluation(self, ragas_dataset: EvaluationDataset) -> dict:
        """Run RAGAS evaluation metrics by scoring each sample directly.

        Args:
            ragas_dataset: The RAGAS EvaluationDataset.

        Returns:
            Dictionary of average metric scores.
        """
        logger.info("Running RAGAS evaluation with Claude Opus 4.6...")

        metrics = self._build_metrics()
        samples = ragas_dataset.samples
        total = len(samples)

        # Collect per-sample scores for each metric
        all_scores: dict[str, list[float]] = {
            "faithfulness": [],
            "answer_relevancy": [],
            "context_precision": [],
            "context_recall": [],
        }

        async def _score_sample(idx: int, sample: SingleTurnSample) -> dict[str, float]:
            """Score a single sample across all metrics with correct kwargs per metric."""
            sample_scores: dict[str, float] = {}

            # Each metric has a different signature
            metric_kwargs = {
                "faithfulness": {
                    "user_input": sample.user_input,
                    "response": sample.response,
                    "retrieved_contexts": sample.retrieved_contexts,
                },
                "answer_relevancy": {
                    "user_input": sample.user_input,
                    "response": sample.response,
                },
                "context_precision": {
                    "user_input": sample.user_input,
                    "reference": sample.reference,
                    "retrieved_contexts": sample.retrieved_contexts,
                },
                "context_recall": {
                    "user_input": sample.user_input,
                    "retrieved_contexts": sample.retrieved_contexts,
                    "reference": sample.reference,
                },
            }

            for metric in metrics:
                metric_name = metric.name
                kwargs = metric_kwargs.get(metric_name, {})
                try:
                    result = await metric.ascore(**kwargs)
                    sample_scores[metric_name] = float(result.value)
                except Exception as e:
                    logger.warning(
                        f"Metric '{metric_name}' failed on sample {idx + 1}: {e}"
                    )
                    sample_scores[metric_name] = 0.0
            return sample_scores

        async def _run_all() -> None:
            """Score all samples sequentially."""
            for idx, sample in enumerate(samples):
                logger.info(f"Scoring sample {idx + 1}/{total}...")
                sample_scores = await _score_sample(idx, sample)
                for metric_name, score in sample_scores.items():
                    if metric_name in all_scores:
                        all_scores[metric_name].append(score)
                logger.info(
                    f"  Sample {idx + 1} scores: "
                    + ", ".join(f"{k}={v:.4f}" for k, v in sample_scores.items())
                )

        import asyncio

        asyncio.run(_run_all())

        # Compute averages
        avg_scores = {}
        for metric_name, scores_list in all_scores.items():
            avg = sum(scores_list) / len(scores_list) if scores_list else 0.0
            avg_scores[metric_name] = avg

        logger.info("RAGAS evaluation results:")
        for metric, score in avg_scores.items():
            status = "PASS" if score >= FAITHFULNESS_THRESHOLD else "FAIL"
            logger.info(f"  {metric}: {score:.4f} [{status}]")

        return avg_scores

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
