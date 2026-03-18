"""Run the full evaluation pipeline."""

from loguru import logger

from src.evaluation.evaluator import RAGEvaluator
from src.config import FAITHFULNESS_THRESHOLD


def main() -> None:
    """Run evaluation: generate answers, score with RAGAS, save report."""
    logger.info("=== Starting Evaluation Pipeline ===")

    evaluator = RAGEvaluator()

    # Step 1: Load golden dataset
    qa_pairs = evaluator.load_golden_dataset()

    # Step 2: Generate answers using the RAG pipeline
    logger.info("=== Generating RAG answers ===")
    results = evaluator.generate_answers(qa_pairs)

    # Step 3: Build RAGAS dataset
    ragas_dataset = evaluator.build_ragas_dataset(results)

    # Step 4: Run RAGAS evaluation
    logger.info("=== Running RAGAS evaluation ===")
    scores = evaluator.run_evaluation(ragas_dataset)

    # Step 5: Save results
    report_path = evaluator.save_results(results, scores)

    # Step 6: Check threshold
    faithfulness_score = scores.get("faithfulness", 0)
    if faithfulness_score >= FAITHFULNESS_THRESHOLD:
        logger.success(
            f"PASSED: faithfulness={faithfulness_score:.4f} "
            f">= threshold={FAITHFULNESS_THRESHOLD}"
        )
    else:
        logger.error(
            f"FAILED: faithfulness={faithfulness_score:.4f} "
            f"< threshold={FAITHFULNESS_THRESHOLD}"
        )

    logger.info(f"Full report: {report_path}")


if __name__ == "__main__":
    main()
