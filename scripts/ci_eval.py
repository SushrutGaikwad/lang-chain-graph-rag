"""CI evaluation script. Exits with non-zero code if quality drops below threshold."""

import sys

from loguru import logger

from src.evaluation.evaluator import RAGEvaluator
from src.config import FAITHFULNESS_THRESHOLD


def main() -> int:
    """Run evaluation and return exit code based on quality threshold.

    Returns:
        0 if all metrics pass, 1 if any metric fails.
    """
    logger.info("=== CI Evaluation Pipeline ===")

    evaluator = RAGEvaluator()

    # Step 1: Load golden dataset
    qa_pairs = evaluator.load_golden_dataset()

    # Step 2: Generate answers
    logger.info("Generating RAG answers...")
    results = evaluator.generate_answers(qa_pairs)

    # Step 3: Build RAGAS dataset
    ragas_dataset = evaluator.build_ragas_dataset(results)

    if len(ragas_dataset.samples) == 0:
        logger.error("No non-declined samples to evaluate. FAILING.")
        return 1

    # Step 4: Run RAGAS evaluation
    logger.info("Running RAGAS evaluation...")
    scores = evaluator.run_evaluation(ragas_dataset)

    # Step 5: Save report
    evaluator.save_results(results, scores)

    # Step 6: Check thresholds
    failed_metrics: list[str] = []
    for metric, score in scores.items():
        if score < FAITHFULNESS_THRESHOLD:
            failed_metrics.append(f"{metric}={score:.4f}")
            logger.error(
                f"FAILED: {metric}={score:.4f} < threshold={FAITHFULNESS_THRESHOLD}"
            )
        else:
            logger.success(
                f"PASSED: {metric}={score:.4f} >= threshold={FAITHFULNESS_THRESHOLD}"
            )

    # Step 7: Summary
    declined_count = sum(1 for r in results if r["declined"])
    declined_pct = declined_count / len(results) * 100

    logger.info(f"Total questions: {len(results)}")
    logger.info(f"Declined: {declined_count} ({declined_pct:.1f}%)")
    logger.info(f"Evaluated: {len(ragas_dataset.samples)}")

    if failed_metrics:
        logger.error(f"CI FAILED: {', '.join(failed_metrics)}")
        return 1

    logger.success("CI PASSED: All metrics above threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
