"""
RAGAS evaluation for SmartRoute-AI RAG pipeline.

Measures four dimensions of retrieval + generation quality:

  faithfulness      — does the answer only use facts from the retrieved context?
  answer_relevancy  — is the answer actually relevant to the question?
  context_recall    — did retrieval find the chunks needed to answer correctly?
  context_precision — are the retrieved chunks precise (low noise)?

Usage (offline eval against a fixed question set):

    python scripts/run_eval.py

Usage (in tests):

    from src.evaluation.ragas_eval import RagasEvaluator
    evaluator = RagasEvaluator(pipeline)
    results = await evaluator.run(questions, ground_truths)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.utils.logger import logger

# RAGAS is an optional dev dependency — the app runs fine without it.
# Install with: pip install ragas
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False


@dataclass
class EvalSample:
    """One evaluation example: question + expected answer + retrieved context."""

    question: str
    ground_truth: str  # the correct answer you wrote yourself
    answer: str = ""  # filled in by the pipeline during eval
    contexts: List[str] = field(default_factory=list)  # filled in by the pipeline


@dataclass
class EvalReport:
    """Aggregated RAGAS scores for a full eval run."""

    faithfulness: float
    answer_relevancy: float
    context_recall: float
    context_precision: float
    sample_count: int
    failed_samples: int = 0

    def passed(self, threshold: float = 0.70) -> bool:
        """Return True if all metrics clear the minimum threshold."""
        return all(
            score >= threshold
            for score in [
                self.faithfulness,
                self.answer_relevancy,
                self.context_recall,
                self.context_precision,
            ]
        )

    def summary(self) -> str:
        return (
            f"Faithfulness:       {self.faithfulness:.2%}\n"
            f"Answer Relevancy:   {self.answer_relevancy:.2%}\n"
            f"Context Recall:     {self.context_recall:.2%}\n"
            f"Context Precision:  {self.context_precision:.2%}\n"
            f"Samples:            {self.sample_count} ({self.failed_samples} failed)"
        )


class RagasEvaluator:
    """Run RAGAS metrics against the SmartRoute-AI pipeline.

    Args:
        pipeline: An initialised InferencePipeline instance.
    """

    def __init__(self, pipeline) -> None:
        if not _RAGAS_AVAILABLE:
            raise RuntimeError(
                "RAGAS is not installed. Run: pip install ragas datasets"
            )
        self.pipeline = pipeline

    async def _run_single(self, sample: EvalSample) -> EvalSample:
        """Run one question through the pipeline and populate answer + contexts."""
        try:
            result = await self.pipeline.run(
                query=sample.question,
                use_retrieval=True,
            )
            sample.answer = result.get("answer", "")

            # Extract the raw context chunks used during retrieval
            context_text, _ = self.pipeline.retriever.retrieve(sample.question)
            # Split into individual chunks (separated by double newline in retriever)
            sample.contexts = [
                c.strip() for c in context_text.split("\n\n") if c.strip()
            ]

        except Exception as exc:
            logger.warning(f"Eval sample failed: {exc}")
            sample.answer = ""
            sample.contexts = []

        return sample

    async def run(
        self,
        samples: List[EvalSample],
        output_path: Optional[Path] = None,
    ) -> EvalReport:
        """Evaluate the pipeline on the given samples and return an EvalReport.

        Args:
            samples:     List of EvalSample with question + ground_truth filled in.
            output_path: Optional path to write the raw RAGAS scores as JSON.
        """
        import asyncio

        logger.info(f"Running RAGAS evaluation on {len(samples)} samples...")

        # Run pipeline for every sample concurrently
        completed = await asyncio.gather(*[self._run_single(s) for s in samples])

        # RAGAS expects a Hugging Face Dataset with specific column names
        ragas_data = {
            "question": [s.question for s in completed],
            "answer": [s.answer for s in completed],
            "contexts": [s.contexts for s in completed],
            "ground_truth": [s.ground_truth for s in completed],
        }
        dataset = Dataset.from_dict(ragas_data)

        failed = sum(1 for s in completed if not s.answer)

        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        )

        report = EvalReport(
            faithfulness=float(scores["faithfulness"]),
            answer_relevancy=float(scores["answer_relevancy"]),
            context_recall=float(scores["context_recall"]),
            context_precision=float(scores["context_precision"]),
            sample_count=len(completed),
            failed_samples=failed,
        )

        logger.info(f"RAGAS evaluation complete:\n{report.summary()}")

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "faithfulness": report.faithfulness,
                        "answer_relevancy": report.answer_relevancy,
                        "context_recall": report.context_recall,
                        "context_precision": report.context_precision,
                        "sample_count": report.sample_count,
                        "passed": report.passed(),
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Eval report saved → {output_path}")

        return report
