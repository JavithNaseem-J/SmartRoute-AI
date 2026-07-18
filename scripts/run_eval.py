from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.evaluation.ragas_eval import EvalReport, EvalSample, RagasEvaluator  # noqa: E402
from src.pipeline.inference import InferencePipeline  # noqa: E402
from src.utils.logger import logger  # noqa: E402

# Edit this list to add your own domain-specific evaluation questions.
# ground_truth = the ideal answer you expect from a well-functioning RAG system.
# ─────────────────────────────────────────────────────────────────────────────
EVAL_SAMPLES = [
    EvalSample(
        question="What is route optimisation?",
        ground_truth=(
            "Route optimisation is the process of determining the most "
            "cost-efficient and time-efficient route for a vehicle or shipment, "
            "considering constraints like capacity, time windows, and traffic."
        ),
    ),
    EvalSample(
        question="What factors affect freight shipping costs?",
        ground_truth=(
            "Freight shipping costs are affected by distance, weight, volume, "
            "mode of transport, fuel surcharges, delivery urgency, and route complexity."
        ),
    ),
    EvalSample(
        question="What is LTL shipping?",
        ground_truth=(
            "LTL (Less Than Truckload) shipping is a freight service where "
            "multiple shippers share space on a single truck, paying only for "
            "the portion they use. It's cost-effective for smaller shipments "
            "that don't fill an entire trailer."
        ),
    ),
    EvalSample(
        question="How does multi-modal freight work?",
        ground_truth=(
            "Multi-modal freight uses two or more modes of transport (e.g. "
            "truck + rail + ship) in a single shipment under one contract, "
            "combining the cost benefits of each mode for long-distance routes."
        ),
    ),
    EvalSample(
        question="What is a bill of lading?",
        ground_truth=(
            "A bill of lading is a legal document issued by a carrier to a "
            "shipper that details the type, quantity, and destination of goods "
            "being carried. It serves as a receipt, a contract of carriage, "
            "and a document of title."
        ),
    ),
]


async def main() -> None:
    logger.info("Initialising pipeline...")
    pipeline = InferencePipeline()

    evaluator = RagasEvaluator(pipeline)

    output_path = Path("data/eval_report.json")
    report: EvalReport = await evaluator.run(EVAL_SAMPLES, output_path=output_path)

    print("\n" + "=" * 50)
    print("RAGAS EVALUATION REPORT")
    print("=" * 50)
    print(report.summary())
    print("=" * 50)

    if report.passed(threshold=0.70):
        print("✅ All metrics above 70% — RAG pipeline is performing well.")
        sys.exit(0)
    else:
        print("❌ One or more metrics below 70% — investigate retrieval quality.")
        print("   Tip: check context_recall first (are the right docs indexed?)")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
