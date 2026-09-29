import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.offline_eval import evaluate_dataset  # noqa: E402


async def main() -> None:
    dataset = Path("data/evaluation/rag_eval.json")
    report = await evaluate_dataset(dataset)
    print(json.dumps({"benchmark": "supplied_passage_reranking", **asdict(report)}, indent=2))
    if not report.passed(0.8):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
