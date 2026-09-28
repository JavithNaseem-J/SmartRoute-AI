from pathlib import Path

import pytest

from src.evaluation.offline_eval import evaluate_dataset


@pytest.mark.asyncio
async def test_offline_rag_evaluation_clears_quality_gate(monkeypatch):
    monkeypatch.setenv("RERANKER_MODE", "local")
    report = await evaluate_dataset(Path("data/evaluation/rag_eval.json"))

    assert report.sample_count >= 5
    assert report.passed(0.8)
