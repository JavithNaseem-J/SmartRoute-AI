"""
Tests for concurrent batch_run() in InferencePipeline.

Strategy: mock pipeline.run() with a 1-second async sleep to prove that
batch_run completes in ~1 second (parallel via asyncio.gather) rather than
N seconds (sequential). All tests are now async and run with pytest-asyncio.
"""

import asyncio
import time
from unittest.mock import patch

import pytest

from src.pipeline.inference import InferencePipeline


async def mock_run_1s(query, strategy=None, use_retrieval=True, session_id=None):
    """Simulates a 1-second async LLM call."""
    await asyncio.sleep(1)
    return {"query": query, "answer": "mock answer", "success": True}


@pytest.fixture(scope="module")
def pipeline():
    return InferencePipeline()


@pytest.mark.asyncio
async def test_batch_run_preserves_order(pipeline):
    """Results are in the same order as inputs regardless of completion order."""
    queries = ["Q1", "Q2", "Q3"]

    with patch.object(pipeline, "run", side_effect=mock_run_1s):
        results = await pipeline.batch_run(queries)

    assert len(results) == 3
    # asyncio.gather preserves input order
    assert results[0]["query"] == "Q1"
    assert results[1]["query"] == "Q2"
    assert results[2]["query"] == "Q3"


@pytest.mark.asyncio
async def test_batch_run_is_concurrent(pipeline):
    """5 queries each sleeping 1 second should complete in ~1 second, not 5."""
    queries = [f"Q{i}" for i in range(5)]

    with patch.object(pipeline, "run", side_effect=mock_run_1s):
        start = time.time()
        await pipeline.batch_run(queries)
        elapsed = time.time() - start

    assert elapsed < 2.5, (
        f"batch_run was sequential: completed in {elapsed:.1f}s "
        f"(expected <2.5s for 5 parallel 1s tasks)"
    )


@pytest.mark.asyncio
async def test_batch_run_empty_list(pipeline):
    """Empty input returns empty list without error."""
    result = await pipeline.batch_run([])
    assert result == []


@pytest.mark.asyncio
async def test_batch_run_single_query(pipeline):
    """Single-element batch works correctly."""
    with patch.object(pipeline, "run", side_effect=mock_run_1s):
        result = await pipeline.batch_run(["only one"])

    assert len(result) == 1
    assert result[0]["query"] == "only one"
