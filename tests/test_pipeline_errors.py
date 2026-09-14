from unittest.mock import MagicMock

import pytest

from src.pipeline.inference import InferencePipeline


class FakeSemanticCache:
    async def get(self, *args, **kwargs):
        return None

    async def set(self, *args, **kwargs):
        return None


class FakeBudgetManager:
    def estimate_query_cost(self, *args, **kwargs):
        return 0.0

    async def check_budget(self, *args, **kwargs):
        return True, "ok"


class FakeRetriever:
    async def retrieve(self, *args, **kwargs):
        return "", []


class FakeMemory:
    async def get_history(self, *args, **kwargs):
        return []

    async def add_turn(self, *args, **kwargs):
        return None


class FakeRouter:
    async def route(self, *args, **kwargs):
        return {
            "model_id": "primary-model",
            "fallback_model": "fallback-model",
            "complexity": "simple",
            "confidence": 0.95,
            "strategy": "cost_optimized",
        }


class FailingModelManager:
    def load_model(self, model_id):
        raise RuntimeError("provider secret leaked")


class StreamingModel:
    def __init__(self, chunks):
        self.chunks = chunks

    def count_tokens(self, text):
        return len(text)

    def get_cost(self, input_tokens, output_tokens):
        return 0.0

    async def astream(self, *args, **kwargs):
        for chunk in self.chunks:
            yield chunk


class StreamingModelManager:
    def __init__(self):
        self.loaded = []

    def load_model(self, model_id):
        self.loaded.append(model_id)
        if model_id == "primary-model":
            return StreamingModel(["Error: primary failed"])
        return StreamingModel(["fallback answer"])


def make_pipeline(model_manager):
    pipeline = InferencePipeline.__new__(InferencePipeline)
    pipeline.router = FakeRouter()
    pipeline.model_manager = model_manager
    pipeline.retriever = FakeRetriever()
    pipeline.tracker = MagicMock()
    pipeline.budget_manager = FakeBudgetManager()
    pipeline.semantic_cache = FakeSemanticCache()
    pipeline.memory = FakeMemory()
    return pipeline


@pytest.mark.asyncio
async def test_pipeline_error_response_does_not_expose_exception_text():
    pipeline = make_pipeline(FailingModelManager())

    result = await pipeline.run("hello", user_id="user-1")

    assert result["success"] is False
    assert result["answer"] == "Request failed. Please try again."
    assert result["error"] == "pipeline_error"
    assert "provider secret" not in result["answer"]
    assert "provider secret" not in result["error"]


@pytest.mark.asyncio
async def test_streaming_fallback_uses_fallback_model_key():
    manager = StreamingModelManager()
    pipeline = make_pipeline(manager)

    events = [
        event
        async for event in pipeline.astream_run("hello", user_id="user-1", session_id="session-1")
    ]

    assert manager.loaded == ["primary-model", "fallback-model"]
    assert {"type": "replace", "content": ""} in events
    assert events[-1]["type"] == "done"
    assert events[-1]["result"]["answer"] == "fallback answer"
    assert events[-1]["result"]["model_used"] == "fallback-model"
