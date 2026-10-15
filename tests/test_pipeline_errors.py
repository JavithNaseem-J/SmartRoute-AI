from unittest.mock import AsyncMock, MagicMock

import pytest

from src.pipeline.inference import InferencePipeline
from src.cost.budget import BudgetUnavailableError
from src.models.openai_compatible_model import LLMProviderError


class FakeSemanticCache:
    async def get(self, *args, **kwargs):
        return None

    async def set(self, *args, **kwargs):
        return None

    async def invalidate_user(self, *args, **kwargs):
        return None


class FakeBudgetManager:
    def estimate_query_cost(self, *args, **kwargs):
        return 0.0

    async def check_budget(self, *args, **kwargs):
        return True, "ok"


class FakeRetriever:
    last_citations = []

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
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


class StreamingModelManager:
    def __init__(self):
        self.loaded = []

    def load_model(self, model_id):
        self.loaded.append(model_id)
        if model_id == "primary-model":
            return StreamingModel([RuntimeError("primary provider failed")])
        return StreamingModel(["fallback answer"])


class AllStreamingModelsFailManager:
    def load_model(self, model_id):
        return StreamingModel([RuntimeError(f"{model_id} secret provider failure")])


class GeneratingModel:
    model_id = "provider-model"

    def __init__(self, answer):
        self.answer = answer

    def count_tokens(self, text):
        return len(text)

    def get_cost(self, input_tokens, output_tokens):
        return 0.25

    async def agenerate(self, *args, **kwargs):
        return {"text": self.answer, "input_tokens": 7, "output_tokens": 5}


class GeneratingModelManager:
    def __init__(self, answer):
        self.answer = answer

    def load_model(self, model_id):
        return GeneratingModel(self.answer)


class CategorizedFailingModel(GeneratingModel):
    async def agenerate(self, *args, **kwargs):
        raise LLMProviderError("safe provider failure", category="rate_limited")


class CategorizedFailingModelManager:
    def load_model(self, model_id):
        return CategorizedFailingModel("")


class StaticStreamingModelManager:
    def __init__(self, answer):
        self.answer = answer

    def load_model(self, model_id):
        return StreamingModel([self.answer])


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

    result = await pipeline.run("hello", user_id="user-1", use_retrieval=False)

    assert result["success"] is False
    assert result["answer"] == "Request failed. Please try again."
    assert result["error"] == "pipeline_error"
    assert "provider secret" not in result["answer"]
    assert "provider secret" not in result["error"]


@pytest.mark.asyncio
async def test_pipeline_error_response_survives_metric_logging_failure():
    pipeline = make_pipeline(FailingModelManager())
    pipeline.tracker.log_query.side_effect = RuntimeError("database unavailable")

    result = await pipeline.run("hello", user_id="user-1", use_retrieval=False)

    assert result["success"] is False
    assert result["answer"] == "Request failed. Please try again."
    assert result["error"] == "pipeline_error"


@pytest.mark.asyncio
async def test_provider_failure_category_survives_pipeline_fallback():
    pipeline = make_pipeline(CategorizedFailingModelManager())

    result = await pipeline.run("hello", user_id="user-1", use_retrieval=False)

    assert result["success"] is False
    assert result["error"] == "pipeline_error"
    assert result["error_category"] == "rate_limited"
    assert "provider failure" not in result["answer"]


def configure_grounded_retrieval(pipeline, monkeypatch):
    pipeline.retriever.retrieve = AsyncMock(
        return_value=(
            "[C1] handbook.txt\nThe support window is 47 minutes.",
            ["Source 1: handbook.txt"],
        )
    )
    pipeline.retriever.last_citations = [
        {
            "id": "C1",
            "filename": "handbook.txt",
            "page": None,
            "section": None,
            "excerpt": "The support window is 47 minutes.",
        }
    ]
    monkeypatch.setattr(
        "src.pipeline.inference.list_active_documents",
        lambda tracker, user_id: [{"storage_path": "eval/handbook.txt"}],
    )


@pytest.mark.asyncio
async def test_grounded_answer_without_marker_fails_closed_and_retains_usage(monkeypatch):
    pipeline = make_pipeline(GeneratingModelManager("The support window is 47 minutes."))
    configure_grounded_retrieval(pipeline, monkeypatch)
    pipeline.semantic_cache.set = AsyncMock()
    pipeline.memory.add_turn = AsyncMock()

    result = await pipeline.run(
        "What is the support window?",
        user_id="user-1",
        session_id="session-1",
        use_retrieval=True,
    )

    assert result["success"] is False
    assert result["error"] == "citation_error"
    assert result["error_category"] == "missing_valid_citations"
    assert result["model_used"] == "provider-model"
    assert result["input_tokens"] == 7
    assert result["output_tokens"] == 5
    assert result["cost"] == 0.25
    assert result["citations"] == []
    assert result["sources"] == ["Source 1: handbook.txt"]
    pipeline.semantic_cache.set.assert_not_awaited()
    pipeline.memory.add_turn.assert_not_awaited()
    assert pipeline.tracker.log_query.call_args.kwargs["success"] is False
    assert pipeline.tracker.log_query.call_args.kwargs["cost"] == 0.25


@pytest.mark.asyncio
async def test_explicit_document_not_found_answer_succeeds_without_citation(monkeypatch):
    pipeline = make_pipeline(GeneratingModelManager(InferencePipeline.DOCUMENT_NOT_FOUND_ANSWER))
    configure_grounded_retrieval(pipeline, monkeypatch)
    pipeline.semantic_cache.set = AsyncMock()
    pipeline.memory.add_turn = AsyncMock()

    result = await pipeline.run(
        "What is the airport lounge policy?",
        user_id="user-1",
        session_id="session-1",
        use_retrieval=True,
    )

    assert result["success"] is True
    assert result["error"] is None
    assert result["answer"] == InferencePipeline.DOCUMENT_NOT_FOUND_ANSWER
    assert result["citations"] == []
    assert result["sources"] == ["Source 1: handbook.txt"]
    assert result["cost"] == 0.25
    pipeline.semantic_cache.set.assert_not_awaited()
    pipeline.memory.add_turn.assert_awaited_once_with(
        "user-1",
        "session-1",
        "What is the airport lounge policy?",
        InferencePipeline.DOCUMENT_NOT_FOUND_ANSWER,
    )
    assert pipeline.tracker.log_query.call_args.kwargs["success"] is True


@pytest.mark.asyncio
async def test_document_not_found_answer_discards_citation_marker(monkeypatch):
    answer = f"{InferencePipeline.DOCUMENT_NOT_FOUND_ANSWER} [C1]"
    pipeline = make_pipeline(GeneratingModelManager(answer))
    configure_grounded_retrieval(pipeline, monkeypatch)
    pipeline.semantic_cache.set = AsyncMock()
    pipeline.memory.add_turn = AsyncMock()

    result = await pipeline.run(
        "What is the airport lounge policy?",
        user_id="user-1",
        session_id="session-1",
        use_retrieval=True,
    )

    assert result["success"] is True
    assert result["answer"] == InferencePipeline.DOCUMENT_NOT_FOUND_ANSWER
    assert result["citations"] == []
    pipeline.semantic_cache.set.assert_not_awaited()
    pipeline.memory.add_turn.assert_awaited_once_with(
        "user-1",
        "session-1",
        "What is the airport lounge policy?",
        InferencePipeline.DOCUMENT_NOT_FOUND_ANSWER,
    )


@pytest.mark.asyncio
async def test_grounded_answer_with_valid_marker_still_succeeds(monkeypatch):
    pipeline = make_pipeline(GeneratingModelManager("The support window is 47 minutes [C1]."))
    configure_grounded_retrieval(pipeline, monkeypatch)

    result = await pipeline.run("What is the support window?", user_id="user-1", use_retrieval=True)

    assert result["success"] is True
    assert result["error"] is None
    assert [citation["id"] for citation in result["citations"]] == ["C1"]


@pytest.mark.asyncio
async def test_uncited_grounded_cache_entry_is_ignored(monkeypatch):
    pipeline = make_pipeline(GeneratingModelManager("The support window is 47 minutes [C1]."))
    configure_grounded_retrieval(pipeline, monkeypatch)
    pipeline.semantic_cache.get = AsyncMock(
        return_value={
            "answer": "Old uncited answer",
            "sources": ["Source 1: handbook.txt"],
            "citations": [],
        }
    )

    result = await pipeline.run("What is the support window?", user_id="user-1", use_retrieval=True)

    assert result["success"] is True
    assert result["answer"] != "Old uncited answer"
    assert result["model_used"] == "provider-model"
    assert [citation["id"] for citation in result["citations"]] == ["C1"]


@pytest.mark.asyncio
async def test_stream_replaces_uncited_grounded_answer_with_terminal_failure(monkeypatch):
    pipeline = make_pipeline(StaticStreamingModelManager("Uncited grounded answer"))
    configure_grounded_retrieval(pipeline, monkeypatch)
    pipeline.semantic_cache.set = AsyncMock()
    pipeline.memory.add_turn = AsyncMock()

    events = [
        event
        async for event in pipeline.astream_run(
            "What is the support window?",
            user_id="user-1",
            session_id="session-1",
            use_retrieval=True,
        )
    ]

    done_events = [event for event in events if event["type"] == "done"]
    assert len(done_events) == 1
    assert done_events[0]["result"]["success"] is False
    assert done_events[0]["result"]["error"] == "citation_error"
    assert done_events[0]["result"]["error_category"] == "missing_valid_citations"
    assert events[-2] == {
        "type": "replace",
        "content": InferencePipeline.MISSING_CITATIONS_ANSWER,
    }
    pipeline.semantic_cache.set.assert_not_awaited()
    pipeline.memory.add_turn.assert_not_awaited()


@pytest.mark.asyncio
async def test_stream_allows_explicit_document_not_found_answer_without_citation(monkeypatch):
    pipeline = make_pipeline(
        StaticStreamingModelManager(InferencePipeline.DOCUMENT_NOT_FOUND_ANSWER)
    )
    configure_grounded_retrieval(pipeline, monkeypatch)
    pipeline.semantic_cache.set = AsyncMock()
    pipeline.memory.add_turn = AsyncMock()

    events = [
        event
        async for event in pipeline.astream_run(
            "What is the airport lounge policy?",
            user_id="user-1",
            session_id="session-1",
            use_retrieval=True,
        )
    ]

    done_events = [event for event in events if event["type"] == "done"]
    assert len(done_events) == 1
    assert done_events[0]["result"]["success"] is True
    assert done_events[0]["result"]["error"] is None
    assert done_events[0]["result"]["answer"] == InferencePipeline.DOCUMENT_NOT_FOUND_ANSWER
    assert done_events[0]["result"]["citations"] == []
    assert not any(event["type"] == "replace" for event in events)
    pipeline.semantic_cache.set.assert_not_awaited()
    pipeline.memory.add_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_discards_citation_marker_from_document_not_found_answer(monkeypatch):
    answer = f"{InferencePipeline.DOCUMENT_NOT_FOUND_ANSWER} [C1]"
    pipeline = make_pipeline(StaticStreamingModelManager(answer))
    configure_grounded_retrieval(pipeline, monkeypatch)
    pipeline.semantic_cache.set = AsyncMock()
    pipeline.memory.add_turn = AsyncMock()

    events = [
        event
        async for event in pipeline.astream_run(
            "What is the airport lounge policy?",
            user_id="user-1",
            session_id="session-1",
            use_retrieval=True,
        )
    ]

    result = events[-1]["result"]
    assert events[-2] == {
        "type": "replace",
        "content": InferencePipeline.DOCUMENT_NOT_FOUND_ANSWER,
    }
    assert result["success"] is True
    assert result["answer"] == InferencePipeline.DOCUMENT_NOT_FOUND_ANSWER
    assert result["citations"] == []
    pipeline.semantic_cache.set.assert_not_awaited()
    pipeline.memory.add_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_streaming_fallback_uses_fallback_model_key():
    manager = StreamingModelManager()
    pipeline = make_pipeline(manager)

    events = [
        event
        async for event in pipeline.astream_run(
            "hello", user_id="user-1", session_id="session-1", use_retrieval=False
        )
    ]

    assert manager.loaded == ["primary-model", "fallback-model"]
    assert {"type": "replace", "content": ""} in events
    assert events[-1]["type"] == "done"
    assert events[-1]["result"]["answer"] == "fallback answer"
    assert events[-1]["result"]["model_used"] == "fallback-model"


@pytest.mark.asyncio
async def test_streaming_fallback_failure_is_terminal_error_without_secret_text():
    pipeline = make_pipeline(AllStreamingModelsFailManager())

    events = [
        event
        async for event in pipeline.astream_run(
            "hello", user_id="user-1", session_id="session-1", use_retrieval=False
        )
    ]

    assert events[-2] == {"type": "replace", "content": "Request failed. Please try again."}
    assert events[-1]["type"] == "done"
    assert events[-1]["result"]["success"] is False
    assert events[-1]["result"]["error"] == "pipeline_error"
    assert "secret" not in events[-1]["result"]["answer"]


@pytest.mark.asyncio
async def test_rag_with_no_sources_does_not_fall_back_to_general_model():
    pipeline = make_pipeline(FailingModelManager())
    pipeline.budget_manager.check_budget = MagicMock()

    result = await pipeline.run("what is your personality?", user_id="user-1")

    assert result["success"] is True
    assert result["sources"] == []
    assert "uploaded documents" in result["answer"]
    assert result["routing_info"]["reason"] == "no_retrieved_document_sources"
    pipeline.budget_manager.check_budget.assert_not_called()


@pytest.mark.asyncio
async def test_budget_infrastructure_failure_is_reported_as_service_unavailable():
    pipeline = make_pipeline(FailingModelManager())

    async def unavailable(*args, **kwargs):
        raise BudgetUnavailableError("redis host secret")

    pipeline.budget_manager.check_budget = unavailable

    result = await pipeline.run("hello", user_id="user-1", use_retrieval=False)

    assert result["success"] is False
    assert result["error"] == "budget_unavailable"
    assert result["answer"] == "Service is temporarily unavailable. Please try again later."
    assert "redis" not in result["answer"].lower()


@pytest.mark.asyncio
async def test_streaming_rag_with_no_sources_returns_honest_no_source_answer():
    pipeline = make_pipeline(FailingModelManager())

    events = [
        event
        async for event in pipeline.astream_run(
            "what is your personality?", user_id="user-1", session_id="session-1"
        )
    ]

    assert events[0]["type"] == "metadata"
    assert events[0]["data"]["sources"] == []
    assert events[1]["type"] == "chunk"
    assert "uploaded documents" in events[1]["content"]
    assert events[-1]["type"] == "done"
    assert events[-1]["result"]["routing_info"]["reason"] == "no_retrieved_document_sources"


def test_citation_validation_removes_invented_markers_and_unused_evidence():
    citations = [
        {
            "id": "C1",
            "filename": "Resume.pdf",
            "page": 2,
            "section": None,
            "excerpt": "Backend engineering experience.",
        },
        {
            "id": "C2",
            "filename": "Cover Letter.pdf",
            "page": 1,
            "section": None,
            "excerpt": "Application for the backend role.",
        },
    ]

    answer, used = InferencePipeline._validate_answer_citations(
        "Backend experience [C1]. Invented evidence [C9].", citations
    )

    assert answer == "Backend experience [C1]. Invented evidence ."
    assert [citation["id"] for citation in used] == ["C1"]


def test_citation_validation_normalizes_provider_bracket_variants():
    citations = [
        {
            "id": "C1",
            "filename": "Cover Letter.pdf",
            "page": 1,
            "section": None,
            "excerpt": "Backend Engineer position.",
        }
    ]

    answer, used = InferencePipeline._validate_answer_citations(
        "Backend Engineer【C1】. Again ［c1］. Invented 【C9】.", citations
    )

    assert answer == "Backend Engineer[C1]. Again [C1]. Invented ."
    assert [citation["id"] for citation in used] == ["C1"]


@pytest.mark.asyncio
async def test_stream_normalizes_provider_citation_and_keeps_evidence(monkeypatch):
    pipeline = make_pipeline(StreamingModelManager())
    pipeline.model_manager.load_model = lambda tier: StreamingModel(["Backend Engineer【C1】"])
    pipeline.retriever.last_citations = [
        {
            "id": "C1",
            "filename": "Cover Letter.pdf",
            "page": 1,
            "section": None,
            "excerpt": "Backend Engineer position.",
        }
    ]
    pipeline.retriever.retrieve = AsyncMock(
        return_value=(
            "[C1] Cover Letter.pdf, page 1\nBackend Engineer position.",
            ["Source 1: Cover Letter.pdf - page 1"],
        )
    )
    monkeypatch.setattr(
        "src.pipeline.inference.list_active_documents",
        lambda tracker, user_id: [{"storage_path": "eval/Cover Letter.pdf"}],
    )

    events = [
        event
        async for event in pipeline.astream_run("Which role?", user_id="user-1", use_retrieval=True)
    ]

    assert any(event == {"type": "replace", "content": "Backend Engineer[C1]"} for event in events)
    assert events[-1]["result"]["answer"] == "Backend Engineer[C1]"
    assert [citation["id"] for citation in events[-1]["result"]["citations"]] == ["C1"]
