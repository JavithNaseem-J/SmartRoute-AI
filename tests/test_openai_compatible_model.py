from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.models.openai_compatible_model import (
    LLMAuthenticationError,
    LLMProviderError,
    OpenAICompatibleModel,
)


def make_model() -> OpenAICompatibleModel:
    return OpenAICompatibleModel(
        provider="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key="test-key",
        model_id="openai/gpt-oss-20b",
        cost_per_1k_input=0.000075,
        cost_per_1k_output=0.0003,
    )


def test_provider_model_rejects_missing_active_key():
    with pytest.raises(LLMAuthenticationError, match="LLM_API_KEY"):
        OpenAICompatibleModel(
            provider="groq",
            base_url="https://api.groq.com/openai/v1",
            api_key="",
            model_id="openai/gpt-oss-20b",
        )


@pytest.mark.asyncio
async def test_provider_generation_preserves_usage(monkeypatch):
    model = make_model()
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="provider answer"))],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=7),
    )
    monkeypatch.setattr(model, "_call_api", AsyncMock(return_value=response))

    result = await model.agenerate([{"role": "user", "content": "hello"}])

    assert result == {"text": "provider answer", "input_tokens": 12, "output_tokens": 7}


@pytest.mark.asyncio
async def test_provider_stream_failure_is_raised_not_rendered_as_text(monkeypatch):
    model = make_model()
    monkeypatch.setattr(
        model,
        "_call_api_stream",
        AsyncMock(side_effect=RuntimeError("upstream secret")),
    )

    with pytest.raises(LLMProviderError, match="stream failed"):
        _ = [chunk async for chunk in model.astream([{"role": "user", "content": "hello"}])]


@pytest.mark.asyncio
async def test_provider_empty_stream_is_a_failure(monkeypatch):
    model = make_model()

    async def empty_stream():
        if False:
            yield None

    monkeypatch.setattr(model, "_call_api_stream", AsyncMock(return_value=empty_stream()))

    with pytest.raises(LLMProviderError, match="empty stream"):
        _ = [chunk async for chunk in model.astream([{"role": "user", "content": "hello"}])]
