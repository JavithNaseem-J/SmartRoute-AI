from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.model_manager import ModelManager
from src.models.openai_compatible_model import LLMAuthenticationError
from src.models.provider_config import load_provider_settings


MODELS_CONFIG = Path("config/models.yaml")


def test_groq_provider_resolves_production_models(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_API_KEY", "test-groq-key")

    manager = ModelManager(MODELS_CONFIG)

    assert manager.provider == "groq"
    assert manager.model_id("economy") == "openai/gpt-oss-20b"
    assert manager.model_id("quality") == "openai/gpt-oss-120b"
    assert set(manager.available_tiers) == {"economy", "balanced", "quality", "fallback"}


def test_openrouter_uses_active_llm_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_API_KEY", "active-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "legacy-key")

    settings = load_provider_settings(MODELS_CONFIG)

    assert settings.api_key == "active-key"


def test_openrouter_does_not_use_legacy_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "legacy-key")

    settings = load_provider_settings(MODELS_CONFIG)

    assert settings.configured is False
    with pytest.raises(LLMAuthenticationError, match="LLM_API_KEY"):
        ModelManager(MODELS_CONFIG).load_model("economy")


def test_provider_is_not_guessed_from_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unsupported")
    monkeypatch.setenv("LLM_API_KEY", "gsk_key-shape-does-not-matter")

    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        load_provider_settings(MODELS_CONFIG)


def test_missing_active_provider_key_fails_when_model_is_loaded(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    manager = ModelManager(MODELS_CONFIG)

    assert manager.configured is False
    with pytest.raises(LLMAuthenticationError, match="LLM_API_KEY"):
        manager.load_model("economy")


def test_provider_config_rejects_invalid_endpoint(tmp_path, monkeypatch):
    config = tmp_path / "models.yaml"
    config.write_text(
        """
providers:
  groq:
    base_url: not-a-url
    models:
      economy: {model_id: a}
      balanced: {model_id: b}
      quality: {model_id: c}
      fallback: {model_id: d}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    with pytest.raises(ValueError, match="invalid base_url"):
        load_provider_settings(config)


@pytest.mark.asyncio
async def test_provider_validation_checks_configured_models(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    client = MagicMock()
    client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(id="openai/gpt-oss-20b"),
                SimpleNamespace(id="openai/gpt-oss-120b"),
            ]
        )
    )

    with patch("src.models.model_manager.AsyncOpenAI", return_value=client):
        manager = ModelManager(MODELS_CONFIG)
        await manager.validate_provider()

    assert manager.provider_ready is True
