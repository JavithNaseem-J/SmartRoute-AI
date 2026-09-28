import os
from pathlib import Path
from typing import Any, Dict

from openai import AsyncOpenAI

from src.models.base import BaseLLM
from src.models.openai_compatible_model import OpenAICompatibleModel
from src.models.provider_config import ProviderSettings, load_provider_settings
from src.utils.logger import logger

# Dynamic project root so paths work from any working directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ModelManager:
    """Resolve logical model tiers for one active OpenAI-compatible provider."""

    def __init__(self, config_path: Path = _PROJECT_ROOT / "config" / "models.yaml"):
        self.config_path = config_path
        self.settings: ProviderSettings = load_provider_settings(config_path)
        self.config: Dict[str, Any] = {
            "provider": self.settings.name,
            "models": self.settings.models,
        }
        self.loaded_models: Dict[str, BaseLLM] = {}
        self.provider_ready = False
        logger.info(f"ModelManager initialized: provider={self.settings.name}")

    @property
    def provider(self) -> str:
        return self.settings.name

    @property
    def configured(self) -> bool:
        return self.settings.configured

    @property
    def available_tiers(self) -> list[str]:
        return list(self.settings.models)

    def model_config(self, tier: str) -> Dict[str, Any]:
        return self.settings.model(tier)

    def model_id(self, tier: str) -> str:
        return str(self.model_config(tier)["model_id"])

    def _provider_headers(self) -> Dict[str, str] | None:
        if self.provider != "openrouter":
            return None
        return {
            "HTTP-Referer": os.getenv("APP_PUBLIC_URL", "http://localhost:8000"),
            "X-Title": "SmartRoute-AI",
        }

    async def validate_provider(self) -> None:
        """Authenticate and verify every configured physical model ID."""
        if not self.configured:
            raise RuntimeError(f"LLM_API_KEY is required for active provider '{self.provider}'.")

        client = AsyncOpenAI(
            base_url=self.settings.base_url,
            api_key=self.settings.api_key,
            default_headers=self._provider_headers(),
            timeout=15.0,
        )
        try:
            response = await client.models.list()
        except Exception as exc:
            self.provider_ready = False
            raise RuntimeError(f"Active LLM provider '{self.provider}' validation failed.") from exc

        available = {model.id for model in response.data}
        configured = {self.model_id(tier) for tier in self.available_tiers}
        missing = sorted(configured - available)
        if missing:
            self.provider_ready = False
            raise RuntimeError(
                f"Active provider '{self.provider}' does not expose configured models: "
                f"{', '.join(missing)}"
            )
        self.provider_ready = True

    def load_model(self, tier: str) -> BaseLLM:
        """Return a cached model for a logical tier."""
        if tier in self.loaded_models:
            return self.loaded_models[tier]

        cfg = self.model_config(tier)
        model = OpenAICompatibleModel(
            provider=self.provider,
            base_url=self.settings.base_url,
            api_key=self.settings.api_key,
            model_id=cfg["model_id"],
            cost_per_1k_input=cfg.get("cost_per_1k_input", 0.0),
            cost_per_1k_output=cfg.get("cost_per_1k_output", 0.0),
            max_tokens=cfg.get("max_tokens", 4096),
            default_headers=self._provider_headers(),
        )

        self.loaded_models[tier] = model
        logger.info(f"Loaded model tier: {tier} -> {model.model_id}")
        return model
