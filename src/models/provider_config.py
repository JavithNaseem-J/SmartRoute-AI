import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

import yaml  # type: ignore[import-untyped]

SUPPORTED_PROVIDERS = ("groq", "openrouter")


@dataclass(frozen=True)
class ProviderSettings:
    name: str
    base_url: str
    api_key: str
    models: Dict[str, Dict[str, Any]]

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def model(self, tier: str) -> Dict[str, Any]:
        try:
            return self.models[tier]
        except KeyError as error:
            raise ValueError(f"Unknown model tier '{tier}' for provider '{self.name}'.") from error


def load_provider_settings(config_path: Path) -> ProviderSettings:
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    provider = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
    providers = config.get("providers", {})
    if provider not in providers:
        supported = ", ".join(sorted(providers)) or ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(f"Unsupported LLM_PROVIDER '{provider}'. Choose one of: {supported}.")

    provider_config = providers[provider]
    api_key = os.getenv("LLM_API_KEY", "").strip()

    base_url = str(provider_config.get("base_url", "")).rstrip("/")
    models = provider_config.get("models", {})
    if not base_url:
        raise ValueError(f"Provider '{provider}' has no base_url in {config_path}.")
    parsed_url = urlparse(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError(f"Provider '{provider}' has an invalid base_url: {base_url}.")
    if not models:
        raise ValueError(f"Provider '{provider}' has no model tiers in {config_path}.")

    required_tiers = {"economy", "balanced", "quality", "fallback"}
    missing_tiers = sorted(required_tiers - set(models))
    if missing_tiers:
        raise ValueError(
            f"Provider '{provider}' is missing model tiers: {', '.join(missing_tiers)}."
        )

    for tier in sorted(required_tiers):
        model = models[tier]
        if not isinstance(model, dict) or not str(model.get("model_id", "")).strip():
            raise ValueError(
                f"Provider '{provider}' tier '{tier}' must define a non-empty model_id."
            )

    return ProviderSettings(
        name=provider,
        base_url=base_url,
        api_key=api_key,
        models=models,
    )
