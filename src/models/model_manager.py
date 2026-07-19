from pathlib import Path
from typing import Dict

import yaml  # type: ignore[import-untyped]

from src.models.base import BaseLLM
from src.models.groq_model import GroqModel
from src.utils.logger import logger

# Dynamic project root so paths work from any working directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ModelManager:
    """Load and cache LLM instances by config key.

    Returns BaseLLM objects — the caller never depends on GroqModel directly,
    making provider swaps (OpenAI, Gemini, vLLM) a one-line config change.
    """

    def __init__(self, config_path: Path = _PROJECT_ROOT / "config" / "models.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.loaded_models: Dict[str, BaseLLM] = {}
        logger.info("ModelManager initialized")

    def load_model(self, model_id: str) -> BaseLLM:
        """Return a cached model instance, loading it on first access."""
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]

        if model_id not in self.config.get("groq_models", {}):
            raise ValueError(f"Unknown model: '{model_id}'. Check config/models.yaml.")

        cfg = self.config["groq_models"][model_id]
        model: BaseLLM = GroqModel(
            model_id=cfg["model_id"],
            cost_per_1k_input=cfg.get("cost_per_1k_input", 0.0),
            cost_per_1k_output=cfg.get("cost_per_1k_output", 0.0),
            max_tokens=cfg.get("max_tokens", 4096),
        )
        self.loaded_models[model_id] = model
        logger.info(f"Loaded model: {model_id}")
        return model

    def list_available_models(self) -> Dict:
        """List all models declared in config."""
        return {"groq": list(self.config.get("groq_models", {}).keys())}

    def list_loaded_models(self) -> list:
        """List currently instantiated model keys."""
        return list(self.loaded_models.keys())
