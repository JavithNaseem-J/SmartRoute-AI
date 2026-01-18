import yaml
from pathlib import Path
from typing import Dict

from src.models.groq_model import GroqModel
from src.utils.logger import logger


class ModelManager:
    """Manage Groq models for inference"""
    
    def __init__(self, config_path: Path = Path("config/models.yaml")):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.loaded_models: Dict[str, GroqModel] = {}
        logger.info("Model manager initialized")
    
    def load_model(self, model_id: str) -> GroqModel:
        """Load a model by ID, returns cached if already loaded."""
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]
        
        if model_id not in self.config.get('groq_models', {}):
            raise ValueError(f"Unknown model: {model_id}")
        
        cfg = self.config['groq_models'][model_id]
        model = GroqModel(
            model_id=cfg['model_id'],
            cost_per_1k_input=cfg.get('cost_per_1k_input', 0.0),
            cost_per_1k_output=cfg.get('cost_per_1k_output', 0.0),
            max_tokens=cfg.get('max_tokens', 4096)
        )
        self.loaded_models[model_id] = model
        logger.info(f"####### Loaded Groq model: {model_id} #######")
        return model
    
    def list_available_models(self) -> Dict:
        """List all available models."""
        return {'groq': list(self.config.get('groq_models', {}).keys())}
    
    def list_loaded_models(self) -> list:
        """List currently loaded models."""
        return list(self.loaded_models.keys())