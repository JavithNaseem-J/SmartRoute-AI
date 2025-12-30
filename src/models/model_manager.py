import yaml
from pathlib import Path
from typing import Dict, Optional

from .groq_model import GroqModel
from ..utils.logger import logger


class ModelManager:
    """Manage Groq models for inference"""
    
    def __init__(self, config_path: Path = Path("config/models.yaml")):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.loaded_models: Dict[str, GroqModel] = {}
        
        logger.info("Model manager initialized")
    
    def load_model(self, model_id: str) -> GroqModel:
        """
        Load a model by ID
        
        Args:
            model_id: Model identifier (e.g., 'llama_3_1_8b', 'llama4_scout_17b')
        
        Returns:
            Loaded model instance
        """
        
        # Return if already loaded
        if model_id in self.loaded_models:
            logger.info(f"Model {model_id} already loaded")
            return self.loaded_models[model_id]
        
        # Try loading from Groq models
        if model_id in self.config.get('groq_models', {}):
            model_config = self.config['groq_models'][model_id]
            
            model = GroqModel(
                model_id=model_config['model_id'],
                cost_per_1k_input=model_config.get('cost_per_1k_input', 0.0),
                cost_per_1k_output=model_config.get('cost_per_1k_output', 0.0),
                max_tokens=model_config.get('max_tokens', 4096)
            )
            
            self.loaded_models[model_id] = model
            logger.info(f"✓ Loaded Groq model: {model_id}")
            return model
        
        else:
            raise ValueError(f"Unknown model: {model_id}")
    
    def get_model(self, model_id: str) -> Optional[GroqModel]:
        """Get a loaded model without loading if not present"""
        return self.loaded_models.get(model_id)
    
    def unload_model(self, model_id: str):
        """Unload a model to free memory"""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            logger.info(f"Unloaded model: {model_id}")
    
    def list_available_models(self) -> Dict:
        """List all available models"""
        return {
            'groq': list(self.config.get('groq_models', {}).keys())
        }
    
    def list_loaded_models(self) -> list:
        """List currently loaded models"""
        return list(self.loaded_models.keys())
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get information about a model"""
        
        model = self.get_model(model_id)
        if model:
            return model.get_info()
        
        # Return config info if not loaded
        if model_id in self.config.get('groq_models', {}):
            return {
                'type': 'groq',
                'config': self.config['groq_models'][model_id],
                'loaded': False
            }
        
        return None
    
    def get_model_cost_info(self, model_id: str) -> Dict:
        """Get cost information for a model"""
        
        if model_id in self.config.get('groq_models', {}):
            cfg = self.config['groq_models'][model_id]
            return {
                'type': 'groq',
                'cost_per_1k_input': cfg.get('cost_per_1k_input', 0.0),
                'cost_per_1k_output': cfg.get('cost_per_1k_output', 0.0)
            }
        
        return {}