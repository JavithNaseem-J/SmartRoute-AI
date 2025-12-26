import yaml
from pathlib import Path
from typing import Dict, Union, Optional

from .local_model import LocalModel
from .api_model import APIModel
from .groq_model import GroqModel
from ..utils.logger import logger


class ModelManager:
    """Manage all models (local + API)"""
    
    def __init__(self, config_path: Path = Path("config/models.yaml")):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.loaded_models: Dict[str, Union[LocalModel, APIModel, GroqModel]] = {}
        
        logger.info("Model manager initialized")
    
    def load_model(self, model_id: str) -> Union[LocalModel, APIModel, GroqModel]:
        """
        Load a model by ID
        
        Args:
            model_id: Model identifier (e.g., 'llama_3_2_1b', 'gpt_4o_mini')
        
        Returns:
            Loaded model instance
        """
        
        # Return if already loaded
        if model_id in self.loaded_models:
            logger.info(f"Model {model_id} already loaded")
            return self.loaded_models[model_id]
        
        # Try loading from local models
        if model_id in self.config.get('local_models', {}):
            model_config = self.config['local_models'][model_id]
            
            model = LocalModel(
                model_id=model_config['model_id'],
                quantization=model_config.get('quantization', '4bit'),
                max_tokens=model_config.get('max_tokens', 2048)
            )
            
            self.loaded_models[model_id] = model
            logger.info(f"✓ Loaded local model: {model_id}")
            return model
        
        # Try loading from Groq models (FREE and fast!)
        elif model_id in self.config.get('groq_models', {}):
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
        
        # Try loading from API models
        elif model_id in self.config.get('api_models', {}):
            model_config = self.config['api_models'][model_id]
            
            model = APIModel(
                model_id=model_config['model_id'],
                provider=model_config['provider'],
                cost_per_1k_input=model_config['cost_per_1k_input'],
                cost_per_1k_output=model_config['cost_per_1k_output'],
                max_tokens=model_config.get('max_tokens', 4096)
            )
            
            self.loaded_models[model_id] = model
            logger.info(f"✓ Loaded API model: {model_id}")
            return model
        
        else:
            raise ValueError(f"Unknown model: {model_id}")
    
    def get_model(self, model_id: str) -> Optional[Union[LocalModel, APIModel]]:
        """Get a loaded model without loading if not present"""
        return self.loaded_models.get(model_id)
    
    def unload_model(self, model_id: str):
        """Unload a model to free memory"""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            
            # Clear GPU cache if local model
            if model_id in self.config.get('local_models', {}):
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            logger.info(f"Unloaded model: {model_id}")
    
    def list_available_models(self) -> Dict:
        """List all available models"""
        return {
            'local': list(self.config.get('local_models', {}).keys()),
            'groq': list(self.config.get('groq_models', {}).keys()),
            'api': list(self.config.get('api_models', {}).keys())
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
        if model_id in self.config.get('local_models', {}):
            return {
                'type': 'local',
                'config': self.config['local_models'][model_id],
                'loaded': False
            }
        
        if model_id in self.config.get('api_models', {}):
            return {
                'type': 'api',
                'config': self.config['api_models'][model_id],
                'loaded': False
            }
        
        return None
    
    def get_model_cost_info(self, model_id: str) -> Dict:
        """Get cost information for a model"""
        
        if model_id in self.config.get('local_models', {}):
            return {
                'type': 'local',
                'cost_per_1k_input': 0.0,
                'cost_per_1k_output': 0.0
            }
        
        if model_id in self.config.get('api_models', {}):
            cfg = self.config['api_models'][model_id]
            return {
                'type': 'api',
                'cost_per_1k_input': cfg['cost_per_1k_input'],
                'cost_per_1k_output': cfg['cost_per_1k_output']
            }
        
        return {}