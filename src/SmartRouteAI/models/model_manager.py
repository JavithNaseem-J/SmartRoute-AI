from typing import Dict, Optional, Union
from src.models.local_model import LocalModel, LocalModelManager
from src.models.api_model import APIModel
from src.utils.config import load_model_config
from src.utils.logging import logger


class ModelManager:
    """Manage all models (local + API) - Custom orchestration"""
    
    def __init__(self):
        self.config = load_model_config()
        self.local_manager = LocalModelManager()
        self.api_models: Dict[str, APIModel] = {}
        self.active_models = set()
        
        logger.info("Model Manager initialized")
    
    def load_model(self, model_id: str) -> Union[LocalModel, APIModel]:
        """Load a model by ID"""
        
        if model_id in self.active_models:
            return self.get_model(model_id)
        
        # Try local models
        if model_id in self.config.get("local_models", {}):
            model_config = self.config["local_models"][model_id]
            model = self.local_manager.load_model(model_id, model_config)
            self.active_models.add(model_id)
            return model
        
        # Try API models
        elif model_id in self.config.get("api_models", {}):
            model_config = self.config["api_models"][model_id]
            model = APIModel(
                model_id=model_id,
                model_name=model_config["model_id"],
                provider=model_config["provider"],
                cost_per_1k_input=model_config["cost_per_1k_input"],
                cost_per_1k_output=model_config["cost_per_1k_output"],
                max_tokens=model_config.get("max_tokens", 4096),
                temperature=model_config.get("temperature", 0.7)
            )
            self.api_models[model_id] = model
            self.active_models.add(model_id)
            return model
        
        else:
            raise ValueError(f"Unknown model: {model_id}")
    
    def get_model(self, model_id: str) -> Optional[Union[LocalModel, APIModel]]:
        """Get a loaded model"""
        model = self.local_manager.get_model(model_id)
        if model:
            return model
        return self.api_models.get(model_id)
    
    def unload_model(self, model_id: str):
        """Unload a model"""
        if model_id in self.local_manager.models:
            self.local_manager.unload_model(model_id)
        elif model_id in self.api_models:
            del self.api_models[model_id]
        
        self.active_models.discard(model_id)
        logger.info(f"Unloaded {model_id}")
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get model information"""
        model = self.get_model(model_id)
        if model:
            return model.get_info()
        return None
    
    def list_active_models(self):
        """List all active models"""
        return list(self.active_models)
    
    def get_model_cost(self, model_id: str) -> Dict:
        """Get cost information for a model"""
        
        if model_id in self.config.get("local_models", {}):
            return {
                "type": "local",
                "cost_per_1k": 0.0,
                "cost_per_query": 0.0
            }
        
        elif model_id in self.config.get("api_models", {}):
            config = self.config["api_models"][model_id]
            return {
                "type": "api",
                "cost_per_1k_input": config["cost_per_1k_input"],
                "cost_per_1k_output": config["cost_per_1k_output"]
            }
        
        return {}