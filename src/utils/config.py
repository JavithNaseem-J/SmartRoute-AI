import yaml
from pathlib import Path
from typing import Any, Dict
import os
from dotenv import load_dotenv

load_dotenv()


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_env(key: str, default: Any = None) -> Any:
    """Get environment variable with default"""
    return os.getenv(key, default)


def ensure_dir(path: Path):
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)


class Config:
    """Central configuration manager"""
    
    def __init__(self, config_dir: Path = Path("config")):
        self.config_dir = config_dir
        
        # Load all configs
        self.models = load_yaml(config_dir / "models.yaml")
        self.routing = load_yaml(config_dir / "routing.yaml")
    
    def get_model_config(self, model_id: str) -> Dict:
        """Get configuration for a specific model"""
        
        if model_id in self.models.get('local_models', {}):
            return self.models['local_models'][model_id]
        
        if model_id in self.models.get('groq_models', {}):
            return self.models['groq_models'][model_id]
        
        if model_id in self.models.get('api_models', {}):
            return self.models['api_models'][model_id]
        
        raise ValueError(f"Unknown model: {model_id}")
    
    def get_routing_strategy(self, strategy: str) -> Dict:
        """Get routing strategy configuration"""
        
        if strategy not in self.routing.get('strategies', {}):
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return self.routing['strategies'][strategy]
    
    def get_budget_limits(self) -> Dict:
        """Get budget limits"""
        return self.routing.get('budgets', {})