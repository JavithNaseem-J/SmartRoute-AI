"""
Configuration management
Uses Pydantic for environment variables
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Environment variables with validation"""
    
    # API Keys
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    
    # Model Settings
    local_model_path: str = Field("models/local", env="LOCAL_MODEL_PATH")
    use_gpu: bool = Field(True, env="USE_GPU")
    gpu_memory_fraction: float = Field(0.8, env="GPU_MEMORY_FRACTION")
    
    # Database
    database_url: str = Field("sqlite:///data/costs/usage.db", env="DATABASE_URL")
    
    # Budgets
    daily_budget: float = Field(10.0, env="DAILY_BUDGET")
    weekly_budget: float = Field(50.0, env="WEEKLY_BUDGET")
    monthly_budget: float = Field(200.0, env="MONTHLY_BUDGET")
    
    # API
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_workers: int = Field(4, env="API_WORKERS")
    
    # Monitoring
    mlflow_tracking_uri: str = Field("http://localhost:5000", env="MLFLOW_TRACKING_URI")
    prometheus_port: int = Field(9090, env="PROMETHEUS_PORT")
    enable_monitoring: bool = Field(True, env="ENABLE_MONITORING")
    
    # Routing
    default_strategy: str = Field("cost_optimized", env="DEFAULT_STRATEGY")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: str = Field("logs/app.log", env="LOG_FILE")
    
    # Development
    debug: bool = Field(False, env="DEBUG")
    reload: bool = Field(True, env="RELOAD")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class ConfigManager:
    """Manage YAML configurations"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._configs: Dict[str, Dict] = {}
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load a YAML config file"""
        if config_name in self._configs:
            return self._configs[config_name]
        
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self._configs[config_name] = config
        return config
    
    def get(self, config_name: str, key: str, default: Any = None) -> Any:
        """Get nested config value using dot notation"""
        config = self.load_config(config_name)
        keys = key.split('.')
        
        value = config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


@lru_cache()
def get_config_manager() -> ConfigManager:
    """Get cached config manager"""
    return ConfigManager()


def load_model_config() -> Dict:
    """Load model configuration"""
    return get_config_manager().load_config("models")


def load_routing_config() -> Dict:
    """Load routing configuration"""
    return get_config_manager().load_config("routing")


def load_app_config() -> Dict:
    """Load application configuration"""
    return get_config_manager().load_config("config")