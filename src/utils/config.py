"""
Configuration management for AdBot
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    host: str = Field(default="localhost", alias="DB_HOST")
    port: int = Field(default=7500, alias="DB_PORT")
    name: str = Field(default="adbot_dev", alias="DB_NAME")
    user: str = Field(default="adbot", alias="DB_USER")
    password: str = Field(alias="DB_PASSWORD")
    pool_size: int = Field(default=10, alias="DB_POOL_SIZE")

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def async_url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisConfig(BaseSettings):
    host: str = Field(default="localhost", alias="REDIS_HOST")
    port: int = Field(default=7501, alias="REDIS_PORT")
    db: int = Field(default=0, alias="REDIS_DB")
    password: Optional[str] = Field(default=None, alias="REDIS_PASSWORD")

    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class RLConfig(BaseSettings):
    algorithm: str = Field(default="PPO", alias="RL_ALGORITHM")
    learning_rate: float = Field(default=0.0003, alias="RL_LEARNING_RATE")
    batch_size: int = Field(default=64, alias="RL_BATCH_SIZE")
    n_epochs: int = Field(default=10, alias="RL_N_EPOCHS")
    gamma: float = Field(default=0.99, alias="RL_GAMMA")
    gae_lambda: float = Field(default=0.95, alias="RL_GAE_LAMBDA")
    clip_range: float = Field(default=0.2, alias="RL_CLIP_RANGE")
    n_steps: int = Field(default=2048, alias="RL_N_STEPS")
    ent_coef: float = Field(default=0.01, alias="RL_ENT_COEF")
    vf_coef: float = Field(default=0.5, alias="RL_VF_COEF")
    max_grad_norm: float = Field(default=0.5, alias="RL_MAX_GRAD_NORM")


class MLConfig(BaseSettings):
    tracking_uri: str = Field(default="http://localhost:7502", alias="MLFLOW_TRACKING_URI")
    experiment_name: str = Field(default="adbot_experiments", alias="MLFLOW_EXPERIMENT_NAME")
    artifact_root: str = Field(default="./mlruns", alias="MLFLOW_ARTIFACT_ROOT")


class SecurityConfig(BaseSettings):
    jwt_secret: str = Field(alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expiration_minutes: int = Field(default=60, alias="JWT_EXPIRATION_MINUTES")


class APIConfig(BaseSettings):
    host: str = Field(default="0.0.0.0", alias="API_HOST")
    port: int = Field(default=7507, alias="API_PORT")
    cors_origins: list[str] = Field(default=["http://localhost:3000"], alias="CORS_ORIGINS")
    rate_limit_enabled: bool = Field(default=True, alias="RATE_LIMIT_ENABLED")
    rate_limit_per_minute: int = Field(default=60, alias="RATE_LIMIT_PER_MINUTE")


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    name: str = Field(default="AdBot", alias="APP_NAME")
    version: str = Field(default="0.1.0", alias="APP_VERSION")
    environment: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=True, alias="APP_DEBUG")
    
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    api: APIConfig = Field(default_factory=APIConfig)


class ConfigManager:
    """Configuration manager for AdBot"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self._config_data: Dict[str, Any] = {}
        self._app_config: Optional[AppConfig] = None
        
    def _find_config_file(self) -> str:
        """Find the configuration file"""
        possible_paths = [
            "configs/default.yaml",
            "configs/production.yaml",
            "configs/development.yaml",
            "default.yaml",
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
                
        raise FileNotFoundError("No configuration file found")
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self._config_data:
            with open(self.config_path, "r") as f:
                self._config_data = yaml.safe_load(f)
        return self._config_data
    
    def get_app_config(self) -> AppConfig:
        """Get application configuration with environment variables"""
        if not self._app_config:
            self._app_config = AppConfig()
        return self._app_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        config = self.load_config()
        keys = key.split(".")
        
        value = config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def reload(self) -> None:
        """Reload configuration from file"""
        self._config_data = {}
        self._app_config = None