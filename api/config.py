"""
Configuration management for FastAPI backend.

Handles environment variables, model paths, and application settings.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = "SilentVoice API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Model settings
    model_checkpoint_path: str = Field(
        default="ml/models/checkpoints/best_model.pt",
        description="Path to trained model checkpoint"
    )
    model_device: str = Field(
        default="cuda",
        description="Device to run model on (cuda/cpu)"
    )
    
    # Feature extraction settings
    feature_dim: int = Field(
        default=279,
        description="Input feature dimension"
    )
    model_complexity: int = Field(
        default=1,
        description="MediaPipe model complexity (0=Lite, 1=Full, 2=Heavy)"
    )
    
    # Inference settings
    confidence_threshold: float = Field(
        default=0.5,
        description="Minimum confidence threshold for predictions"
    )
    batch_size: int = Field(
        default=1,
        description="Batch size for inference"
    )
    
    # Translation settings
    label_mapping_path: Optional[str] = Field(
        default=None,
        description="Path to label mapping JSON file"
    )
    default_language: str = Field(
        default="en",
        description="Default language for translations"
    )
    
    # TTS settings
    enable_tts: bool = Field(
        default=False,
        description="Enable text-to-speech"
    )
    tts_language: str = Field(
        default="en",
        description="TTS language"
    )
    
    # API settings
    api_host: str = Field(
        default="0.0.0.0",
        description="API host"
    )
    api_port: int = Field(
        default=8000,
        description="API port"
    )
    api_prefix: str = Field(
        default="/api/v1",
        description="API URL prefix"
    )
    
    # CORS settings
    cors_origins: list = Field(
        default=["*"],
        description="CORS allowed origins"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def get_model_path(self) -> Path:
        """Get absolute path to model checkpoint."""
        return Path(self.model_checkpoint_path).absolute()
    
    def get_label_mapping_path(self) -> Optional[Path]:
        """Get absolute path to label mapping file."""
        if self.label_mapping_path:
            return Path(self.label_mapping_path).absolute()
        return None


# Global settings instance
settings = Settings()
