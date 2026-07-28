"""
MediaPipe Configuration Manager.

Manages MediaPipe landmark extraction configuration.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class MediaPipeConfig:
    """MediaPipe configuration."""
    # Hands
    max_num_hands: int = 2
    hands_model_complexity: int = 1
    hands_min_detection_confidence: float = 0.5
    hands_min_tracking_confidence: float = 0.5
    
    # Face
    max_num_faces: int = 1
    refine_landmarks: bool = True
    face_min_detection_confidence: float = 0.5
    face_min_tracking_confidence: float = 0.5
    
    # Pose
    enable_pose: bool = False
    pose_model_complexity: int = 1
    pose_min_detection_confidence: float = 0.5
    pose_min_tracking_confidence: float = 0.5
    
    # Output
    include_z_coordinates: bool = True
    normalize_coordinates: bool = True
    use_relative_coordinates: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MediaPipeConfig':
        """Create config from dictionary."""
        return cls(**{
            k: v for k, v in config_dict.items()
            if k in cls.__dataclass_fields__
        })
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'MediaPipeConfig':
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save config to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


class MediaPipeConfigManager:
    """Manages MediaPipe configurations."""
    
    DEFAULT_CONFIG = MediaPipeConfig()
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config manager.
        
        Args:
            config_path: Path to config file
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = self._load_config()
    
    def _load_config(self) -> MediaPipeConfig:
        """Load configuration from file or use default."""
        if self.config_path and self.config_path.exists():
            return MediaPipeConfig.from_yaml(str(self.config_path))
        return self.DEFAULT_CONFIG
    
    def save_config(self, output_path: str) -> None:
        """
        Save current configuration.
        
        Args:
            output_path: Path to save configuration
        """
        self.config.to_yaml(output_path)
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def get_config(self) -> MediaPipeConfig:
        """Get current configuration."""
        return self.config
    
    def reset_to_default(self) -> None:
        """Reset configuration to default."""
        self.config = self.DEFAULT_CONFIG
