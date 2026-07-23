"""
Configuration file loader for YAML and JSON files.
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        file_path: Path to the YAML file
    
    Returns:
        Dictionary containing the configuration
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON configuration file.
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        Dictionary containing the configuration
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(path, 'r') as f:
        return json.load(f)
