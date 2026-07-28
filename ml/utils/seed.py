"""
Random seed control for reproducibility.
"""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic algorithms (may impact performance)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_seed() -> Optional[int]:
    """
    Get current random seed state (for logging purposes).
    
    Returns:
        Current seed if set, None otherwise
    """
    try:
        return torch.initial_seed()
    except RuntimeError:
        return None
