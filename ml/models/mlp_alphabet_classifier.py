"""
MLP Classifier for ASL Alphabet Recognition.

PyTorch implementation of a feedforward neural network for single-frame
hand landmark classification (static alphabet signs).
"""

import json
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AlphabetMLPConfig:
    """Hyperparameters and configuration for Alphabet MLP Classifier."""
    input_dim: int = 63           # MediaPipe Hands: 21 landmarks × 3 coordinates (x,y,z)
    hidden_dims: list = None       # List of hidden layer dimensions
    num_classes: int = 29         # A-Z + SPACE + DELETE + NOTHING = 29 classes
    dropout: float = 0.3          # Dropout probability
    use_batch_norm: bool = True   # Use batch normalization
    activation: str = 'relu'      # Activation function: 'relu', 'leaky_relu', 'gelu'

    def __post_init__(self):
        """Set default hidden dimensions if not provided."""
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'AlphabetMLPConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class AlphabetMLP(nn.Module):
    """
    Feedforward Neural Network for ASL Alphabet Classification.
    
    Processes single-frame hand landmarks (63 features) to classify
    static alphabet signs (A-Z + special classes).
    """

    def __init__(self, config: Optional[AlphabetMLPConfig] = None):
        """
        Initialize Alphabet MLP Classifier.

        Args:
            config: AlphabetMLPConfig hyperparameters object
        """
        super().__init__()
        self.config = config or AlphabetMLPConfig()
        
        # Build hidden layers
        layers = []
        input_dim = self.config.input_dim
        
        for hidden_dim in self.config.hidden_dims:
            # Linear layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            # Batch normalization
            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            if self.config.activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif self.config.activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1, inplace=True))
            elif self.config.activation == 'gelu':
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU(inplace=True))
            
            # Dropout
            layers.append(nn.Dropout(self.config.dropout))
            
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, self.config.num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        logger = __import__('logging').getLogger(__name__)
        logger.info(f"Initialized AlphabetMLP with config: {self.config.to_dict()}")

    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (B, 63) where B is batch size

        Returns:
            Logits tensor of shape (B, num_classes)
        """
        return self.network(x)

    def get_config(self) -> AlphabetMLPConfig:
        """Get the model configuration."""
        return self.config

    def save_config(self, path: Path):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

    @classmethod
    def load_config(cls, path: Path) -> 'AlphabetMLPConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return AlphabetMLPConfig.from_dict(config_dict)


def create_alphabet_mlp(
    input_dim: int = 63,
    hidden_dims: list = None,
    num_classes: int = 29,
    dropout: float = 0.3,
    use_batch_norm: bool = True,
    activation: str = 'relu'
) -> AlphabetMLP:
    """
    Factory function to create an AlphabetMLP model.

    Args:
        input_dim: Input feature dimension (default: 63 for hand landmarks)
        hidden_dims: List of hidden layer dimensions
        num_classes: Number of output classes
        dropout: Dropout probability
        use_batch_norm: Whether to use batch normalization
        activation: Activation function

    Returns:
        Initialized AlphabetMLP model
    """
    config = AlphabetMLPConfig(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        activation=activation
    )
    return AlphabetMLP(config)


if __name__ == '__main__':
    # Test the model
    model = create_alphabet_mlp()
    x = torch.randn(32, 63)  # Batch of 32 samples
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
