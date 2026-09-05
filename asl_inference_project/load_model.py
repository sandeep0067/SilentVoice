"""
Model Loading and Testing Script for ASL Alphabet Recognition.

This script loads the trained best_model.pt checkpoint and verifies it loads correctly.
"""

import torch
from pathlib import Path
from model import AlphabetMLP, AlphabetMLPConfig


def load_and_test_model():
    """Load the trained model and run verification tests."""
    
    checkpoint_path = Path('../best_model.pt')
    config_path = Path('../model_config.json')
    
    print("=" * 60)
    print("ASL Alphabet Recognition Model Loading and Testing")
    print("=" * 60)
    
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return False
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if config_path.exists():
        model_config = AlphabetMLPConfig.load_config(config_path)
    else:
        model_config = AlphabetMLPConfig()
    
    model = AlphabetMLP(model_config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 63).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    if 'label_mapping' in checkpoint:
        label_mapping = checkpoint['label_mapping']
        print(f"Classes: {list(label_mapping.values())}")
    
    print("=" * 60)
    print("Model loaded and tested successfully!")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    success = load_and_test_model()
    exit(0 if success else 1)
