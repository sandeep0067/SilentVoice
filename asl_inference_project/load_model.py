"""
Model Loading and Testing Script for ASL Alphabet Recognition.

This script loads the trained best_model.pt checkpoint and verifies it loads correctly
by printing parameter count and running a dummy forward pass with random 63-dim input.
"""

import torch
from pathlib import Path
from model import AlphabetMLP, AlphabetMLPConfig


def load_and_test_model():
    """Load the trained model and run verification tests."""
    
    # Paths
    checkpoint_path = Path('../best_model.pt')
    config_path = Path('../model_config.json')
    
    print("=" * 60)
    print("ASL Alphabet Recognition Model Loading and Testing")
    print("=" * 60)
    
    # Check if files exist
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please ensure best_model.pt is in the models/ folder")
        return False
    
    print(f"\n1. Loading checkpoint from: {checkpoint_path}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"2. Using device: {device}")
    
    # Load model config
    if config_path.exists():
        print(f"3. Loading config from: {config_path}")
        model_config = AlphabetMLPConfig.load_config(config_path)
        print(f"   Config: {model_config.to_dict()}")
    else:
        print(f"3. Config file not found, using default configuration")
        model_config = AlphabetMLPConfig()
    
    # Create model
    print("4. Creating model architecture...")
    model = AlphabetMLP(model_config)
    
    # Load checkpoint
    print("5. Loading model weights from checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check checkpoint structure
    print(f"   Checkpoint keys: {list(checkpoint.keys())}")
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("   [OK] Model state dict loaded successfully")
    else:
        print("   WARNING: 'model_state_dict' key not found in checkpoint")
        print("   Attempting to load checkpoint directly as state dict...")
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n6. Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Print model architecture
    print(f"\n7. Model Architecture:")
    print(model)
    
    # Run dummy forward pass
    print(f"\n8. Running dummy forward pass with random input...")
    dummy_input = torch.randn(1, 63).to(device)  # Single sample, 63 features
    print(f"   Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"   Output shape: {output.shape}")
    print(f"   Output logits (first 5 classes): {output[0, :5].tolist()}")
    
    # Apply softmax to get probabilities
    probabilities = torch.softmax(output, dim=1)
    print(f"   Probabilities (first 5 classes): {probabilities[0, :5].tolist()}")
    
    # Get predicted class
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = torch.max(probabilities, dim=1).values.item()
    
    print(f"   Predicted class: {predicted_class}")
    print(f"   Confidence: {confidence:.4f}")
    
    # Load label mapping if available
    if 'label_mapping' in checkpoint:
        label_mapping = checkpoint['label_mapping']
        print(f"\n9. Label mapping found in checkpoint:")
        for idx, label in label_mapping.items():
            print(f"   Class {idx}: {label}")
        
        predicted_label = label_mapping.get(predicted_class, f"Class_{predicted_class}")
        print(f"   Predicted label: {predicted_label}")
    else:
        print(f"\n9. No label mapping in checkpoint, using default classes")
        # Default classes: A-Z + 'del' + 'nothing' + 'space' (sorted alphabetically)
        default_labels = sorted(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['del', 'nothing', 'space'])
        print(f"   Default classes: {default_labels}")
        if predicted_class < len(default_labels):
            predicted_label = default_labels[predicted_class]
            print(f"   Predicted label: {predicted_label}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Model loaded and tested successfully!")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    success = load_and_test_model()
    exit(0 if success else 1)
