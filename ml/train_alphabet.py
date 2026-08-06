"""
Training script for ASL Alphabet Classifier.

Usage:
    python ml/train_alphabet.py --data_dir ml/datasets/processed_asl --epochs 50 --batch_size 32
"""

import argparse
import logging
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from ml.models.mlp_alphabet_classifier import AlphabetMLP, AlphabetMLPConfig
from ml.training.trainer import BaselineTrainer
from ml.utils.seed import set_seed


class ASLAlphabetDataset(Dataset):
    """PyTorch Dataset for ASL Alphabet preprocessed data."""

    def __init__(self, data_dir: Path):
        """
        Initialize ASL Alphabet dataset.

        Args:
            data_dir: Path to directory containing landmarks.npy and metadata.json
        """
        self.data_dir = Path(data_dir)
        
        # Load landmarks
        landmarks_path = self.data_dir / 'landmarks.npy'
        if not landmarks_path.exists():
            raise FileNotFoundError(f"Landmarks file not found: {landmarks_path}")
        
        self.landmarks = np.load(landmarks_path)
        
        # Load metadata
        metadata_path = self.data_dir / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.labels = self.metadata['labels']
        self.num_classes = self.metadata['num_classes']
        
        # Create label to index mapping
        unique_labels = sorted(list(set(self.labels)))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Convert labels to indices
        self.label_indices = [self.label_to_idx[label] for label in self.labels]
        
        logger = logging.getLogger(__name__)
        logger.info(f"Loaded {len(self.landmarks)} samples from {data_dir}")
        logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Feature dimension: {self.landmarks.shape[1]}")

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        landmark = torch.FloatTensor(self.landmarks[idx])
        label = torch.LongTensor([self.label_indices[idx]])
        return landmark, label.squeeze()

    def get_num_classes(self):
        return self.num_classes

    def get_class_names(self):
        return list(self.label_to_idx.keys())


def setup_logging(log_level: str = 'INFO') -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ml/models/alphabet_training.log')
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ASL Alphabet classifier')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='ml/datasets/processed_asl',
                        help='Directory containing processed ASL alphabet dataset')
    parser.add_argument('--train_split', type=str, default='train',
                        help='Training split directory name')
    parser.add_argument('--val_split', type=str, default='val',
                        help='Validation split directory name')
    
    # Model arguments
    parser.add_argument('--input_dim', type=int, default=63,
                        help='Input feature dimension (63 for hand landmarks)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128, 64],
                        help='Hidden layer dimensions (space-separated)')
    parser.add_argument('--num_classes', type=int, default=29,
                        help='Number of alphabet classes (A-Z + SPACE + DELETE + NOTHING)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability')
    parser.add_argument('--use_batch_norm', action='store_true', default=True,
                        help='Use batch normalization')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'leaky_relu', 'gelu'],
                        help='Activation function')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Scheduler arguments
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['none', 'step', 'cosine', 'plateau'],
                        help='Learning rate scheduler type')
    parser.add_argument('--scheduler_t_max', type=int, default=50,
                        help='T_max for CosineAnnealingLR')
    parser.add_argument('--scheduler_step_size', type=int, default=10,
                        help='Step size for StepLR')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1,
                        help='Gamma for StepLR and ReduceLROnPlateau')
    
    # System arguments
    parser.add_argument('--device', type=str, default=None,
                        help='Device to train on (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Enable deterministic algorithms for reproducibility')
    
    # Checkpoint and logging
    parser.add_argument('--checkpoint_dir', type=str, default='ml/models/alphabet_checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='ml/models/alphabet_logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    return parser.parse_args()


def get_scheduler(
    scheduler_type: str,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace
):
    """Create learning rate scheduler."""
    if scheduler_type == 'none':
        return None
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_gamma
        )
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.scheduler_t_max
        )
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.scheduler_gamma,
            patience=5,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Set random seed
    set_seed(args.seed, deterministic=args.deterministic)
    logger.info(f"Random seed set to {args.seed}")
    
    # Log configuration
    logger.info("Training configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Setup device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load datasets
    data_dir = Path(args.data_dir)
    train_dir = data_dir / args.train_split
    val_dir = data_dir / args.val_split
    
    if not train_dir.exists():
        logger.error(f"Training directory not found: {train_dir}")
        sys.exit(1)
    if not val_dir.exists():
        logger.error(f"Validation directory not found: {val_dir}")
        sys.exit(1)
    
    logger.info(f"Loading training data from {train_dir}")
    logger.info(f"Loading validation data from {val_dir}")
    
    train_dataset = ASLAlphabetDataset(train_dir)
    val_dataset = ASLAlphabetDataset(val_dir)
    
    # Update num_classes from dataset if not specified
    if args.num_classes == 29:
        args.num_classes = train_dataset.get_num_classes()
        logger.info(f"Detected {args.num_classes} classes from dataset")
    
    # Log class names
    class_names = train_dataset.get_class_names()
    logger.info(f"Classes: {', '.join(class_names)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=False
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Number of classes: {args.num_classes}")
    
    # Create model
    config = AlphabetMLPConfig(
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        num_classes=args.num_classes,
        dropout=args.dropout,
        use_batch_norm=args.use_batch_norm,
        activation=args.activation
    )
    
    model = AlphabetMLP(config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Save model config
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_config(checkpoint_dir / 'model_config.json')
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    scheduler = get_scheduler(args.scheduler, optimizer, args)
    if scheduler:
        logger.info(f"Using {args.scheduler} learning rate scheduler")
    
    # Create trainer
    trainer = BaselineTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        early_stopping_patience=args.patience,
        use_amp=args.use_amp,
        resume_from=args.resume_from
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.fit(train_loader, val_loader, epochs=args.epochs)
    
    # Log final results
    logger.info("Training completed!")
    logger.info(f"Final training loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"Final training accuracy: {history['train_acc'][-1]:.2f}%")
    logger.info(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    logger.info(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%")
    logger.info(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    
    logger.info(f"Checkpoints saved to {args.checkpoint_dir}")
    logger.info(f"TensorBoard logs saved to {args.log_dir}")
    logger.info("To view TensorBoard logs, run: tensorboard --logdir ml/models/alphabet_logs")


if __name__ == '__main__':
    main()
