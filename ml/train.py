"""
Production-ready training script for ISL recognition model.

Usage:
    python ml/train.py --data_dir ml/datasets/processed --epochs 50 --batch_size 32
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.models.bilstm_baseline import BiLSTMBaseline, BiLSTMConfig
from ml.training.trainer import BaselineTrainer
from ml.datasets.dataloader import ISLDataset, get_dataloader
from ml.utils.seed import set_seed


def setup_logging(log_level: str = 'INFO') -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ml/models/training.log')
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ISL recognition model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='ml/datasets/processed',
                        help='Directory containing processed dataset')
    parser.add_argument('--train_split', type=str, default='train',
                        help='Training split directory name')
    parser.add_argument('--val_split', type=str, default='val',
                        help='Validation split directory name')
    
    # Model arguments
    parser.add_argument('--input_dim', type=int, default=279,
                        help='Input feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='LSTM hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--num_classes', type=int, default=25,
                        help='Number of gesture classes')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                        help='Use bidirectional LSTM')
    parser.add_argument('--pooling_type', type=str, default='mean',
                        choices=['mean', 'max', 'last', 'attention'],
                        help='Temporal pooling strategy')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
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
    parser.add_argument('--checkpoint_dir', type=str, default='ml/models/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='ml/models/logs',
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
    
    train_sequences = train_dir / 'sequences.npy'
    val_sequences = val_dir / 'sequences.npy'
    
    if not train_sequences.exists():
        logger.error(f"Training sequences not found: {train_sequences}")
        sys.exit(1)
    if not val_sequences.exists():
        logger.error(f"Validation sequences not found: {val_sequences}")
        sys.exit(1)
    
    logger.info(f"Loading training data from {train_sequences}")
    logger.info(f"Loading validation data from {val_sequences}")
    
    train_dataset = ISLDataset(train_sequences)
    val_dataset = ISLDataset(val_sequences)
    
    # Update num_classes from dataset if not specified
    if args.num_classes == 25:
        args.num_classes = train_dataset.get_num_classes()
        logger.info(f"Detected {args.num_classes} classes from dataset")
    
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
    config = BiLSTMConfig(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        pooling_type=args.pooling_type
    )
    
    model = BiLSTMBaseline(config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
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
    logger.info("To view TensorBoard logs, run: tensorboard --logdir ml/models/logs")


if __name__ == '__main__':
    main()
