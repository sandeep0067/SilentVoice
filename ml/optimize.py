"""
Hyperparameter optimization script using Optuna.

Usage:
    python ml/optimize.py --data_dir ml/datasets/processed --n_trials 50
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml.datasets.dataloader import ISLDataset
from ml.optimization.optuna_optimizer import OptunaOptimizer, OptimizationConfig
from ml.utils.seed import set_seed


def setup_logging(log_level: str = 'INFO') -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ml/models/optimization.log')
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Hyperparameter optimization with Optuna')
    
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
    
    # Optimization arguments
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of optimization trials')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Optimization timeout in seconds')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs')
    parser.add_argument('--study_name', type=str, default='bilstm_optimization',
                        help='Optuna study name')
    parser.add_argument('--direction', type=str, default='maximize',
                        choices=['maximize', 'minimize'],
                        help='Optimization direction')
    
    # Search space arguments
    parser.add_argument('--hidden_dim_min', type=int, default=32,
                        help='Minimum hidden dimension')
    parser.add_argument('--hidden_dim_max', type=int, default=256,
                        help='Maximum hidden dimension')
    parser.add_argument('--num_layers_min', type=int, default=1,
                        help='Minimum number of LSTM layers')
    parser.add_argument('--num_layers_max', type=int, default=4,
                        help='Maximum number of LSTM layers')
    parser.add_argument('--lr_min', type=float, default=1e-5,
                        help='Minimum learning rate')
    parser.add_argument('--lr_max', type=float, default=1e-2,
                        help='Maximum learning rate')
    parser.add_argument('--batch_sizes', type=str, default='16,32,64,128',
                        help='Comma-separated batch size choices')
    parser.add_argument('--dropout_min', type=float, default=0.1,
                        help='Minimum dropout rate')
    parser.add_argument('--dropout_max', type=float, default=0.5,
                        help='Maximum dropout rate')
    parser.add_argument('--weight_decay_min', type=float, default=1e-6,
                        help='Minimum weight decay')
    parser.add_argument('--weight_decay_max', type=float, default=1e-3,
                        help='Maximum weight decay')
    
    # Training arguments
    parser.add_argument('--epochs_per_trial', type=int, default=20,
                        help='Number of epochs per trial')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='Early stopping patience for trials')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for data loading (not optimization)')
    
    # System arguments
    parser.add_argument('--device', type=str, default=None,
                        help='Device to train on (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='ml/models/optuna_checkpoints',
                        help='Directory to save trial checkpoints')
    parser.add_argument('--log_dir', type=str, default='ml/models/optuna_logs',
                        help='Directory to save optimization logs')
    parser.add_argument('--optuna_storage', type=str, default='sqlite:///ml/models/optimization.db',
                        help='Optuna storage URL')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    return parser.parse_args()


def main():
    """Main optimization function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Set random seed
    set_seed(args.seed, deterministic=False)
    logger.info(f"Random seed set to {args.seed}")
    
    # Log configuration
    logger.info("Optimization configuration:")
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
    
    # Get number of classes
    num_classes = train_dataset.get_num_classes()
    logger.info(f"Number of classes: {num_classes}")
    
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
    
    # Parse batch size choices
    batch_size_choices = [int(x) for x in args.batch_sizes.split(',')]
    
    # Create optimization configuration
    opt_config = OptimizationConfig(
        hidden_dim_range=(args.hidden_dim_min, args.hidden_dim_max),
        num_layers_range=(args.num_layers_min, args.num_layers_max),
        learning_rate_range=(args.lr_min, args.lr_max),
        batch_size_choices=batch_size_choices,
        dropout_range=(args.dropout_min, args.dropout_max),
        weight_decay_range=(args.weight_decay_min, args.weight_decay_max),
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        study_name=args.study_name,
        direction=args.direction,
        epochs_per_trial=args.epochs_per_trial,
        early_stopping_patience=args.early_stopping_patience,
        optuna_storage=args.optuna_storage,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    # Create optimizer
    optimizer = OptunaOptimizer(
        config=opt_config,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        input_dim=args.input_dim,
        device=device
    )
    
    # Run optimization
    logger.info("Starting hyperparameter optimization...")
    logger.info(f"Number of trials: {args.n_trials}")
    logger.info(f"Optimization direction: {args.direction}")
    
    try:
        results = optimizer.optimize()
        
        # Print results
        logger.info("\n" + "=" * 60)
        logger.info("Optimization Results")
        logger.info("=" * 60)
        logger.info(f"Best trial: {results['best_trial_number']}")
        logger.info(f"Best value: {results['best_value']:.4f}")
        logger.info(f"Best parameters:")
        for param, value in results['best_params'].items():
            logger.info(f"  {param}: {value}")
        logger.info(f"Total trials completed: {results['n_trials']}")
        
        # Get optimization statistics
        stats = optimizer.get_optimization_history()
        logger.info("\n" + "=" * 60)
        logger.info("Optimization Statistics")
        logger.info("=" * 60)
        logger.info(f"Total trials: {stats['total_trials']}")
        logger.info(f"Completed trials: {stats['completed_trials']}")
        logger.info(f"Pruned trials: {stats['pruned_trials']}")
        logger.info(f"Failed trials: {stats['failed_trials']}")
        if stats['mean_value'] is not None:
            logger.info(f"Mean value: {stats['mean_value']:.4f}")
            logger.info(f"Std value: {stats['std_value']:.4f}")
        
        logger.info(f"\nResults saved to {args.log_dir}")
        logger.info(f"Checkpoints saved to {args.checkpoint_dir}")
        logger.info(f"Optuna database: {args.optuna_storage}")
        
        logger.info("\nTo visualize optimization results, run:")
        logger.info(f"  optuna dashboard {args.optuna_storage}")
        
    except KeyboardInterrupt:
        logger.info("\nOptimization interrupted by user")
        logger.info("Partial results have been saved")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
