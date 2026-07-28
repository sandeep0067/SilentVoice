"""
Production-ready evaluation script for ISL recognition model.

Usage:
    python ml/evaluate.py --checkpoint ml/models/checkpoints/best_model.pt --data_dir ml/datasets/processed
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.models.bilstm_baseline import BiLSTMBaseline, BiLSTMConfig
from ml.datasets.dataloader import ISLDataset
from ml.evaluation.metrics import EvaluationMetrics, evaluate_model
from ml.evaluation.visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_per_class_metrics,
    plot_misclassified_distribution,
    generate_evaluation_report
)


def setup_logging(log_level: str = 'INFO') -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ml/models/evaluation.log')
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate ISL recognition model')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
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
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='ml/datasets/processed',
                        help='Directory containing processed dataset')
    parser.add_argument('--test_split', type=str, default='test',
                        help='Test split directory name')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to evaluate on (cuda/cpu)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='ml/models/evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--plot_confusion', action='store_true', default=True,
                        help='Generate confusion matrix plots')
    parser.add_argument('--plot_roc', action='store_true', default=True,
                        help='Generate ROC curve plots')
    parser.add_argument('--plot_per_class', action='store_true', default=True,
                        help='Generate per-class metrics plots')
    parser.add_argument('--plot_misclassified', action='store_true', default=True,
                        help='Generate misclassification analysis')
    parser.add_argument('--export_csv', action='store_true', default=True,
                        help='Export metrics to CSV')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> tuple:
    """
    Load model and configuration from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Tuple of (model, config, metadata)
    """
    logger = logging.getLogger(__name__)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load config from checkpoint if available
    if 'model_config' in checkpoint and checkpoint['model_config'] is not None:
        config_dict = checkpoint['model_config']
        config = BiLSTMConfig.from_dict(config_dict)
        logger.info("Loaded model configuration from checkpoint")
    else:
        # Use default config
        config = BiLSTMConfig()
        logger.info("Using default model configuration")
    
    # Create model
    model = BiLSTMBaseline(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from {checkpoint_path}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    return model, config, checkpoint


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Log configuration
    logger.info("Evaluation configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Setup device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load model
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    model, config, checkpoint_metadata = load_model_from_checkpoint(args.checkpoint, device)
    
    # Update num_classes from config
    num_classes = config.num_classes
    
    # Load test dataset
    data_dir = Path(args.data_dir)
    test_dir = data_dir / args.test_split
    test_sequences = test_dir / 'sequences.npy'
    
    if not test_sequences.exists():
        logger.error(f"Test sequences not found: {test_sequences}")
        sys.exit(1)
    
    logger.info(f"Loading test data from {test_sequences}")
    
    test_dataset = ISLDataset(test_sequences)
    
    # Update num_classes from dataset if different
    dataset_num_classes = test_dataset.get_num_classes()
    if dataset_num_classes != num_classes:
        logger.warning(f"Dataset has {dataset_num_classes} classes, model has {num_classes}")
        num_classes = dataset_num_classes
    
    # Get class names
    class_names = [test_dataset.idx_to_label[i] for i in range(num_classes)]
    logger.info(f"Class names: {class_names}")
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=False
    )
    
    logger.info(f"Test samples: {len(test_dataset)}")
    logger.info(f"Number of classes: {num_classes}")
    
    # Evaluate model
    logger.info("Starting evaluation...")
    evaluator, metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        num_classes=num_classes,
        class_names=class_names
    )
    
    # Print classification report
    logger.info("\n" + "=" * 60)
    logger.info("Classification Report")
    logger.info("=" * 60)
    logger.info("\n" + evaluator.get_classification_report())
    
    # Print summary metrics
    logger.info("\n" + "=" * 60)
    logger.info("Summary Metrics")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    logger.info(f"Recall (Macro): {metrics['recall_macro']:.4f}")
    logger.info(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
    logger.info(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
    logger.info(f"Recall (Weighted): {metrics['recall_weighted']:.4f}")
    logger.info(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    
    if 'roc_auc_macro' in metrics:
        logger.info(f"ROC AUC (Macro): {metrics['roc_auc_macro']:.4f}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export metrics to CSV
    if args.export_csv:
        csv_path = output_dir / 'metrics_summary.csv'
        evaluator.export_to_csv(str(csv_path))
    
    # Get misclassified samples
    misclassified = evaluator.get_misclassified_samples()
    logger.info(f"\nTotal misclassified samples: {len(misclassified)}")
    
    if misclassified:
        # Save misclassified samples to JSON
        misclassified_path = output_dir / 'misclassified_samples.json'
        import json
        with open(misclassified_path, 'w') as f:
            json.dump(misclassified, f, indent=2)
        logger.info(f"Misclassified samples saved to {misclassified_path}")
    
    # Generate plots
    logger.info("\nGenerating evaluation plots...")
    
    generate_evaluation_report(
        metrics=metrics,
        output_dir=str(output_dir / 'plots'),
        plot_confusion=args.plot_confusion,
        plot_roc=args.plot_roc,
        plot_per_class=args.plot_per_class,
        plot_misclassified=args.plot_misclassified,
        misclassified_samples=misclassified if args.plot_misclassified else None
    )
    
    # Save complete metrics to JSON
    metrics_json = {
        'summary': {
            'accuracy': float(metrics['accuracy']),
            'precision_macro': float(metrics['precision_macro']),
            'recall_macro': float(metrics['recall_macro']),
            'f1_macro': float(metrics['f1_macro']),
            'precision_weighted': float(metrics['precision_weighted']),
            'recall_weighted': float(metrics['recall_weighted']),
            'f1_weighted': float(metrics['f1_weighted'])
        },
        'per_class': metrics['per_class_metrics'],
        'class_names': metrics['class_names'],
        'num_misclassified': len(misclassified),
        'total_samples': len(evaluator.all_labels)
    }
    
    if 'roc_auc_macro' in metrics:
        metrics_json['summary']['roc_auc_macro'] = float(metrics['roc_auc_macro'])
    
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    logger.info(f"\nEvaluation completed!")
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"  - Metrics summary: {csv_path if args.export_csv else 'N/A'}")
    logger.info(f"  - Metrics JSON: {metrics_path}")
    logger.info(f"  - Plots: {output_dir / 'plots'}")
    logger.info(f"  - Misclassified samples: {misclassified_path if misclassified else 'N/A'}")


if __name__ == '__main__':
    main()
