"""
Real-time inference script for ISL recognition.

Usage:
    python ml/infer_realtime.py --checkpoint ml/models/checkpoints/best_model.pt --data_dir ml/datasets/processed
"""

import argparse
import logging
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn

from ml.models.bilstm_baseline import BiLSTMBaseline, BiLSTMConfig
from ml.inference.realtime.pipeline import RealtimeInferencePipeline, RealtimeConfig
from ml.inference.processors.holistic_feature_extractor import HolisticExtractionConfig
from ml.datasets.dataloader import ISLDataset


def setup_logging(log_level: str = 'INFO') -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ml/models/inference.log')
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Real-time ISL recognition inference')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input_dim', type=int, default=279,
                        help='Input feature dimension')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='ml/datasets/processed',
                        help='Directory containing processed dataset for class names')
    parser.add_argument('--train_split', type=str, default='train',
                        help='Training split directory name for class names')
    
    # Sliding window settings
    parser.add_argument('--window_size', type=int, default=30,
                        help='Number of frames in sliding window')
    parser.add_argument('--window_stride', type=int, default=1,
                        help='Stride for sliding window')
    
    # Temporal smoothing settings
    parser.add_argument('--smoothing_window', type=int, default=5,
                        help='Number of predictions to smooth')
    parser.add_argument('--smoothing_threshold', type=float, default=0.3,
                        help='Minimum confidence for smoothing')
    
    # Confidence threshold
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Minimum confidence to accept prediction')
    
    # FPS settings
    parser.add_argument('--target_fps', type=int, default=30,
                        help='Target FPS')
    parser.add_argument('--max_latency_ms', type=float, default=100.0,
                        help='Maximum acceptable latency in ms')
    
    # Display settings
    parser.add_argument('--display_landmarks', action='store_true', default=True,
                        help='Display landmarks on frame')
    parser.add_argument('--display_predictions', action='store_true', default=True,
                        help='Display predictions on frame')
    parser.add_argument('--display_fps', action='store_true', default=True,
                        help='Display FPS on frame')
    parser.add_argument('--display_confidence', action='store_true', default=True,
                        help='Display confidence scores')
    
    # Camera settings
    parser.add_argument('--camera_id', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--camera_width', type=int, default=640,
                        help='Camera width')
    parser.add_argument('--camera_height', type=int, default=480,
                        help='Camera height')
    
    # Feature extraction settings
    parser.add_argument('--model_complexity', type=int, default=1,
                        help='MediaPipe model complexity (0=Lite, 1=Full, 2=Heavy)')
    parser.add_argument('--min_detection_confidence', type=float, default=0.5,
                        help='MediaPipe minimum detection confidence')
    parser.add_argument('--min_tracking_confidence', type=float, default=0.5,
                        help='MediaPipe minimum tracking confidence')
    
    # System settings
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run inference on (cuda/cpu)')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """
    Load model from checkpoint.
    
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


def load_class_names(data_dir: str, train_split: str) -> list:
    """
    Load class names from dataset.
    
    Args:
        data_dir: Directory containing processed dataset
        train_split: Training split directory name
        
    Returns:
        List of class names
    """
    logger = logging.getLogger(__name__)
    
    data_path = Path(data_dir) / train_split
    sequences_path = data_path / 'sequences.npy'
    
    if not sequences_path.exists():
        logger.warning(f"Training sequences not found at {sequences_path}")
        logger.warning("Using default class names")
        return [f"Class_{i}" for i in range(25)]
    
    # Load dataset to get class names
    dataset = ISLDataset(sequences_path)
    class_names = [dataset.idx_to_label[i] for i in range(dataset.get_num_classes())]
    
    logger.info(f"Loaded {len(class_names)} class names from dataset")
    return class_names


def main():
    """Main inference function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Log configuration
    logger.info("Real-time inference configuration:")
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
    
    model, config, checkpoint_metadata = load_model(args.checkpoint, device)
    
    # Load class names
    class_names = load_class_names(args.data_dir, args.train_split)
    
    # Update num_classes from config if different
    if config.num_classes != len(class_names):
        logger.warning(f"Model has {config.num_classes} classes, dataset has {len(class_names)}")
        logger.warning("Using model's num_classes")
    
    # Create real-time config
    realtime_config = RealtimeConfig(
        window_size=args.window_size,
        window_stride=args.window_stride,
        smoothing_window=args.smoothing_window,
        smoothing_threshold=args.smoothing_threshold,
        confidence_threshold=args.confidence_threshold,
        target_fps=args.target_fps,
        max_latency_ms=args.max_latency_ms,
        display_landmarks=args.display_landmarks,
        display_predictions=args.display_predictions,
        display_fps=args.display_fps,
        display_confidence=args.display_confidence,
        camera_id=args.camera_id,
        camera_width=args.camera_width,
        camera_height=args.camera_height
    )
    
    # Create feature extraction config
    feature_config = HolisticExtractionConfig.get_default_isl_config()
    feature_config.model_complexity = args.model_complexity
    feature_config.min_detection_confidence = args.min_detection_confidence
    feature_config.min_tracking_confidence = args.min_tracking_confidence
    
    # Create pipeline
    pipeline = RealtimeInferencePipeline(
        model=model,
        class_names=class_names,
        config=realtime_config,
        feature_extractor_config=feature_config
    )
    
    logger.info("Real-time inference pipeline initialized")
    logger.info(f"Number of classes: {len(class_names)}")
    logger.info(f"Class names: {class_names}")
    logger.info(f"Window size: {args.window_size} frames")
    logger.info(f"Smoothing window: {args.smoothing_window} predictions")
    
    # Run inference
    try:
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Print statistics
        stats = pipeline.get_statistics()
        logger.info("\n" + "=" * 60)
        logger.info("Inference Statistics")
        logger.info("=" * 60)
        logger.info(f"Average FPS: {stats['fps']:.2f}")
        logger.info(f"Average Latency: {stats['latency_ms']:.2f} ms")
        logger.info(f"Total Predictions: {stats['total_predictions']}")
        logger.info(f"Valid Predictions: {stats['valid_predictions']}")
        logger.info(f"Buffer Fullness: {stats['buffer_fullness']:.2%}")


if __name__ == '__main__':
    main()
