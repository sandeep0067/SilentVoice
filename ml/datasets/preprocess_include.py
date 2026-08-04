"""
INCLUDE Dataset Preprocessing Script

This script preprocesses the INCLUDE dataset for ISL recognition training.
It handles video validation, MediaPipe landmark extraction, sequence generation,
and train/val/test splitting with subject-based separation.
"""

import argparse
import logging
import sys
from pathlib import Path
import json
import csv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_metadata(dataset_root: str, output_path: str):
    """
    Create a sample metadata file for INCLUDE dataset.
    
    Args:
        dataset_root: Path to INCLUDE dataset root
        output_path: Path to save metadata.csv
    """
    dataset_path = Path(dataset_root)
    output_file = Path(output_path)
    
    logger.info(f"Scanning dataset at {dataset_root}")
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(dataset_path.rglob(f'*{ext}'))
    
    if not video_files:
        logger.error(f"No video files found in {dataset_root}")
        return False
    
    logger.info(f"Found {len(video_files)} video files")
    
    # Create metadata entries
    metadata = []
    for i, video_path in enumerate(video_files, 1):
        # Extract relative path
        rel_path = video_path.relative_to(dataset_path)
        
        # Try to extract subject and gesture from path
        path_parts = rel_path.parts
        subject = 'unknown'
        gesture = 'unknown'
        
        # INCLUDE structure: Category/GestureName/video.mov
        if len(path_parts) >= 2:
            gesture = path_parts[-2]
            subject = video_path.stem
        
        metadata.append({
            'video_id': f'{i:04d}',
            'video_path': str(rel_path),
            'subject': subject,
            'gesture': gesture,
            'duration': 0.0,
            'frame_count': 0,
            'fps': 30.0,
            'resolution': '1920x1080',
            'quality': 'good',
            'lighting_condition': 'normal',
            'background': 'plain',
            'created_at': '2026-08-03T00:00:00',
            'split': 'train'
        })
    
    # Write metadata CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['video_id', 'video_path', 'subject', 'gesture', 'duration', 'frame_count', 'fps', 'resolution', 'quality', 'lighting_condition', 'background', 'created_at', 'split']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)
    
    logger.info(f"Created metadata file at {output_path} with {len(metadata)} entries")
    return True


def validate_dataset(dataset_root: str, metadata_path: str):
    """
    Validate the INCLUDE dataset structure.
    
    Args:
        dataset_root: Path to INCLUDE dataset root
        metadata_path: Path to metadata.csv file
    """
    dataset_path = Path(dataset_root)
    metadata_file = Path(metadata_path)
    
    logger.info("Validating dataset structure...")
    
    # Check dataset root exists
    if not dataset_path.exists():
        logger.error(f"Dataset root does not exist: {dataset_root}")
        return False
    
    # Check metadata file exists
    if not metadata_file.exists():
        logger.error(f"Metadata file does not exist: {metadata_file}")
        logger.info("Run with --create-metadata to generate a sample metadata file")
        return False
    
    # Read and validate metadata
    with open(metadata_file, 'r') as f:
        reader = csv.DictReader(f)
        metadata_rows = list(reader)
    
    logger.info(f"Found {len(metadata_rows)} entries in metadata file")
    
    # Validate video files exist
    missing_files = []
    for row in metadata_rows:
        video_path = dataset_path / row['video_path']
        if not video_path.exists():
            missing_files.append(row['video_id'])
    
    if missing_files:
        logger.warning(f"{len(missing_files)} video files are missing:")
        for video_id in missing_files[:10]:  # Show first 10
            logger.warning(f"  - {video_id}")
        if len(missing_files) > 10:
            logger.warning(f"  ... and {len(missing_files) - 10} more")
    else:
        logger.info("All video files found")
    
    # Check for unique video IDs
    video_ids = [row['video_id'] for row in metadata_rows]
    if len(video_ids) != len(set(video_ids)):
        logger.warning("Duplicate video IDs found in metadata")
    
    # Check for gesture labels
    gestures = set(row['gesture'] for row in metadata_rows)
    logger.info(f"Found {len(gestures)} unique gesture labels")
    if len(gestures) < 5:
        logger.warning(f"Only {len(gestures)} gestures found - recommend at least 10")
    
    logger.info("Dataset validation complete")
    return True


def preprocess_include_dataset(
    dataset_root: str,
    output_root: str,
    metadata_path: str,
    sequence_length: int = 30,
    stride: int = 15,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    apply_augmentation: bool = True,
    skip_existing: bool = True
):
    """
    Preprocess INCLUDE dataset for training.
    
    Args:
        dataset_root: Path to INCLUDE dataset root
        output_root: Path to save preprocessed data
        metadata_path: Path to metadata.csv file
        sequence_length: Length of sequences for training
        stride: Stride for sequence generation
        train_ratio: Ratio for training split
        val_ratio: Ratio for validation split
        test_ratio: Ratio for test split
        apply_augmentation: Whether to apply data augmentation
        skip_existing: Whether to skip already processed videos
    """
    try:
        from ml.datasets.preprocessing_pipeline import PreprocessingPipeline, PipelineConfig
    except ImportError:
        logger.error("Could not import preprocessing pipeline. Ensure all dependencies are installed.")
        return False
    
    logger.info("Starting INCLUDE dataset preprocessing...")
    logger.info(f"Dataset root: {dataset_root}")
    logger.info(f"Output root: {output_root}")
    logger.info(f"Sequence length: {sequence_length}, Stride: {stride}")
    
    # Create pipeline config
    config = PipelineConfig(
        dataset_root=dataset_root,
        output_root=output_root,
        sequence_length=sequence_length,
        stride=stride,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        apply_augmentation=apply_augmentation,
        skip_existing=skip_existing,
        min_duration=1.0,
        max_duration=5.0,
        min_fps=24.0,
        target_fps=30.0,
        apply_quality_filter=True,
        min_quality_level='medium',
        max_num_hands=2,
        include_z_coordinates=True,
        batch_size=32,
        num_workers=4
    )
    
    # Initialize pipeline
    try:
        pipeline = PreprocessingPipeline(config)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return False
    
    # Run preprocessing
    try:
        statistics = pipeline.run_full_pipeline()
        logger.info("Preprocessing completed successfully!")
        logger.info(f"Statistics: {json.dumps(statistics, indent=2)}")
        return True
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        pipeline.cleanup()


def main():
    """Main entry point for INCLUDE dataset preprocessing."""
    parser = argparse.ArgumentParser(
        description='Preprocess INCLUDE dataset for ISL recognition training'
    )
    
    parser.add_argument(
        '--dataset-root',
        type=str,
        default='ml/datasets/raw/INCLUDE',
        help='Path to INCLUDE dataset root directory'
    )
    
    parser.add_argument(
        '--output-root',
        type=str,
        default='ml/datasets/processed',
        help='Path to save preprocessed data'
    )
    
    parser.add_argument(
        '--metadata-path',
        type=str,
        default='ml/datasets/raw/metadata.csv',
        help='Path to metadata.csv file'
    )
    
    parser.add_argument(
        '--create-metadata',
        action='store_true',
        help='Create a sample metadata file from dataset structure'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate dataset without preprocessing'
    )
    
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=30,
        help='Length of sequences for training (default: 30)'
    )
    
    parser.add_argument(
        '--stride',
        type=int,
        default=15,
        help='Stride for sequence generation (default: 15)'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Ratio for training split (default: 0.7)'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Ratio for validation split (default: 0.15)'
    )
    
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Ratio for test split (default: 0.15)'
    )
    
    parser.add_argument(
        '--no-augmentation',
        action='store_true',
        help='Disable data augmentation'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Reprocess all videos (skip existing disabled)'
    )
    
    args = parser.parse_args()
    
    # Create metadata if requested
    if args.create_metadata:
        logger.info("Creating sample metadata file...")
        success = create_sample_metadata(args.dataset_root, args.metadata_path)
        if success:
            logger.info("Metadata file created successfully")
            logger.info("Please review and edit the metadata file before preprocessing")
            logger.info("Then run this script again without --create-metadata")
        return
    
    # Validate dataset
    if not validate_dataset(args.dataset_root, args.metadata_path):
        logger.error("Dataset validation failed")
        sys.exit(1)
    
    # If validate only, exit here
    if args.validate_only:
        logger.info("Validation complete. Exiting.")
        return
    
    # Run preprocessing
    success = preprocess_include_dataset(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        metadata_path=args.metadata_path,
        sequence_length=args.sequence_length,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        apply_augmentation=not args.no_augmentation,
        skip_existing=not args.force
    )
    
    if success:
        logger.info("=" * 60)
        logger.info("INCLUDE dataset preprocessing completed successfully!")
        logger.info("=" * 60)
        logger.info(f"Preprocessed data saved to: {args.output_root}")
        logger.info(f"  - Landmarks: {args.output_root}/landmarks/")
        logger.info(f"  - Sequences: {args.output_root}/sequences/")
        logger.info(f"  - Quality reports: {args.output_root}/quality_reports/")
        logger.info(f"  - Statistics: {args.output_root}/pipeline_statistics.json")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Review the preprocessing statistics")
        logger.info("2. Check quality reports for any issues")
        logger.info("3. Run training: python ml/train.py")
        sys.exit(0)
    else:
        logger.error("Preprocessing failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
