"""
Preprocessing Pipeline Orchestrator for INCLUDE dataset.

Coordinates the entire preprocessing pipeline from raw videos to training sequences.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

from ml.datasets.metadata_manager import MetadataManager, VideoMetadata
from ml.datasets.dataset_validator import DatasetValidator
from ml.datasets.video_preprocessor import VideoPreprocessor
from ml.datasets.frame_extractor import FrameExtractor, FrameExtractionConfig
from ml.datasets.quality_filter import QualityFilter
from ml.inference.processors.landmark_extractor import LandmarkExtractor, LandmarkExtractionConfig
from ml.inference.processors.sequence_builder import SequenceBuilder, SequenceBuilderConfig
from ml.inference.processors.data_augmentation import DataAugmenter, AugmentationConfig
from ml.datasets.sequence_generator import SequenceGenerator, SequenceGeneratorConfig
from ml.utils.file_helpers import ensure_directory, find_video_files


@dataclass
class PipelineConfig:
    """Configuration for the preprocessing pipeline."""
    # Input/Output
    dataset_root: str
    output_root: str
    
    # Video preprocessing
    min_duration: float = 1.0
    max_duration: float = 5.0
    min_fps: float = 24.0
    target_fps: float = 30.0
    
    # Quality filtering
    apply_quality_filter: bool = True
    min_quality_level: str = "medium"
    
    # Landmark extraction
    max_num_hands: int = 2
    include_z_coordinates: bool = True
    
    # Sequence generation
    sequence_length: int = 30
    stride: int = 15
    
    # Augmentation
    apply_augmentation: bool = True
    augmentation_config: Optional[dict] = None
    
    # Splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Processing
    batch_size: int = 32
    num_workers: int = 4
    skip_existing: bool = True
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PipelineConfig':
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class PreprocessingPipeline:
    """Orchestrates the entire preprocessing pipeline."""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize preprocessing pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.dataset_root = Path(config.dataset_root)
        self.output_root = Path(config.output_root)
        
        # Initialize components
        self.metadata_manager = MetadataManager(
            str(self.dataset_root / 'metadata.csv')
        )
        
        self.video_preprocessor = VideoPreprocessor(
            min_duration=config.min_duration,
            max_duration=config.max_duration,
            min_fps=config.min_fps,
            target_fps=config.target_fps
        )
        
        self.quality_filter = QualityFilter()
        
        self.landmark_extractor = LandmarkExtractor(
            LandmarkExtractionConfig(
                max_num_hands=config.max_num_hands,
                include_z_coordinates=config.include_z_coordinates
            )
        )
        
        self.sequence_builder = SequenceBuilder(
            SequenceBuilderConfig(
                sequence_length=config.sequence_length,
                stride=config.stride
            )
        )
        
        self.sequence_generator = SequenceGenerator(
            SequenceGeneratorConfig(
                sequence_length=config.sequence_length,
                stride=config.stride,
                train_ratio=config.train_ratio,
                val_ratio=config.val_ratio,
                test_ratio=config.test_ratio,
                apply_augmentation=config.apply_augmentation,
                augmentation_config=config.augmentation_config
            )
        )
        
        # Create output directories
        self._create_output_directories()
    
    def _create_output_directories(self) -> None:
        """Create output directories."""
        directories = [
            self.output_root / 'landmarks',
            self.output_root / 'sequences' / 'train',
            self.output_root / 'sequences' / 'val',
            self.output_root / 'sequences' / 'test',
            self.output_root / 'quality_reports',
            self.output_root / 'logs',
        ]
        
        for directory in directories:
            ensure_directory(str(directory))
    
    def run_full_pipeline(self) -> Dict:
        """
        Run the complete preprocessing pipeline.
        
        Returns:
            Dictionary with pipeline statistics
        """
        statistics = {
            'total_videos': 0,
            'processed_videos': 0,
            'failed_videos': 0,
            'skipped_videos': 0,
            'quality_filtered': 0,
            'total_sequences': 0,
            'train_sequences': 0,
            'val_sequences': 0,
            'test_sequences': 0,
        }
        
        # Step 1: Validate dataset
        print("Validating dataset...")
        validator = DatasetValidator(str(self.dataset_root), self.metadata_manager)
        validation_result = validator.validate_all()
        
        if not validation_result.is_valid:
            print(f"Dataset validation failed: {validation_result}")
            validator.generate_report(str(self.output_root / 'quality_reports' / 'validation_report.json'))
        
        # Step 2: Process videos
        print("Processing videos...")
        landmark_data = {}
        labels = {}
        subjects = {}
        
        for video_id, metadata in self.metadata_manager.metadata.items():
            statistics['total_videos'] += 1
            
            video_path = self.dataset_root / metadata.video_path
            
            # Skip if output exists
            if self.config.skip_existing:
                landmark_path = self.output_root / 'landmarks' / f'{video_id}.npy'
                if landmark_path.exists():
                    statistics['skipped_videos'] += 1
                    continue
            
            # Validate video
            is_valid, reason = self.video_preprocessor.validate_video(str(video_path))
            if not is_valid:
                print(f"Video validation failed for {video_id}: {reason}")
                statistics['failed_videos'] += 1
                continue
            
            # Quality filter
            if self.config.apply_quality_filter:
                is_accepted, reason, quality_metrics = self.quality_filter.filter_video(str(video_path))
                if not is_accepted:
                    print(f"Quality filter rejected {video_id}: {reason}")
                    statistics['quality_filtered'] += 1
                    continue
            
            # Extract frames
            print(f"Extracting frames from {video_id}...")
            frame_extractor = FrameExtractor()
            try:
                frames = frame_extractor.extract_frames(str(video_path))
            except Exception as e:
                print(f"Frame extraction failed for {video_id}: {e}")
                statistics['failed_videos'] += 1
                continue
            
            # Extract landmarks
            print(f"Extracting landmarks from {video_id}...")
            try:
                landmarks = self.landmark_extractor.extract_landmarks_batch(frames)
                # Filter out None values
                landmarks = [lm for lm in landmarks if lm is not None]
                
                if not landmarks:
                    print(f"No landmarks extracted for {video_id}")
                    statistics['failed_videos'] += 1
                    continue
                
                landmarks_array = np.array(landmarks, dtype=np.float32)
                
                # Save landmarks
                landmark_path = self.output_root / 'landmarks' / f'{video_id}.npy'
                np.save(landmark_path, landmarks_array)
                
                landmark_data[video_id] = landmarks_array
                labels[video_id] = metadata.gesture
                subjects[video_id] = metadata.subject
                
                statistics['processed_videos'] += 1
                
            except Exception as e:
                print(f"Landmark extraction failed for {video_id}: {e}")
                statistics['failed_videos'] += 1
                continue
        
        # Step 3: Generate subject-based splits
        print("Generating subject-based splits...")
        video_ids = list(landmark_data.keys())
        subject_splits = self.sequence_generator.generate_subject_based_splits(
            video_ids, subjects, labels
        )
        
        # Step 4: Generate sequences
        print("Generating sequences...")
        sequences = self.sequence_generator.generate_sequences(
            landmark_data, labels, subject_splits
        )
        
        # Step 5: Save sequences
        print("Saving sequences...")
        self.sequence_generator.save_sequences(sequences, str(self.output_root / 'sequences'))
        
        # Step 6: Calculate statistics
        seq_stats = self.sequence_generator.get_statistics(sequences)
        
        for split, stats in seq_stats.items():
            statistics[f'{split}_sequences'] = stats['num_sequences']
            statistics['total_sequences'] += stats['num_sequences']
        
        # Step 7: Save statistics
        statistics_path = self.output_root / 'pipeline_statistics.json'
        with open(statistics_path, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        print(f"Pipeline completed. Statistics: {statistics}")
        
        return statistics
    
    def process_single_video(self, video_id: str) -> Optional[np.ndarray]:
        """
        Process a single video and return landmarks.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Landmarks array or None if processing fails
        """
        metadata = self.metadata_manager.get_video(video_id)
        if not metadata:
            print(f"Video metadata not found: {video_id}")
            return None
        
        video_path = self.dataset_root / metadata.video_path
        
        # Validate video
        is_valid, reason = self.video_preprocessor.validate_video(str(video_path))
        if not is_valid:
            print(f"Video validation failed: {reason}")
            return None
        
        # Extract frames
        frame_extractor = FrameExtractor()
        frames = frame_extractor.extract_frames(str(video_path))
        
        # Extract landmarks
        landmarks = self.landmark_extractor.extract_landmarks_batch(frames)
        landmarks = [lm for lm in landmarks if lm is not None]
        
        if not landmarks:
            return None
        
        return np.array(landmarks, dtype=np.float32)
    
    def get_pipeline_status(self) -> Dict:
        """
        Get current pipeline status.
        
        Returns:
            Dictionary with pipeline status
        """
        status = {
            'dataset_root': str(self.dataset_root),
            'output_root': str(self.output_root),
            'total_videos': len(self.metadata_manager.metadata),
            'processed_landmarks': len(list((self.output_root / 'landmarks').glob('*.npy'))),
            'sequences_generated': self._check_sequences_generated(),
        }
        
        return status
    
    def _check_sequences_generated(self) -> bool:
        """Check if sequences have been generated."""
        train_dir = self.output_root / 'sequences' / 'train'
        return train_dir.exists() and (train_dir / 'sequences.npy').exists()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.landmark_extractor.close()


def main():
    """Main entry point for preprocessing pipeline."""
    # Load configuration
    config_path = Path(__file__).parent.parent.parent / 'experiments' / 'configs' / 'preprocessing_config.yaml'
    
    if config_path.exists():
        config = PipelineConfig.from_yaml(str(config_path))
    else:
        # Use default configuration
        config = PipelineConfig(
            dataset_root='ml/datasets/raw',
            output_root='ml/datasets/processed'
        )
    
    # Run pipeline
    pipeline = PreprocessingPipeline(config)
    
    try:
        statistics = pipeline.run_full_pipeline()
        print("Preprocessing pipeline completed successfully!")
        print(f"Statistics: {statistics}")
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise
    finally:
        pipeline.cleanup()


if __name__ == '__main__':
    main()
