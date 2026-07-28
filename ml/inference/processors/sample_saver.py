"""
Sample Saver and Dataset Exporter for ISL Feature Extraction.

Provides efficient saving and loading mechanisms for extracted feature sequences
using compressed NumPy archives (.npz) and array files (.npy) for ML training.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict

from ml.inference.processors.holistic_feature_extractor import HolisticFeatures, FeatureVector

logger = logging.getLogger(__name__)


@dataclass
class SampleMetadata:
    """Metadata associated with a saved landmark sequence sample."""
    sample_id: str
    label: str
    num_frames: int
    feature_dim: int
    fps: float = 30.0
    subject_id: Optional[str] = None
    source_video: Optional[str] = None
    extra: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert metadata to dictionary."""
        return asdict(self)


class SampleSaver:
    """
    Efficient saver and loader for sign language feature vector samples.
    
    Supports:
    - Compressed .npz format (recommended: high compression, fast read/write, retains metadata).
    - Uncompressed .npy format.
    - Structured directory organization by dataset split and class label.
    """

    def __init__(self, base_output_dir: Union[str, Path]):
        """
        Initialize SampleSaver.

        Args:
            base_output_dir: Root directory for saving processed dataset samples.
        """
        self.base_dir = Path(base_output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_sample_npz(
        self,
        features: Union[np.ndarray, List[HolisticFeatures]],
        label: str,
        sample_id: str,
        split: str = "train",
        metadata: Optional[SampleMetadata] = None,
        compress: bool = True
    ) -> Path:
        """
        Save a feature sequence sample as a compressed .npz archive.

        Args:
            features: 2D numpy array of shape (T_frames, D_features) or list of HolisticFeatures
            label: Gesture/sign label name
            sample_id: Unique identifier for the sample
            split: Dataset split ('train', 'val', 'test')
            metadata: Optional SampleMetadata object
            compress: Whether to use zip compression (.npz)

        Returns:
            Path to saved .npz file
        """
        # Convert HolisticFeatures list to 2D numpy array if needed
        if isinstance(features, list):
            feat_matrix = np.array([f.get_feature_vector() for f in features], dtype=np.float32)
        else:
            feat_matrix = np.array(features, dtype=np.float32)

        if feat_matrix.ndim != 2:
            raise ValueError(f"Features must be a 2D array (T_frames, D_features), got shape {feat_matrix.shape}")

        # Prepare output path
        output_dir = self.base_dir / split / label
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"{sample_id}.npz"

        # Prepare metadata JSON string
        if metadata is None:
            metadata = SampleMetadata(
                sample_id=sample_id,
                label=label,
                num_frames=feat_matrix.shape[0],
                feature_dim=feat_matrix.shape[1]
            )

        metadata_json = json.dumps(metadata.to_dict())

        # Save compressed or uncompressed archive
        if compress:
            np.savez_compressed(
                file_path,
                features=feat_matrix,
                label=label,
                sample_id=sample_id,
                metadata=metadata_json
            )
        else:
            np.savez(
                file_path,
                features=feat_matrix,
                label=label,
                sample_id=sample_id,
                metadata=metadata_json
            )

        logger.debug(f"Saved sample {sample_id} to {file_path}")
        return file_path

    def load_sample_npz(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
        """
        Load a feature sample and its metadata from a .npz archive.

        Args:
            file_path: Path to .npz file

        Returns:
            Tuple of (features_array, metadata_dict)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Sample file not found: {file_path}")

        data = np.load(file_path, allow_pickle=True)
        features = data['features']
        
        metadata = {}
        if 'metadata' in data:
            metadata = json.loads(str(data['metadata']))
        else:
            metadata = {
                'label': str(data.get('label', '')),
                'sample_id': str(data.get('sample_id', file_path.stem))
            }

        return features, metadata

    def save_batch_samples(
        self,
        samples: List[Tuple[Union[np.ndarray, List[HolisticFeatures]], str, str]],
        split: str = "train"
    ) -> List[Path]:
        """
        Save a batch of samples.

        Args:
            samples: List of tuples (features, label, sample_id)
            split: Dataset split ('train', 'val', 'test')

        Returns:
            List of saved file paths
        """
        saved_paths = []
        for features, label, sample_id in samples:
            path = self.save_sample_npz(features, label, sample_id, split=split)
            saved_paths.append(path)
        return saved_paths
