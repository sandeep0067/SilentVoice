"""
Sequence Generator for training data.

Generates sequences from landmark data with train/val/test splits.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

from ml.inference.processors.sequence_builder import SequenceBuilder, SequenceBuilderConfig
from ml.inference.processors.data_augmentation import DataAugmenter, AugmentationConfig


@dataclass
class SequenceGeneratorConfig:
    """Configuration for sequence generation."""
    # Sequence building
    sequence_length: int = 30
    stride: int = 15
    min_sequence_length: int = 10
    
    # Data splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Augmentation
    apply_augmentation: bool = True
    augmentation_config: Optional[dict] = None
    
    # Output
    save_to_disk: bool = True
    output_format: str = 'npy'  # 'npy' or 'hdf5'


class SequenceGenerator:
    """Generates training sequences from landmark data."""
    
    def __init__(self, config: Optional[SequenceGeneratorConfig] = None):
        """
        Initialize sequence generator.
        
        Args:
            config: Sequence generator configuration
        """
        self.config = config or SequenceGeneratorConfig()
        
        # Initialize sequence builder
        self.sequence_builder = SequenceBuilder(
            SequenceBuilderConfig(
                sequence_length=self.config.sequence_length,
                stride=self.config.stride,
                min_sequence_length=self.config.min_sequence_length
            )
        )
        
        # Initialize augmenter if enabled
        self.augmenter = None
        if self.config.apply_augmentation:
            aug_config = self.config.augmentation_config or {}
            self.augmenter = DataAugmenter(AugmentationConfig(**aug_config))
    
    def generate_sequences(
        self,
        landmark_data: Dict[str, np.ndarray],
        labels: Dict[str, str],
        subject_splits: Optional[Dict[str, str]] = None
    ) -> Dict[str, Tuple[np.ndarray, List[str]]]:
        """
        Generate sequences from landmark data.
        
        Args:
            landmark_data: Dictionary mapping video_id to landmark arrays
            labels: Dictionary mapping video_id to gesture labels
            subject_splits: Dictionary mapping video_id to split ('train', 'val', 'test')
            
        Returns:
            Dictionary mapping split to (sequences, labels)
        """
        splits = {'train': [], 'val': [], 'test': []}
        
        for video_id, landmarks in landmark_data.items():
            label = labels.get(video_id, 'unknown')
            split = subject_splits.get(video_id, 'train') if subject_splits else 'train'
            
            # Build sequences
            sequences, seq_labels = self.sequence_builder.build_sequences(landmarks, label)
            
            if len(sequences) == 0:
                continue
            
            # Apply augmentation for training data
            if split == 'train' and self.augmenter:
                sequences, seq_labels = self.augmenter.augment_with_original(sequences, seq_labels)
            
            splits[split].append((sequences, seq_labels))
        
        # Combine sequences per split
        result = {}
        for split in splits:
            if not splits[split]:
                result[split] = (np.array([]), [])
                continue
            
            all_sequences = np.vstack([s[0] for s in splits[split]])
            all_labels = []
            for s in splits[split]:
                all_labels.extend(s[1])
            
            result[split] = (all_sequences, all_labels)
        
        return result
    
    def generate_subject_based_splits(
        self,
        video_ids: List[str],
        subjects: Dict[str, str],
        labels: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Generate subject-based train/val/test splits.
        
        Args:
            video_ids: List of video IDs
            subjects: Dictionary mapping video_id to subject
            labels: Dictionary mapping video_id to label
            
        Returns:
            Dictionary mapping video_id to split
        """
        # Get unique subjects
        unique_subjects = sorted(set(subjects.values()))
        
        # Shuffle subjects
        np.random.shuffle(unique_subjects)
        
        # Calculate split indices
        n_subjects = len(unique_subjects)
        train_end = int(n_subjects * self.config.train_ratio)
        val_end = train_end + int(n_subjects * self.config.val_ratio)
        
        train_subjects = set(unique_subjects[:train_end])
        val_subjects = set(unique_subjects[train_end:val_end])
        test_subjects = set(unique_subjects[val_end:])
        
        # Assign splits based on subjects
        splits = {}
        for video_id in video_ids:
            subject = subjects.get(video_id, 'unknown')
            if subject in train_subjects:
                splits[video_id] = 'train'
            elif subject in val_subjects:
                splits[video_id] = 'val'
            elif subject in test_subjects:
                splits[video_id] = 'test'
            else:
                splits[video_id] = 'train'
        
        return splits
    
    def save_sequences(
        self,
        sequences: Dict[str, Tuple[np.ndarray, List[str]]],
        output_dir: str
    ) -> None:
        """
        Save sequences to disk.
        
        Args:
            sequences: Dictionary mapping split to (sequences, labels)
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split, (seq_array, labels) in sequences.items():
            split_dir = output_path / split
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Save sequences
            if self.config.output_format == 'npy':
                np.save(split_dir / 'sequences.npy', seq_array)
                
                # Save labels as JSON
                with open(split_dir / 'labels.json', 'w') as f:
                    json.dump(labels, f)
            elif self.config.output_format == 'hdf5':
                import h5py
                
                with h5py.File(split_dir / 'data.h5', 'w') as f:
                    f.create_dataset('sequences', data=seq_array)
                    f.create_dataset('labels', data=np.array(labels, dtype='S'))
    
    def load_sequences(
        self,
        input_dir: str,
        split: str
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Load sequences from disk.
        
        Args:
            input_dir: Input directory
            split: Split to load ('train', 'val', 'test')
            
        Returns:
            Tuple of (sequences, labels)
        """
        input_path = Path(input_dir) / split
        
        if self.config.output_format == 'npy':
            sequences = np.load(input_path / 'sequences.npy')
            
            with open(input_path / 'labels.json', 'r') as f:
                labels = json.load(f)
        elif self.config.output_format == 'hdf5':
            import h5py
            
            with h5py.File(input_path / 'data.h5', 'r') as f:
                sequences = f['sequences'][:]
                labels = [label.decode('utf-8') for label in f['labels'][:]]
        else:
            raise ValueError(f"Unknown format: {self.config.output_format}")
        
        return sequences, labels
    
    def get_statistics(
        self,
        sequences: Dict[str, Tuple[np.ndarray, List[str]]]
    ) -> Dict[str, Dict]:
        """
        Get statistics for generated sequences.
        
        Args:
            sequences: Dictionary mapping split to (sequences, labels)
            
        Returns:
            Dictionary of statistics per split
        """
        stats = {}
        
        for split, (seq_array, labels) in sequences.items():
            if len(seq_array) == 0:
                stats[split] = {
                    'num_sequences': 0,
                    'num_labels': 0,
                    'unique_labels': 0
                }
                continue
            
            unique_labels = len(set(labels))
            label_counts = {label: labels.count(label) for label in set(labels)}
            
            stats[split] = {
                'num_sequences': len(seq_array),
                'num_labels': len(labels),
                'unique_labels': unique_labels,
                'label_counts': label_counts,
                'sequence_shape': seq_array.shape,
                'mean': float(np.mean(seq_array)),
                'std': float(np.std(seq_array))
            }
        
        return stats
