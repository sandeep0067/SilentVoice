"""
Sequence Builder for landmark data.

Builds fixed-length sequences from landmark data for model training.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Generator
from dataclasses import dataclass


@dataclass
class SequenceBuilderConfig:
    """Configuration for sequence building."""
    sequence_length: int = 30
    stride: int = 15  # Overlap between sequences
    padding_mode: str = 'zero'  # 'zero', 'repeat', 'edge'
    normalize_temporal: bool = False
    min_sequence_length: int = 10  # Minimum valid sequence length


class SequenceBuilder:
    """Builds sequences from landmark data."""
    
    def __init__(self, config: Optional[SequenceBuilderConfig] = None):
        """
        Initialize sequence builder.
        
        Args:
            config: Sequence builder configuration
        """
        self.config = config or SequenceBuilderConfig()
    
    def build_sequences(
        self,
        landmarks: np.ndarray,
        label: str
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build sequences from landmark array.
        
        Args:
            landmarks: Landmark array of shape (num_frames, landmark_dim)
            label: Gesture label
            
        Returns:
            Tuple of (sequences array, labels list)
        """
        sequences = []
        labels = []
        
        num_frames = landmarks.shape[0]
        
        # Skip if too short
        if num_frames < self.config.min_sequence_length:
            return np.array([]), []
        
        # Generate sequences with sliding window
        for start_idx in range(
            0,
            num_frames - self.config.sequence_length + 1,
            self.config.stride
        ):
            end_idx = start_idx + self.config.sequence_length
            
            if end_idx > num_frames:
                # Pad if sequence extends beyond data
                sequence = self._pad_sequence(
                    landmarks[start_idx:],
                    self.config.sequence_length
                )
            else:
                sequence = landmarks[start_idx:end_idx]
            
            sequences.append(sequence)
            labels.append(label)
        
        if not sequences:
            return np.array([]), []
        
        return np.array(sequences, dtype=np.float32), labels
    
    def build_sequences_generator(
        self,
        landmarks: np.ndarray,
        label: str
    ) -> Generator[Tuple[np.ndarray, str], None, None]:
        """
        Build sequences as a generator (memory efficient).
        
        Args:
            landmarks: Landmark array of shape (num_frames, landmark_dim)
            label: Gesture label
            
        Yields:
            Tuple of (sequence, label)
        """
        num_frames = landmarks.shape[0]
        
        if num_frames < self.config.min_sequence_length:
            return
        
        for start_idx in range(
            0,
            num_frames - self.config.sequence_length + 1,
            self.config.stride
        ):
            end_idx = start_idx + self.config.sequence_length
            
            if end_idx > num_frames:
                sequence = self._pad_sequence(
                    landmarks[start_idx:],
                    self.config.sequence_length
                )
            else:
                sequence = landmarks[start_idx:end_idx]
            
            yield sequence, label
    
    def build_dynamic_sequences(
        self,
        landmarks: np.ndarray,
        label: str
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build sequences with dynamic length padding.
        
        Args:
            landmarks: Landmark array of shape (num_frames, landmark_dim)
            label: Gesture label
            
        Returns:
            Tuple of (sequences array, labels list)
        """
        num_frames = landmarks.shape[0]
        
        if num_frames < self.config.min_sequence_length:
            return np.array([]), []
        
        # Pad or truncate to fixed length
        sequence = self._pad_sequence(
            landmarks,
            self.config.sequence_length
        )
        
        return np.array([sequence], dtype=np.float32), [label]
    
    def _pad_sequence(
        self,
        sequence: np.ndarray,
        target_length: int
    ) -> np.ndarray:
        """
        Pad sequence to target length.
        
        Args:
            sequence: Input sequence
            target_length: Target length
            
        Returns:
            Padded sequence
        """
        current_length = sequence.shape[0]
        
        if current_length >= target_length:
            return sequence[:target_length]
        
        padding_length = target_length - current_length
        landmark_dim = sequence.shape[1]
        
        if self.config.padding_mode == 'zero':
            padding = np.zeros((padding_length, landmark_dim), dtype=np.float32)
        elif self.config.padding_mode == 'repeat':
            padding = np.tile(sequence[-1:], (padding_length, 1))
        elif self.config.padding_mode == 'edge':
            padding = np.tile(sequence[-1:], (padding_length, 1))
        else:
            padding = np.zeros((padding_length, landmark_dim), dtype=np.float32)
        
        return np.vstack([sequence, padding])
    
    def build_batch_sequences(
        self,
        landmarks_list: List[np.ndarray],
        labels: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build sequences from multiple landmark arrays.
        
        Args:
            landmarks_list: List of landmark arrays
            labels: List of corresponding labels
            
        Returns:
            Tuple of (batch sequences array, batch labels list)
        """
        all_sequences = []
        all_labels = []
        
        for landmarks, label in zip(landmarks_list, labels):
            sequences, seq_labels = self.build_sequences(landmarks, label)
            all_sequences.append(sequences)
            all_labels.extend(seq_labels)
        
        if not all_sequences:
            return np.array([]), []
        
        return np.vstack(all_sequences), all_labels
    
    def normalize_temporal(self, sequence: np.ndarray) -> np.ndarray:
        """
        Normalize sequence temporally (optional).
        
        Args:
            sequence: Input sequence
            
        Returns:
            Temporally normalized sequence
        """
        # Placeholder for temporal normalization
        # Could implement time warping or interpolation here
        return sequence
    
    def calculate_statistics(
        self,
        sequences: np.ndarray
    ) -> dict:
        """
        Calculate statistics for sequences.
        
        Args:
            sequences: Sequences array
            
        Returns:
            Dictionary of statistics
        """
        return {
            'num_sequences': sequences.shape[0],
            'sequence_length': sequences.shape[1],
            'landmark_dim': sequences.shape[2],
            'mean': np.mean(sequences),
            'std': np.std(sequences),
            'min': np.min(sequences),
            'max': np.max(sequences)
        }
