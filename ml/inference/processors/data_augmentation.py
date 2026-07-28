"""
Modular Data Augmentation Pipeline for Landmark Sequences in ISL Recognition.

Applies semantic-preserving spatial and temporal transformations to 2D/3D landmark sequences:
- Temporal cropping & padding
- Time stretching (speed up / slow down)
- Gaussian noise (jitter)
- Spatial Rotation (2D/3D Euler rotation around reference origin)
- Spatial Scaling (uniform / scale jitter)
- Spatial Translation (x, y, z offset shift)
- Landmark Dropout (keypoint occlusion masking)

All augmentations are individually configurable and designed to preserve sign language semantics.
"""

import logging
import random
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for landmark sequence data augmentation."""
    
    # Master toggle
    enabled: bool = True
    
    # Individual Augmentation Toggles & Probabilities
    enable_temporal_cropping: bool = True
    prob_temporal_cropping: float = 0.3
    crop_ratio_range: Tuple[float, float] = (0.75, 0.95)  # Retain 75% - 95% of sequence length
    
    enable_time_stretching: bool = True
    prob_time_stretching: float = 0.4
    time_stretch_range: Tuple[float, float] = (0.8, 1.25)  # Speed factor 0.8x (slower) to 1.25x (faster)
    
    enable_gaussian_noise: bool = True
    prob_gaussian_noise: float = 0.5
    gaussian_noise_std: float = 0.008  # Small std dev to prevent semantic shape distortion
    
    enable_rotation: bool = True
    prob_rotation: float = 0.4
    rotation_range_deg: float = 12.0  # Max +/- 12 degrees to keep hand shape upright & readable
    
    enable_scaling: bool = True
    prob_scaling: float = 0.4
    scaling_range: Tuple[float, float] = (0.88, 1.12)  # +/- 12% scale factor
    
    enable_translation: bool = True
    prob_translation: float = 0.4
    translation_range: float = 0.04  # +/- 4% coordinate shift
    
    enable_landmark_dropout: bool = True
    prob_landmark_dropout: float = 0.3
    dropout_rate: float = 0.08  # Max 8% keypoint masking to simulate real-world finger/face occlusion
    
    # Reproducibility
    random_seed: Optional[int] = None
    
    @classmethod
    def get_conservative_config(cls) -> 'AugmentationConfig':
        """Get conservative augmentation config for sensitive signs."""
        return cls(
            enabled=True,
            rotation_range_deg=8.0,
            scaling_range=(0.92, 1.08),
            translation_range=0.02,
            gaussian_noise_std=0.004,
            dropout_rate=0.04
        )
    
    @classmethod
    def get_disabled_config(cls) -> 'AugmentationConfig':
        """Get config with all augmentations disabled."""
        return cls(enabled=False)


class DataAugmenter:
    """
    Applies modular, semantic-preserving data augmentations to landmark sequences.
    
    Sequence Shape: (T_frames, D_features) or (T_frames, N_landmarks, C_coords)
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize DataAugmenter.

        Args:
            config: Augmentation configuration object
        """
        self.config = config or AugmentationConfig()
        
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
            random.seed(self.config.random_seed)

    def augment_sequence(
        self,
        sequence: np.ndarray,
        coord_dim: int = 3
    ) -> np.ndarray:
        """
        Apply sequence of enabled augmentations to input landmark sequence.

        Args:
            sequence: Input array of shape (T, D) or (T, N, C)
            coord_dim: Spatial dimension per landmark (2 or 3)

        Returns:
            Augmented sequence of same data type (float32)
        """
        if not self.config.enabled or sequence is None or sequence.size == 0:
            return sequence.copy() if sequence is not None else sequence

        augmented = sequence.copy().astype(np.float32)
        
        # 1. Temporal Cropping
        if self.config.enable_temporal_cropping and np.random.random() < self.config.prob_temporal_cropping:
            augmented = self.apply_temporal_cropping(augmented)

        # 2. Time Stretching
        if self.config.enable_time_stretching and np.random.random() < self.config.prob_time_stretching:
            augmented = self.apply_time_stretching(augmented)

        # 3. Spatial Rotation
        if self.config.enable_rotation and np.random.random() < self.config.prob_rotation:
            augmented = self.apply_rotation(augmented, coord_dim=coord_dim)

        # 4. Spatial Scaling
        if self.config.enable_scaling and np.random.random() < self.config.prob_scaling:
            augmented = self.apply_scaling(augmented, coord_dim=coord_dim)

        # 5. Spatial Translation
        if self.config.enable_translation and np.random.random() < self.config.prob_translation:
            augmented = self.apply_translation(augmented, coord_dim=coord_dim)

        # 6. Gaussian Noise (Jitter)
        if self.config.enable_gaussian_noise and np.random.random() < self.config.prob_gaussian_noise:
            augmented = self.apply_gaussian_noise(augmented)

        # 7. Landmark Dropout (Occlusion)
        if self.config.enable_landmark_dropout and np.random.random() < self.config.prob_landmark_dropout:
            augmented = self.apply_landmark_dropout(augmented)

        return augmented

    def apply_temporal_cropping(self, sequence: np.ndarray) -> np.ndarray:
        """
        Crop a continuous temporal clip from the sequence and resample back to original length.

        Args:
            sequence: Sequence array of shape (T, D)

        Returns:
            Temporally cropped & resampled sequence of shape (T, D)
        """
        T = sequence.shape[0]
        if T < 4:
            return sequence

        low, high = self.config.crop_ratio_range
        crop_ratio = np.random.uniform(low, high)
        crop_len = max(2, int(T * crop_ratio))

        start_idx = np.random.randint(0, T - crop_len + 1)
        end_idx = start_idx + crop_len

        cropped = sequence[start_idx:end_idx]

        # Resample back to target sequence length T using linear interpolation
        orig_indices = np.linspace(0, crop_len - 1, num=crop_len)
        target_indices = np.linspace(0, crop_len - 1, num=T)

        resampled = np.zeros((T, sequence.shape[1]), dtype=np.float32)
        for d in range(sequence.shape[1]):
            resampled[:, d] = np.interp(target_indices, orig_indices, cropped[:, d])

        return resampled

    def apply_time_stretching(self, sequence: np.ndarray) -> np.ndarray:
        """
        Speed up or slow down gesture motion in time using linear interpolation.

        Args:
            sequence: Sequence array of shape (T, D)

        Returns:
            Time-stretched sequence of shape (T, D)
        """
        T = sequence.shape[0]
        if T < 4:
            return sequence

        low, high = self.config.time_stretch_range
        speed_factor = np.random.uniform(low, high)
        new_len = max(2, int(T * speed_factor))

        orig_indices = np.arange(T)
        stretched_indices = np.linspace(0, T - 1, num=new_len)

        # Interpolate to stretched length
        stretched = np.zeros((new_len, sequence.shape[1]), dtype=np.float32)
        for d in range(sequence.shape[1]):
            stretched[:, d] = np.interp(stretched_indices, orig_indices, sequence[:, d])

        # Resample back to target sequence length T
        resampled_indices = np.linspace(0, new_len - 1, num=T)
        resampled = np.zeros((T, sequence.shape[1]), dtype=np.float32)
        for d in range(sequence.shape[1]):
            resampled[:, d] = np.interp(resampled_indices, np.arange(new_len), stretched[:, d])

        return resampled

    def apply_rotation(
        self,
        sequence: np.ndarray,
        coord_dim: int = 3
    ) -> np.ndarray:
        """
        Apply random 2D rotation in the XY plane around center of keypoints.

        Args:
            sequence: Sequence array of shape (T, D)
            coord_dim: Dimension per landmark (2 or 3)

        Returns:
            Rotated sequence
        """
        angle_deg = np.random.uniform(-self.config.rotation_range_deg, self.config.rotation_range_deg)
        rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        rot_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ], dtype=np.float32)

        augmented = sequence.copy()
        T, D = augmented.shape

        for t in range(T):
            for j in range(0, D, coord_dim):
                if j + 1 < D:
                    # Skip zero-padded missing landmarks
                    if augmented[t, j] == 0 and augmented[t, j + 1] == 0:
                        continue
                    xy = np.array([augmented[t, j], augmented[t, j + 1]], dtype=np.float32)
                    rotated = rot_matrix @ xy
                    augmented[t, j] = rotated[0]
                    augmented[t, j + 1] = rotated[1]

        return augmented

    def apply_scaling(
        self,
        sequence: np.ndarray,
        coord_dim: int = 3
    ) -> np.ndarray:
        """
        Apply random spatial scaling factor to coordinates.

        Args:
            sequence: Sequence array of shape (T, D)
            coord_dim: Dimension per landmark (2 or 3)

        Returns:
            Scaled sequence
        """
        low, high = self.config.scaling_range
        scale = np.random.uniform(low, high)

        augmented = sequence.copy()
        T, D = augmented.shape

        for t in range(T):
            for j in range(0, D, coord_dim):
                if j + coord_dim <= D:
                    # Don't scale if zero-padded
                    if np.all(augmented[t, j:j+coord_dim] == 0):
                        continue
                    augmented[t, j:j+coord_dim] *= scale

        return augmented

    def apply_translation(
        self,
        sequence: np.ndarray,
        coord_dim: int = 3
    ) -> np.ndarray:
        """
        Apply random spatial shift (x, y, z translation) to coordinates.

        Args:
            sequence: Sequence array of shape (T, D)
            coord_dim: Dimension per landmark (2 or 3)

        Returns:
            Translated sequence
        """
        shift_x = np.random.uniform(-self.config.translation_range, self.config.translation_range)
        shift_y = np.random.uniform(-self.config.translation_range, self.config.translation_range)
        shift_z = np.random.uniform(-self.config.translation_range, self.config.translation_range) if coord_dim == 3 else 0.0

        shifts = np.array([shift_x, shift_y, shift_z][:coord_dim], dtype=np.float32)

        augmented = sequence.copy()
        T, D = augmented.shape

        for t in range(T):
            for j in range(0, D, coord_dim):
                if j + coord_dim <= D:
                    # Skip zero-padded missing landmarks
                    if np.all(augmented[t, j:j+coord_dim] == 0):
                        continue
                    augmented[t, j:j+coord_dim] += shifts

        return augmented

    def apply_gaussian_noise(self, sequence: np.ndarray) -> np.ndarray:
        """
        Add zero-mean Gaussian jitter noise to valid non-zero landmarks.

        Args:
            sequence: Sequence array of shape (T, D)

        Returns:
            Jittered sequence
        """
        noise = np.random.normal(0, self.config.gaussian_noise_std, size=sequence.shape).astype(np.float32)
        # Apply noise only where landmark is non-zero
        valid_mask = (sequence != 0)
        return sequence + (noise * valid_mask)

    def apply_landmark_dropout(self, sequence: np.ndarray) -> np.ndarray:
        """
        Randomly mask out (zero out) individual keypoints to simulate finger/hand occlusion.

        Args:
            sequence: Sequence array of shape (T, D)

        Returns:
            Sequence with random keypoint dropout mask applied
        """
        mask = (np.random.random(sequence.shape) > self.config.dropout_rate).astype(np.float32)
        return sequence * mask

    def augment_batch(
        self,
        sequences: np.ndarray,
        labels: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Apply augmentation to a batch of sequences.

        Args:
            sequences: Batch array of shape (B, T, D)
            labels: List of B label strings

        Returns:
            Tuple of (augmented_sequences, labels)
        """
        aug_seqs = []
        for seq in sequences:
            aug_seqs.append(self.augment_sequence(seq))
        return np.array(aug_seqs, dtype=np.float32), labels

    def augment_with_original(
        self,
        sequences: np.ndarray,
        labels: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Concatenate original sequences with their augmented counterparts to double dataset size.

        Args:
            sequences: Batch array of shape (B, T, D)
            labels: List of B label strings

        Returns:
            Tuple of (combined_sequences, combined_labels)
        """
        aug_seqs, aug_labels = self.augment_batch(sequences, labels)
        combined_seqs = np.vstack([sequences, aug_seqs])
        combined_labels = labels + aug_labels
        return combined_seqs, combined_labels
