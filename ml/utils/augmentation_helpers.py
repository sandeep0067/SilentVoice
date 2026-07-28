"""
Augmentation helper functions.
"""

import numpy as np
from typing import List, Tuple, Optional


def rotate_landmarks_2d(
    landmarks: np.ndarray,
    angle_degrees: float
) -> np.ndarray:
    """
    Rotate landmarks in 2D plane.
    
    Args:
        landmarks: Landmarks array (seq_len, landmark_dim)
        angle_degrees: Rotation angle in degrees
        
    Returns:
        Rotated landmarks
    """
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    rotated = landmarks.copy()
    
    for i in range(rotated.shape[0]):
        for j in range(0, rotated.shape[1], 3):
            if j + 1 < rotated.shape[1]:
                x, y = rotated[i, j], rotated[i, j + 1]
                rotated_xy = rotation_matrix @ np.array([x, y])
                rotated[i, j] = rotated_xy[0]
                rotated[i, j + 1] = rotated_xy[1]
    
    return rotated


def scale_landmarks(
    landmarks: np.ndarray,
    scale_factor: float
) -> np.ndarray:
    """
    Scale landmarks by factor.
    
    Args:
        landmarks: Landmarks array (seq_len, landmark_dim)
        scale_factor: Scaling factor
        
    Returns:
        Scaled landmarks
    """
    scaled = landmarks.copy()
    
    for i in range(scaled.shape[0]):
        for j in range(0, scaled.shape[1], 3):
            if j < scaled.shape[1]:
                scaled[i, j] *= scale_factor
            if j + 1 < scaled.shape[1]:
                scaled[i, j + 1] *= scale_factor
    
    return scaled


def translate_landmarks(
    landmarks: np.ndarray,
    shift_x: float,
    shift_y: float
) -> np.ndarray:
    """
    Translate landmarks by offset.
    
    Args:
        landmarks: Landmarks array (seq_len, landmark_dim)
        shift_x: X shift
        shift_y: Y shift
        
    Returns:
        Translated landmarks
    """
    translated = landmarks.copy()
    
    for i in range(translated.shape[0]):
        for j in range(0, translated.shape[1], 3):
            if j < translated.shape[1]:
                translated[i, j] += shift_x
            if j + 1 < translated.shape[1]:
                translated[i, j + 1] += shift_y
    
    return translated


def add_gaussian_noise(
    landmarks: np.ndarray,
    std: float = 0.01
) -> np.ndarray:
    """
    Add Gaussian noise to landmarks.
    
    Args:
        landmarks: Landmarks array (seq_len, landmark_dim)
        std: Standard deviation of noise
        
    Returns:
        Noisy landmarks
    """
    noise = np.random.normal(0, std, landmarks.shape)
    return landmarks + noise


def dropout_landmarks(
    landmarks: np.ndarray,
    dropout_rate: float = 0.1
) -> np.ndarray:
    """
    Randomly drop landmarks to simulate occlusion.
    
    Args:
        landmarks: Landmarks array (seq_len, landmark_dim)
        dropout_rate: Probability of dropping each landmark
        
    Returns:
        Landmarks with dropout applied
    """
    mask = np.random.random(landmarks.shape) > dropout_rate
    return landmarks * mask


def time_warp_sequence(
    sequence: np.ndarray,
    warp_factor: float
) -> np.ndarray:
    """
    Apply time warping to sequence.
    
    Args:
        sequence: Input sequence (seq_len, landmark_dim)
        warp_factor: Time warp factor
        
    Returns:
        Time-warped sequence
    """
    original_length = sequence.shape[0]
    new_length = int(original_length * warp_factor)
    
    if new_length <= 0:
        return sequence
    
    indices = np.linspace(0, original_length - 1, new_length)
    warped = np.zeros((new_length, sequence.shape[1]), dtype=np.float32)
    
    for i in range(sequence.shape[1]):
        warped[:, i] = np.interp(indices, np.arange(original_length), sequence[:, i])
    
    # Resample back to original length
    if warped.shape[0] != original_length:
        indices = np.linspace(0, warped.shape[0] - 1, original_length)
        resampled = np.zeros((original_length, sequence.shape[1]), dtype=np.float32)
        for i in range(sequence.shape[1]):
            resampled[:, i] = np.interp(indices, np.arange(warped.shape[0]), warped[:, i])
        return resampled
    
    return warped


def mixup_sequences(
    seq1: np.ndarray,
    seq2: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Mix two sequences with linear interpolation.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        alpha: Mixing coefficient
        
    Returns:
        Mixed sequence
    """
    return alpha * seq1 + (1 - alpha) * seq2


def cutmix_sequences(
    seq1: np.ndarray,
    seq2: np.ndarray,
    cut_ratio: float = 0.5
) -> np.ndarray:
    """
    Cut and mix two sequences.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        cut_ratio: Ratio of sequence to cut
        
    Returns:
        Mixed sequence
    """
    cut_point = int(seq1.shape[0] * cut_ratio)
    mixed = np.vstack([seq1[:cut_point], seq2[cut_point:]])
    return mixed


def normalize_sequence_range(
    sequence: np.ndarray,
    target_range: Tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """
    Normalize sequence to target range.
    
    Args:
        sequence: Input sequence
        target_range: Target range (min, max)
        
    Returns:
        Normalized sequence
    """
    min_val = np.min(sequence)
    max_val = np.max(sequence)
    
    if max_val - min_val == 0:
        return sequence
    
    normalized = (sequence - min_val) / (max_val - min_val)
    normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
    
    return normalized


def standardize_sequence(
    sequence: np.ndarray
) -> np.ndarray:
    """
    Standardize sequence (zero mean, unit variance).
    
    Args:
        sequence: Input sequence
        
    Returns:
        Standardized sequence
    """
    mean = np.mean(sequence)
    std = np.std(sequence)
    
    if std == 0:
        return sequence - mean
    
    return (sequence - mean) / std
