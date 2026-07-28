"""
Landmark processing utilities.
"""

import numpy as np
from typing import List, Tuple, Optional


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks to [0, 1] range.
    
    Args:
        landmarks: Landmarks array
        
    Returns:
        Normalized landmarks
    """
    # MediaPipe already provides normalized coordinates
    # This is a placeholder for additional normalization
    return landmarks


def make_relative_to_wrist(landmarks: np.ndarray) -> np.ndarray:
    """
    Make hand landmarks relative to wrist position.
    
    Args:
        landmarks: Landmarks array (first 63 elements are hand landmarks)
        
    Returns:
        Relative landmarks
    """
    if landmarks.shape[0] < 21:
        return landmarks
    
    # Use first landmark (wrist) as reference
    wrist = landmarks[0]
    relative_landmarks = landmarks - wrist
    
    return relative_landmarks


def smooth_landmarks(
    landmarks: List[np.ndarray],
    window_size: int = 5
) -> List[np.ndarray]:
    """
    Apply moving average smoothing to landmark sequence.
    
    Args:
        landmarks: List of landmark arrays
        window_size: Size of smoothing window
        
    Returns:
        Smoothed landmarks
    """
    smoothed = []
    
    for i in range(len(landmarks)):
        start = max(0, i - window_size // 2)
        end = min(len(landmarks), i + window_size // 2 + 1)
        window = landmarks[start:end]
        
        smoothed_lm = np.mean(window, axis=0)
        smoothed.append(smoothed_lm)
    
    return smoothed


def interpolate_missing_landmarks(
    landmarks: List[Optional[np.ndarray]],
    method: str = 'linear'
) -> List[np.ndarray]:
    """
    Interpolate missing landmarks in sequence.
    
    Args:
        landmarks: List of landmark arrays (None for missing)
        method: Interpolation method ('linear', 'nearest')
        
    Returns:
        Interpolated landmarks
    """
    interpolated = []
    
    for i, lm in enumerate(landmarks):
        if lm is not None:
            interpolated.append(lm)
        else:
            # Find nearest valid landmarks
            prev_idx = i - 1
            next_idx = i + 1
            
            while prev_idx >= 0 and landmarks[prev_idx] is None:
                prev_idx -= 1
            
            while next_idx < len(landmarks) and landmarks[next_idx] is None:
                next_idx += 1
            
            if prev_idx >= 0 and next_idx < len(landmarks):
                # Linear interpolation
                prev_lm = landmarks[prev_idx]
                next_lm = landmarks[next_idx]
                alpha = (i - prev_idx) / (next_idx - prev_idx)
                interpolated_lm = (1 - alpha) * prev_lm + alpha * next_lm
                interpolated.append(interpolated_lm)
            elif prev_idx >= 0:
                # Use previous landmark
                interpolated.append(landmarks[prev_idx])
            elif next_idx < len(landmarks):
                # Use next landmark
                interpolated.append(landmarks[next_idx])
            else:
                # No valid landmarks, use zeros
                interpolated.append(np.zeros_like(landmarks[0]))
    
    return interpolated


def calculate_landmark_velocity(
    landmarks: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Calculate velocity between consecutive landmarks.
    
    Args:
        landmarks: List of landmark arrays
        
    Returns:
        List of velocity arrays
    """
    velocities = []
    
    for i in range(len(landmarks) - 1):
        velocity = landmarks[i + 1] - landmarks[i]
        velocities.append(velocity)
    
    # Pad last velocity
    if velocities:
        velocities.append(velocities[-1])
    
    return velocities


def calculate_landmark_acceleration(
    landmarks: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Calculate acceleration between consecutive landmarks.
    
    Args:
        landmarks: List of landmark arrays
        
    Returns:
        List of acceleration arrays
    """
    velocities = calculate_landmark_velocity(landmarks)
    accelerations = []
    
    for i in range(len(velocities) - 1):
        acceleration = velocities[i + 1] - velocities[i]
        accelerations.append(acceleration)
    
    # Pad last acceleration
    if accelerations:
        accelerations.append(accelerations[-1])
        accelerations.append(accelerations[-1])
    
    return accelerations


def calculate_landmark_distances(
    landmarks: np.ndarray,
    pairs: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Calculate distances between landmark pairs.
    
    Args:
        landmarks: Landmarks array
        pairs: List of landmark index pairs
        
    Returns:
        Array of distances
    """
    distances = []
    
    for idx1, idx2 in pairs:
        if idx1 < len(landmarks) and idx2 < len(landmarks):
            dist = np.linalg.norm(landmarks[idx1] - landmarks[idx2])
            distances.append(dist)
    
    return np.array(distances)


def filter_landmarks_by_confidence(
    landmarks: List[np.ndarray],
    confidences: List[float],
    threshold: float = 0.5
) -> List[np.ndarray]:
    """
    Filter landmarks by confidence threshold.
    
    Args:
        landmarks: List of landmark arrays
        confidences: List of confidence values
        threshold: Confidence threshold
        
    Returns:
        Filtered landmarks (None for low confidence)
    """
    filtered = []
    
    for lm, conf in zip(landmarks, confidences):
        if conf >= threshold:
            filtered.append(lm)
        else:
            filtered.append(None)
    
    return filtered


def pad_sequence_to_length(
    landmarks: List[np.ndarray],
    target_length: int,
    padding_value: float = 0.0
) -> np.ndarray:
    """
    Pad or truncate landmark sequence to target length.
    
    Args:
        landmarks: List of landmark arrays
        target_length: Target sequence length
        padding_value: Value to use for padding
        
    Returns:
        Padded/truncated sequence array
    """
    if len(landmarks) == 0:
        return np.zeros((target_length, landmarks[0].shape[0])) + padding_value
    
    landmark_dim = landmarks[0].shape[0]
    
    if len(landmarks) >= target_length:
        # Truncate
        return np.array(landmarks[:target_length], dtype=np.float32)
    else:
        # Pad
        padding = np.full(
            (target_length - len(landmarks), landmark_dim),
            padding_value,
            dtype=np.float32
        )
        return np.vstack([np.array(landmarks, dtype=np.float32), padding])
