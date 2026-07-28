"""
Landmark schemas for data validation and type hints.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class HandLandmarks:
    """Hand landmarks data structure."""
    landmarks: List[Tuple[float, float, float]]  # 21 landmarks × 3 coordinates
    handedness: Optional[str] = None  # 'Left' or 'Right'
    confidence: float = 1.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self.landmarks, dtype=np.float32)


@dataclass
class FacialLandmarks:
    """Facial landmarks data structure."""
    landmarks: List[Tuple[float, float, float]]  # 10 key landmarks × 3 coordinates
    confidence: float = 1.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self.landmarks, dtype=np.float32)


@dataclass
class PoseLandmarks:
    """Pose landmarks data structure."""
    landmarks: List[Tuple[float, float, float]]  # 33 landmarks × 3 coordinates
    confidence: float = 1.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self.landmarks, dtype=np.float32)


@dataclass
class CombinedLandmarks:
    """Combined landmarks from all sources."""
    hand_landmarks: List[HandLandmarks]  # 2 hands
    facial_landmarks: Optional[FacialLandmarks] = None
    pose_landmarks: Optional[PoseLandmarks] = None
    timestamp: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to combined numpy array."""
        combined = []
        
        for hand in self.hand_landmarks:
            combined.extend(hand.landmarks)
        
        if self.facial_landmarks:
            combined.extend(self.facial_landmarks.landmarks)
        
        if self.pose_landmarks:
            combined.extend(self.pose_landmarks.landmarks)
        
        return np.array(combined, dtype=np.float32)
    
    def get_dimension(self) -> int:
        """Get total dimension of landmark vector."""
        dim = len(self.hand_landmarks) * 21 * 3
        
        if self.facial_landmarks:
            dim += len(self.facial_landmarks.landmarks)
        
        if self.pose_landmarks:
            dim += len(self.pose_landmarks.landmarks)
        
        return dim


@dataclass
class LandmarkSequence:
    """Sequence of landmarks over time."""
    landmarks: List[CombinedLandmarks]
    gesture_label: Optional[str] = None
    subject_id: Optional[str] = None
    video_id: Optional[str] = None
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array of shape (sequence_length, landmark_dim)."""
        return np.array([lm.to_array() for lm in self.landmarks], dtype=np.float32)
    
    def __len__(self) -> int:
        """Get sequence length."""
        return len(self.landmarks)
