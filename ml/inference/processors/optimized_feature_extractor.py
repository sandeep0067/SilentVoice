"""
Optimized MediaPipe feature extractor for ISL recognition.

Provides performance-optimized feature extraction with configurable
trade-offs between speed and accuracy.
"""

import cv2
import logging
import numpy as np
import mediapipe as mp
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from ml.inference.processors.holistic_feature_extractor import (
    HolisticFeatureExtractor,
    HolisticExtractionConfig,
    HolisticFeatures
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizedExtractionConfig(HolisticExtractionConfig):
    """
    Optimized configuration for feature extraction.
    
    Extends HolisticExtractionConfig with performance-specific settings.
    """
    # Performance optimizations
    use_lite_model: bool = True  # Use MediaPipe Lite model
    use_gpu: bool = True  # Use GPU acceleration if available
    skip_frames: int = 0  # Skip N frames between extractions
    use_threading: bool = False  # Use asynchronous processing
    
    # Resolution optimization
    target_width: int = 480  # Target processing width
    target_height: int = 360  # Target processing height
    
    # Feature optimization
    enable_face_mesh: bool = False  # Disable full face mesh (use subset only)
    refine_face_landmarks: bool = False  # Disable face refinement for speed


class OptimizedFeatureExtractor(HolisticFeatureExtractor):
    """
    Optimized feature extractor with performance enhancements.
    """
    
    def __init__(self, config: OptimizedExtractionConfig):
        """
        Initialize optimized feature extractor.
        
        Args:
            config: Optimized extraction configuration
        """
        super().__init__(config)
        self.optimized_config = config
        self.frame_count = 0
        self.last_features = None
        self.last_frame_time = 0
        
    def _create_holistic(self) -> mp.solutions.holistic.Holistic:
        """
        Create MediaPipe Holistic solution with optimizations.
        
        Returns:
            Configured Holistic solution
        """
        model_complexity = 0 if self.optimized_config.use_lite_model else 1
        
        return mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=self.optimized_config.refine_face_landmarks,
            min_detection_confidence=self.optimized_config.min_detection_confidence,
            min_tracking_confidence=self.optimized_config.min_tracking_confidence
        )
    
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process frame with optimizations.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Feature vector or None if extraction failed
        """
        self.frame_count += 1
        
        # Frame skipping optimization
        if self.optimized_config.skip_frames > 0:
            if self.frame_count % (self.optimized_config.skip_frames + 1) != 0:
                return self.last_features
        
        # Resize frame for faster processing
        if (frame.shape[1] != self.optimized_config.target_width or 
            frame.shape[0] != self.optimized_config.target_height):
            frame = cv2.resize(
                frame, 
                (self.optimized_config.target_width, self.optimized_config.target_height),
                interpolation=cv2.INTER_LINEAR
            )
        
        # Process frame
        features = super().process_frame(frame)
        
        if features is not None:
            self.last_features = features
            self.last_frame_time = self.frame_count
        
        return features
    
    def get_performance_stats(self) -> Dict[str, any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'frame_count': self.frame_count,
            'last_features_age': self.frame_count - self.last_frame_time,
            'config': {
                'use_lite_model': self.optimized_config.use_lite_model,
                'skip_frames': self.optimized_config.skip_frames,
                'target_resolution': f"{self.optimized_config.target_width}x{self.optimized_config.target_height}"
            }
        }


class FastFeatureExtractor(OptimizedFeatureExtractor):
    """
    Maximum performance feature extractor.
    
    Sacrifices some accuracy for maximum speed.
    """
    
    def __init__(self):
        """Initialize with performance-optimized config."""
        config = OptimizedExtractionConfig(
            model_complexity=0,  # Lite model
            enable_hands=True,
            enable_face=False,  # Disable face for speed
            enable_pose=False,  # Disable pose for speed
            use_lite_model=True,
            skip_frames=1,  # Skip every other frame
            target_width=320,  # Lower resolution
            target_height=240,
            refine_face_landmarks=False
        )
        super().__init__(config)


class BalancedFeatureExtractor(OptimizedFeatureExtractor):
    """
    Balanced performance feature extractor.
    
    Good balance between speed and accuracy.
    """
    
    def __init__(self):
        """Initialize with balanced config."""
        config = OptimizedExtractionConfig(
            model_complexity=0,  # Lite model
            enable_hands=True,
            enable_face=True,  # Keep face for NMFs
            enable_pose=False,  # Disable pose
            use_lite_model=True,
            skip_frames=0,  # No skipping
            target_width=480,
            target_height=360,
            refine_face_landmarks=False
        )
        super().__init__(config)


class AccurateFeatureExtractor(OptimizedFeatureExtractor):
    """
    Accuracy-focused feature extractor.
    
    Maintains maximum accuracy with moderate optimizations.
    """
    
    def __init__(self):
        """Initialize with accuracy-focused config."""
        config = OptimizedExtractionConfig(
            model_complexity=1,  # Full model
            enable_hands=True,
            enable_face=True,
            enable_pose=True,  # Enable all features
            use_lite_model=False,
            skip_frames=0,
            target_width=640,
            target_height=480,
            refine_face_landmarks=True
        )
        super().__init__(config)
