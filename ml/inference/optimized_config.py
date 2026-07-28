"""
Optimized configuration for inference pipeline.

This module provides pre-configured settings for optimal performance
while maintaining prediction accuracy.
"""

from dataclasses import dataclass
from typing import Optional
from ml.inference.processors.holistic_feature_extractor import HolisticExtractionConfig
from ml.inference.realtime.pipeline import RealtimeConfig


@dataclass
class OptimizedInferenceConfig:
    """
    Optimized configuration for real-time inference.
    
    Balances performance and accuracy for smooth real-time operation.
    """
    
    # MediaPipe settings (optimized for speed)
    mediapipe_model_complexity: int = 0  # Lite model (2-3x faster)
    mediapipe_enable_hands: bool = True  # Essential for ISL
    mediapipe_enable_face: bool = True  # Important for NMFs
    mediapipe_enable_pose: bool = False  # Disabled for speed (can be enabled if needed)
    mediapipe_min_detection_confidence: float = 0.5
    mediapipe_min_tracking_confidence: float = 0.5
    
    # Camera settings (optimized for speed)
    camera_width: int = 480  # Reduced from 640
    camera_height: int = 360  # Reduced from 480
    camera_fps: int = 30
    
    # Sliding window settings (optimized for latency)
    window_size: int = 20  # Reduced from 30 for lower latency
    window_stride: int = 2  # Skip frames for speed
    smoothing_window: int = 5
    
    # Confidence threshold
    confidence_threshold: float = 0.5
    
    # Performance settings
    use_gpu: bool = True  # Use GPU if available
    use_mixed_precision: bool = True  # FP16 for faster inference
    use_torch_compile: bool = True  # Compile model for speed (PyTorch 2.0+)
    frame_skip: int = 1  # Process every Nth frame (1 = all frames)
    
    # Memory settings
    max_frame_buffer_size: int = 5  # Limit memory usage
    
    @classmethod
    def get_performance_config(cls) -> 'OptimizedInferenceConfig':
        """
        Get configuration optimized for maximum performance.
        
        Sacrifices some accuracy for maximum speed.
        """
        return cls(
            mediapipe_model_complexity=0,
            mediapipe_enable_hands=True,
            mediapipe_enable_face=False,  # Disable face for speed
            mediapipe_enable_pose=False,
            camera_width=320,  # Lower resolution
            camera_height=240,
            window_size=15,  # Smaller window
            window_stride=3,  # More aggressive skipping
            frame_skip=2,  # Skip frames
            use_mixed_precision=True,
            use_torch_compile=True
        )
    
    @classmethod
    def get_balanced_config(cls) -> 'OptimizedInferenceConfig':
        """
        Get configuration balancing performance and accuracy.
        
        Recommended for most use cases.
        """
        return cls()  # Default values are balanced
    
    @classmethod
    def get_accuracy_config(cls) -> 'OptimizedInferenceConfig':
        """
        Get configuration optimized for accuracy.
        
        Slower but maintains maximum accuracy.
        """
        return cls(
            mediapipe_model_complexity=1,  # Full model
            mediapipe_enable_hands=True,
            mediapipe_enable_face=True,
            mediapipe_enable_pose=True,  # Enable all features
            camera_width=640,  # Full resolution
            camera_height=480,
            window_size=30,  # Full window
            window_stride=1,  # No skipping
            frame_skip=1,
            use_mixed_precision=False  # FP32 for accuracy
        )
    
    def to_holistic_config(self) -> HolisticExtractionConfig:
        """Convert to HolisticExtractionConfig."""
        return HolisticExtractionConfig(
            model_complexity=self.mediapipe_model_complexity,
            min_detection_confidence=self.mediapipe_min_detection_confidence,
            min_tracking_confidence=self.mediapipe_min_tracking_confidence,
            enable_hands=self.mediapipe_enable_hands,
            enable_face=self.mediapipe_enable_face,
            enable_pose=self.mediapipe_enable_pose
        )
    
    def to_realtime_config(self, class_names: list) -> RealtimeConfig:
        """Convert to RealtimeConfig."""
        return RealtimeConfig(
            window_size=self.window_size,
            window_stride=self.window_stride,
            smoothing_window=self.smoothing_window,
            confidence_threshold=self.confidence_threshold,
            camera_width=self.camera_width,
            camera_height=self.camera_height,
            camera_id=0,
            class_names=class_names
        )


# Pre-configured presets
PERFORMANCE_CONFIG = OptimizedInferenceConfig.get_performance_config()
BALANCED_CONFIG = OptimizedInferenceConfig.get_balanced_config()
ACCURACY_CONFIG = OptimizedInferenceConfig.get_accuracy_config()
