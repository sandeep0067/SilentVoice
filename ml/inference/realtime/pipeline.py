"""
Real-time inference pipeline for ISL recognition.

Provides webcam input, MediaPipe feature extraction, sliding window prediction,
temporal smoothing, and FPS measurement for low-latency inference.
"""

import cv2
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np

import torch
import torch.nn as nn

from ml.inference.processors.holistic_feature_extractor import (
    HolisticFeatureExtractor,
    HolisticExtractionConfig,
    HolisticFeatures
)


logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Container for inference results."""
    predicted_class: int
    predicted_label: str
    confidence: float
    probabilities: np.ndarray
    timestamp: float
    inference_time_ms: float
    is_valid: bool = True


@dataclass 
class RealtimeConfig:
    """Configuration for real-time inference."""
    # Sliding window settings
    window_size: int = 30  # Number of frames in sliding window
    window_stride: int = 1  # Stride for sliding window
    
    # Temporal smoothing
    smoothing_window: int = 5  # Number of predictions to smooth
    smoothing_threshold: float = 0.3  # Minimum confidence for smoothing
    
    # Confidence threshold
    confidence_threshold: float = 0.5  # Minimum confidence to accept prediction
    
    # FPS settings
    target_fps: int = 30
    max_latency_ms: float = 100.0  # Maximum acceptable latency
    
    # Display settings
    display_landmarks: bool = True
    display_predictions: bool = True
    display_fps: bool = True
    display_confidence: bool = True
    
    # Camera settings
    camera_id: int = 0
    camera_width: int = 640
    camera_height: int = 480


class SlidingWindowBuffer:
    """Buffer for maintaining sliding window of features."""
    
    def __init__(self, window_size: int, feature_dim: int):
        """
        Initialize sliding window buffer.
        
        Args:
            window_size: Number of frames in window
            feature_dim: Dimension of feature vector
        """
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.buffer = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
    def add(self, features: np.ndarray, timestamp: float) -> bool:
        """
        Add features to buffer.
        
        Args:
            features: Feature vector
            timestamp: Frame timestamp
            
        Returns:
            True if buffer is full, False otherwise
        """
        self.buffer.append(features)
        self.timestamps.append(timestamp)
        return len(self.buffer) == self.window_size
    
    def get_window(self) -> Optional[np.ndarray]:
        """
        Get current window as array.
        
        Returns:
            Window array of shape (window_size, feature_dim) or None if not full
        """
        if len(self.buffer) < self.window_size:
            return None
        return np.array(self.buffer, dtype=np.float32)
    
    def get_timestamps(self) -> List[float]:
        """Get timestamps in buffer."""
        return list(self.timestamps)
    
    def clear(self) -> None:
        """Clear buffer."""
        self.buffer.clear()
        self.timestamps.clear()


class TemporalSmoother:
    """Temporal smoothing for predictions."""
    
    def __init__(self, window_size: int, threshold: float):
        """
        Initialize temporal smoother.
        
        Args:
            window_size: Number of predictions to smooth
            threshold: Minimum confidence for smoothing
        """
        self.window_size = window_size
        self.threshold = threshold
        self.history = deque(maxlen=window_size)
        
    def update(self, prediction: int, confidence: float) -> Tuple[int, float]:
        """
        Update smoother with new prediction.
        
        Args:
            prediction: Predicted class
            confidence: Prediction confidence
            
        Returns:
            Tuple of (smoothed_prediction, smoothed_confidence)
        """
        self.history.append((prediction, confidence))
        
        if len(self.history) < self.window_size:
            return prediction, confidence
        
        # Filter low-confidence predictions
        valid_predictions = [(p, c) for p, c in self.history if c >= self.threshold]
        
        if not valid_predictions:
            return prediction, confidence
        
        # Majority voting with confidence weighting
        from collections import Counter
        weighted_votes = Counter()
        total_confidence = 0.0
        
        for p, c in valid_predictions:
            weighted_votes[p] += c
            total_confidence += c
        
        if total_confidence == 0:
            return prediction, confidence
        
        # Get most voted class
        smoothed_prediction = weighted_votes.most_common(1)[0][0]
        smoothed_confidence = weighted_votes[smoothed_prediction] / total_confidence
        
        return smoothed_prediction, smoothed_confidence
    
    def clear(self) -> None:
        """Clear history."""
        self.history.clear()


class RealtimeInferencePipeline:
    """
    Real-time inference pipeline for ISL recognition.
    
    Independent from training pipeline, designed for low-latency inference.
    """
    
    def __init__(
        self,
        model: nn.Module,
        class_names: List[str],
        config: Optional[RealtimeConfig] = None,
        feature_extractor_config: Optional[HolisticExtractionConfig] = None
    ):
        """
        Initialize real-time inference pipeline.
        
        Args:
            model: Trained PyTorch model
            class_names: List of class names
            config: Realtime configuration
            feature_extractor_config: MediaPipe feature extraction config
        """
        self.model = model
        self.class_names = class_names
        self.config = config or RealtimeConfig()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize feature extractor
        self.feature_extractor = HolisticFeatureExtractor(feature_extractor_config)
        self.feature_dim = self.feature_extractor.get_feature_dimensions()['total']
        
        # Initialize sliding window buffer
        self.window_buffer = SlidingWindowBuffer(
            self.config.window_size,
            self.feature_dim
        )
        
        # Initialize temporal smoother
        self.smoother = TemporalSmoother(
            self.config.smoothing_window,
            self.config.smoothing_threshold
        )
        
        # FPS tracking
        self.frame_count = 0
        self.start_time = None
        self.fps = 0.0
        self.latency_ms = 0.0
        
        # Prediction history
        self.prediction_history = deque(maxlen=100)
        
        # Camera
        self.cap = None
        
        logger.info(f"Initialized real-time inference pipeline on {self.device}")
        logger.info(f"Feature dimension: {self.feature_dim}")
        logger.info(f"Window size: {self.config.window_size}")
    
    def initialize_camera(self) -> bool:
        """
        Initialize webcam camera.
        
        Returns:
            True if successful, False otherwise
        """
        self.cap = cv2.VideoCapture(self.config.camera_id)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {self.config.camera_id}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
        return True
    
    def extract_features(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract features from frame using MediaPipe.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Feature vector or None if extraction failed
        """
        try:
            timestamp = time.time()
            features = self.feature_extractor.extract_features(frame, timestamp)
            
            if features.is_valid():
                return features.get_feature_vector()
            else:
                logger.warning("Invalid features extracted")
                return None
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    def predict(self, features: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Make prediction from features.
        
        Args:
            features: Feature vector or window of features
            
        Returns:
            Tuple of (predicted_class, confidence, probabilities)
        """
        # Ensure features are in correct shape
        if features.ndim == 1:
            features = features[np.newaxis, :]  # (1, feature_dim)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        # Add batch dimension if needed
        if features_tensor.ndim == 2:
            features_tensor = features_tensor.unsqueeze(0)  # (1, seq_len, feature_dim)
        
        # Inference
        with torch.no_grad():
            start_inference = time.time()
            logits = self.model(features_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)
            
            inference_time = (time.time() - start_inference) * 1000  # ms
        
        predicted_class = predicted.item()
        confidence_score = confidence.item()
        probabilities_array = probabilities.cpu().numpy()[0]
        
        self.latency_ms = inference_time
        
        return predicted_class, confidence_score, probabilities_array
    
    def process_frame(self, frame: np.ndarray) -> Optional[InferenceResult]:
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            InferenceResult or None if window not full
        """
        # Extract features
        features = self.extract_features(frame)
        if features is None:
            return None
        
        # Add to sliding window
        timestamp = time.time()
        window_full = self.window_buffer.add(features, timestamp)
        
        if not window_full:
            return None
        
        # Get window and predict
        window = self.window_buffer.get_window()
        if window is None:
            return None
        
        predicted_class, confidence, probabilities = self.predict(window)
        
        # Temporal smoothing
        smoothed_class, smoothed_confidence = self.smoother.update(
            predicted_class, confidence
        )
        
        # Create result
        result = InferenceResult(
            predicted_class=smoothed_class,
            predicted_label=self.class_names[smoothed_class],
            confidence=smoothed_confidence,
            probabilities=probabilities,
            timestamp=timestamp,
            inference_time_ms=self.latency_ms,
            is_valid=smoothed_confidence >= self.config.confidence_threshold
        )
        
        # Add to history
        self.prediction_history.append(result)
        
        return result
    
    def update_fps(self) -> None:
        """Update FPS measurement."""
        if self.start_time is None:
            self.start_time = time.time()
            self.frame_count = 0
            return
        
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        
        if elapsed >= 1.0:  # Update every second
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
    
    def draw_overlay(self, frame: np.ndarray, result: Optional[InferenceResult]) -> np.ndarray:
        """
        Draw overlay information on frame.
        
        Args:
            frame: Input frame
            result: Inference result
            
        Returns:
            Frame with overlay
        """
        overlay = frame.copy()
        
        # Draw FPS
        if self.config.display_fps:
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(overlay, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            latency_text = f"Latency: {self.latency_ms:.1f}ms"
            cv2.putText(overlay, latency_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw prediction
        if result is not None and self.config.display_predictions:
            label = result.predicted_label
            conf = result.confidence
            
            # Color based on confidence
            if conf >= 0.8:
                color = (0, 255, 0)  # Green
            elif conf >= 0.5:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red
            
            if self.config.display_confidence:
                pred_text = f"{label}: {conf:.2f}"
            else:
                pred_text = label
            
            cv2.putText(overlay, pred_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Draw confidence bar
            bar_width = int(conf * 200)
            cv2.rectangle(overlay, (10, 100), (10 + bar_width, 110), color, -1)
            cv2.rectangle(overlay, (10, 100), (210, 110), (255, 255, 255), 2)
        
        return overlay
    
    def run(self) -> None:
        """Run real-time inference loop."""
        if not self.initialize_camera():
            return
        
        logger.info("Starting real-time inference loop")
        logger.info("Press 'q' to quit")
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                # Process frame
                result = self.process_frame(frame)
                
                # Update FPS
                self.update_fps()
                
                # Draw overlay
                if self.config.display_landmarks or self.config.display_predictions:
                    frame = self.draw_overlay(frame, result)
                
                # Display frame
                cv2.imshow('ISL Recognition', frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit requested")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in inference loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.feature_extractor.close()
        logger.info("Pipeline cleanup completed")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'fps': self.fps,
            'latency_ms': self.latency_ms,
            'window_size': self.config.window_size,
            'buffer_fullness': len(self.window_buffer.buffer) / self.window_buffer.window_size if self.window_buffer.window_size > 0 else 0.0,
            'total_predictions': len(self.prediction_history),
            'valid_predictions': sum(1 for r in self.prediction_history if r.is_valid)
        }
