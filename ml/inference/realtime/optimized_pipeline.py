"""
Optimized real-time inference pipeline for ISL recognition.

Provides performance-optimized inference with asynchronous processing,
frame skipping, and other enhancements while maintaining accuracy.
"""

import cv2
import time
import logging
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np

import torch
import torch.nn as nn

from ml.inference.processors.optimized_feature_extractor import (
    OptimizedFeatureExtractor,
    OptimizedExtractionConfig,
    FastFeatureExtractor,
    BalancedFeatureExtractor,
    AccurateFeatureExtractor
)
from ml.inference.realtime.pipeline import RealtimeConfig, InferenceResult


logger = logging.getLogger(__name__)


@dataclass
class OptimizedPipelineConfig(RealtimeConfig):
    """
    Optimized configuration for real-time inference pipeline.
    """
    # Performance settings
    use_async_processing: bool = True  # Use separate threads for capture and inference
    max_frame_queue_size: int = 2  # Limit queue size to prevent memory buildup
    frame_skip: int = 1  # Skip N frames (0 = process all)
    
    # Threading settings
    capture_thread_enabled: bool = True
    inference_thread_enabled: bool = True
    
    # Feature extractor type
    extractor_type: str = 'balanced'  # 'fast', 'balanced', 'accurate'


class OptimizedInferencePipeline:
    """
    Optimized real-time inference pipeline with performance enhancements.
    """
    
    def __init__(self, config: OptimizedPipelineConfig):
        """
        Initialize optimized inference pipeline.
        
        Args:
            config: Optimized pipeline configuration
        """
        self.config = config
        self.running = False
        self.lock = threading.Lock()
        
        # Feature extractor
        self.extractor = self._create_extractor()
        
        # Threading queues
        self.frame_queue = queue.Queue(maxsize=config.max_frame_queue_size)
        self.result_queue = queue.Queue(maxsize=config.max_frame_queue_size)
        
        # Threads
        self.capture_thread = None
        self.inference_thread = None
        
        # Performance tracking
        self.frame_count = 0
        self.processed_count = 0
        self.fps = 0.0
        self.last_fps_update = time.time()
        
        # Sliding window
        self.feature_window = deque(maxlen=config.window_size)
        
        # Model (to be loaded separately)
        self.model = None
        self.device = None
        
    def _create_extractor(self) -> OptimizedFeatureExtractor:
        """Create feature extractor based on config."""
        extractor_type = self.config.extractor_type.lower()
        
        if extractor_type == 'fast':
            logger.info("Using FastFeatureExtractor (maximum performance)")
            return FastFeatureExtractor()
        elif extractor_type == 'accurate':
            logger.info("Using AccurateFeatureExtractor (maximum accuracy)")
            return AccurateFeatureExtractor()
        else:  # balanced
            logger.info("Using BalancedFeatureExtractor (balanced)")
            return BalancedFeatureExtractor()
    
    def initialize(self) -> bool:
        """
        Initialize the pipeline.
        
        Returns:
            True if initialization successful
        """
        try:
            # Initialize feature extractor
            if not self.extractor.initialize():
                logger.error("Failed to initialize feature extractor")
                return False
            
            # Initialize camera
            self.cap = cv2.VideoCapture(self.config.camera_id)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.config.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
            
            logger.info(f"Camera initialized: {self.config.camera_width}x{self.config.camera_height}")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            return False
    
    def load_model(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        Load inference model.
        
        Args:
            model: PyTorch model
            device: Device to run inference on
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")
    
    def _capture_thread_func(self):
        """Camera capture thread function."""
        logger.info("Capture thread started")
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Frame skipping
                self.frame_count += 1
                if self.config.frame_skip > 0:
                    if self.frame_count % (self.config.frame_skip + 1) != 0:
                        continue
                
                # Add to queue (non-blocking)
                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    # Drop frame if queue is full
                    pass
                    
            except Exception as e:
                logger.error(f"Capture thread error: {e}")
                break
        
        logger.info("Capture thread stopped")
    
    def _inference_thread_func(self):
        """Inference thread function."""
        logger.info("Inference thread started")
        
        while self.running:
            try:
                # Get frame from queue (blocking with timeout)
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Extract features
                features = self.extractor.process_frame(frame)
                if features is None:
                    continue
                
                # Add to sliding window
                self.feature_window.append(features)
                
                # Run inference if window is full
                if len(self.feature_window) == self.config.window_size:
                    result = self._run_inference()
                    if result:
                        self.result_queue.put(result)
                        self.processed_count += 1
                
                # Update FPS
                current_time = time.time()
                if current_time - self.last_fps_update >= 1.0:
                    self.fps = self.processed_count / (current_time - self.last_fps_update)
                    self.last_fps_update = current_time
                    self.processed_count = 0
                    
            except Exception as e:
                logger.error(f"Inference thread error: {e}")
                break
        
        logger.info("Inference thread stopped")
    
    def _run_inference(self) -> Optional[InferenceResult]:
        """
        Run model inference on current window.
        
        Returns:
            Inference result or None
        """
        if self.model is None:
            return None
        
        try:
            # Prepare input
            window_features = np.array(list(self.feature_window))
            input_tensor = torch.from_numpy(window_features).float().unsqueeze(0).to(self.device)
            
            # Run inference
            start_time = time.time()
            with torch.no_grad():
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        output = self.model(input_tensor)
                else:
                    output = self.model(input_tensor)
            
            inference_time_ms = (time.time() - start_time) * 1000
            
            # Get prediction
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
            
            # Get label
            predicted_label = self.config.class_names[predicted_class] if self.config.class_names else str(predicted_class)
            
            return InferenceResult(
                predicted_class=int(predicted_class),
                predicted_label=predicted_label,
                confidence=float(confidence),
                probabilities=probabilities,
                timestamp=time.time(),
                inference_time_ms=inference_time_ms,
                is_valid=confidence >= self.config.confidence_threshold
            )
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return None
    
    def start(self):
        """Start the pipeline."""
        if self.running:
            logger.warning("Pipeline already running")
            return
        
        self.running = True
        
        # Start threads if enabled
        if self.config.use_async_processing:
            if self.config.capture_thread_enabled:
                self.capture_thread = threading.Thread(target=self._capture_thread_func, daemon=True)
                self.capture_thread.start()
            
            if self.config.inference_thread_enabled:
                self.inference_thread = threading.Thread(target=self._inference_thread_func, daemon=True)
                self.inference_thread.start()
        
        logger.info("Pipeline started")
    
    def stop(self):
        """Stop the pipeline."""
        self.running = False
        
        # Wait for threads to finish
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.inference_thread:
            self.inference_thread.join(timeout=2.0)
        
        # Release camera
        if hasattr(self, 'cap'):
            self.cap.release()
        
        logger.info("Pipeline stopped")
    
    def process_frame(self, frame: np.ndarray) -> Optional[InferenceResult]:
        """
        Process a single frame (synchronous mode).
        
        Args:
            frame: Input frame
            
        Returns:
            Inference result or None
        """
        if self.config.use_async_processing:
            # In async mode, return latest result from queue
            try:
                return self.result_queue.get_nowait()
            except queue.Empty:
                return None
        else:
            # Synchronous processing
            features = self.extractor.process_frame(frame)
            if features is None:
                return None
            
            self.feature_window.append(features)
            
            if len(self.feature_window) == self.config.window_size:
                return self._run_inference()
            
            return None
    
    def get_latest_result(self) -> Optional[InferenceResult]:
        """
        Get the latest inference result (async mode).
        
        Returns:
            Latest result or None
        """
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_fps(self) -> float:
        """Get current FPS."""
        return self.fps
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'fps': self.fps,
            'frame_count': self.frame_count,
            'queue_size': self.frame_queue.qsize(),
            'result_queue_size': self.result_queue.qsize(),
            'window_size': len(self.feature_window),
            'extractor_stats': self.extractor.get_performance_stats() if hasattr(self.extractor, 'get_performance_stats') else {}
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.stop()
        if hasattr(self.extractor, 'cleanup'):
            self.extractor.cleanup()


# Convenience functions for creating optimized pipelines

def create_fast_pipeline(class_names: List[str]) -> OptimizedInferencePipeline:
    """
    Create pipeline optimized for maximum performance.
    
    Args:
        class_names: List of class names
        
    Returns:
        Optimized pipeline
    """
    config = OptimizedPipelineConfig(
        extractor_type='fast',
        window_size=15,
        window_stride=2,
        frame_skip=2,
        camera_width=320,
        camera_height=240,
        use_async_processing=True,
        class_names=class_names
    )
    return OptimizedInferencePipeline(config)


def create_balanced_pipeline(class_names: List[str]) -> OptimizedInferencePipeline:
    """
    Create pipeline with balanced performance and accuracy.
    
    Args:
        class_names: List of class names
        
    Returns:
        Optimized pipeline
    """
    config = OptimizedPipelineConfig(
        extractor_type='balanced',
        window_size=20,
        window_stride=1,
        frame_skip=0,
        camera_width=480,
        camera_height=360,
        use_async_processing=True,
        class_names=class_names
    )
    return OptimizedInferencePipeline(config)


def create_accurate_pipeline(class_names: List[str]) -> OptimizedInferencePipeline:
    """
    Create pipeline optimized for accuracy.
    
    Args:
        class_names: List of class names
        
    Returns:
        Optimized pipeline
    """
    config = OptimizedPipelineConfig(
        extractor_type='accurate',
        window_size=30,
        window_stride=1,
        frame_skip=0,
        camera_width=640,
        camera_height=480,
        use_async_processing=False,  # Synchronous for accuracy
        class_names=class_names
    )
    return OptimizedInferencePipeline(config)
