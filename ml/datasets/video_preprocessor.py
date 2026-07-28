"""
Video Preprocessor for INCLUDE dataset.

Handles video quality filtering, frame extraction, and preprocessing.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class VideoInfo:
    """Information about a video file."""
    path: str
    duration: float
    frame_count: int
    fps: float
    resolution: Tuple[int, int]
    codec: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'path': self.path,
            'duration': self.duration,
            'frame_count': self.frame_count,
            'fps': self.fps,
            'resolution': f"{self.resolution[0]}x{self.resolution[1]}",
            'codec': self.codec
        }


class VideoPreprocessor:
    """Preprocesses video files for landmark extraction."""
    
    def __init__(
        self,
        min_duration: float = 1.0,
        max_duration: float = 5.0,
        min_fps: float = 24.0,
        min_resolution: Tuple[int, int] = (640, 480),
        target_fps: float = 30.0
    ):
        """
        Initialize video preprocessor.
        
        Args:
            min_duration: Minimum video duration in seconds
            max_duration: Maximum video duration in seconds
            min_fps: Minimum frames per second
            min_resolution: Minimum resolution (width, height)
            target_fps: Target FPS for resampling
        """
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_fps = min_fps
        self.min_resolution = min_resolution
        self.target_fps = target_fps
    
    def get_video_info(self, video_path: str) -> Optional[VideoInfo]:
        """
        Get information about a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoInfo object or None if video cannot be opened
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        codec = cap.get(cv2.CAP_PROP_FOURCC)
        
        cap.release()
        
        return VideoInfo(
            path=video_path,
            duration=duration,
            frame_count=frame_count,
            fps=fps,
            resolution=(width, height),
            codec=str(codec)
        )
    
    def validate_video(self, video_path: str) -> Tuple[bool, str]:
        """
        Validate video meets quality requirements.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (is_valid, reason)
        """
        info = self.get_video_info(video_path)
        
        if info is None:
            return False, "Cannot open video file"
        
        if info.duration < self.min_duration:
            return False, f"Duration too short: {info.duration:.2f}s < {self.min_duration}s"
        
        if info.duration > self.max_duration:
            return False, f"Duration too long: {info.duration:.2f}s > {self.max_duration}s"
        
        if info.fps < self.min_fps:
            return False, f"FPS too low: {info.fps:.2f} < {self.min_fps}"
        
        if info.resolution[0] < self.min_resolution[0] or info.resolution[1] < self.min_resolution[1]:
            return False, f"Resolution too low: {info.resolution} < {self.min_resolution}"
        
        return True, "Valid"
    
    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame quality.
        
        Args:
            frame: Input frame
            
        Returns:
            Enhanced frame
        """
        # Convert to grayscale for histogram equalization
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            enhanced = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        else:
            enhanced = cv2.equalizeHist(frame)
        
        # Mild contrast enhancement
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=10)
        
        # Mild Gaussian blur for noise reduction
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced
    
    def resize_frame(
        self,
        frame: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Resize frame to target size.
        
        Args:
            frame: Input frame
            target_size: Target size (width, height), None to keep original
            
        Returns:
            Resized frame
        """
        if target_size is None:
            return frame
        
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    
    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame to [0, 1] range.
        
        Args:
            frame: Input frame
            
        Returns:
            Normalized frame
        """
        return frame.astype(np.float32) / 255.0
    
    def preprocess_frame(self, frame: np.ndarray, enhance: bool = True) -> np.ndarray:
        """
        Apply full preprocessing pipeline to frame.
        
        Args:
            frame: Input frame
            enhance: Whether to apply enhancement
            
        Returns:
            Preprocessed frame
        """
        if enhance:
            frame = self.enhance_frame(frame)
        
        frame = self.normalize_frame(frame)
        
        return frame
