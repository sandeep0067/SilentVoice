"""
Quality Filter for INCLUDE dataset.

Filters videos based on quality metrics.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Quality metrics for a video."""
    brightness: float
    contrast: float
    blur_score: float
    motion_score: float
    overall_quality: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'brightness': self.brightness,
            'contrast': self.contrast,
            'blur_score': self.blur_score,
            'motion_score': self.motion_score,
            'overall_quality': self.overall_quality
        }


class QualityFilter:
    """Filters videos based on quality metrics."""
    
    def __init__(
        self,
        min_brightness: float = 50.0,
        max_brightness: float = 200.0,
        min_contrast: float = 30.0,
        max_blur_score: float = 100.0,
        min_motion_score: float = 10.0
    ):
        """
        Initialize quality filter.
        
        Args:
            min_brightness: Minimum brightness (0-255)
            max_brightness: Maximum brightness (0-255)
            min_contrast: Minimum contrast
            max_blur_score: Maximum blur score (lower is sharper)
            min_motion_score: Minimum motion score (higher is more motion)
        """
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.max_blur_score = max_blur_score
        self.min_motion_score = min_motion_score
    
    def calculate_quality_metrics(self, video_path: str) -> QualityMetrics:
        """
        Calculate quality metrics for a video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            QualityMetrics object
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frames = []
        frame_count = 0
        max_frames = 30  # Sample 30 frames for efficiency
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_count += 1
        
        cap.release()
        
        if not frames:
            raise ValueError("No frames extracted from video")
        
        # Calculate metrics
        brightness = self._calculate_brightness(frames)
        contrast = self._calculate_contrast(frames)
        blur_score = self._calculate_blur_score(frames)
        motion_score = self._calculate_motion_score(frames)
        
        # Determine overall quality
        overall_quality = self._assess_overall_quality(
            brightness, contrast, blur_score, motion_score
        )
        
        return QualityMetrics(
            brightness=brightness,
            contrast=contrast,
            blur_score=blur_score,
            motion_score=motion_score,
            overall_quality=overall_quality
        )
    
    def _calculate_brightness(self, frames: List[np.ndarray]) -> float:
        """Calculate average brightness."""
        brightness_values = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            brightness_values.append(brightness)
        
        return np.mean(brightness_values)
    
    def _calculate_contrast(self, frames: List[np.ndarray]) -> float:
        """Calculate average contrast (standard deviation)."""
        contrast_values = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            contrast = np.std(gray)
            contrast_values.append(contrast)
        
        return np.mean(contrast_values)
    
    def _calculate_blur_score(self, frames: List[np.ndarray]) -> float:
        """Calculate blur score using Laplacian variance."""
        blur_scores = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            blur_scores.append(variance)
        
        # Lower variance means more blur
        return np.mean(blur_scores)
    
    def _calculate_motion_score(self, frames: List[np.ndarray]) -> float:
        """Calculate motion score based on frame differences."""
        if len(frames) < 2:
            return 0.0
        
        motion_scores = []
        
        for i in range(len(frames) - 1):
            frame1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            diff = cv2.absdiff(frame1, frame2)
            motion = np.mean(diff)
            motion_scores.append(motion)
        
        return np.mean(motion_scores)
    
    def _assess_overall_quality(
        self,
        brightness: float,
        contrast: float,
        blur_score: float,
        motion_score: float
    ) -> str:
        """Assess overall quality based on metrics."""
        issues = []
        
        if brightness < self.min_brightness or brightness > self.max_brightness:
            issues.append("lighting")
        
        if contrast < self.min_contrast:
            issues.append("contrast")
        
        if blur_score < 50:  # Too blurry
            issues.append("blur")
        
        if motion_score < self.min_motion_score:
            issues.append("static")
        
        if not issues:
            return "high"
        elif len(issues) == 1:
            return "medium"
        else:
            return "low"
    
    def filter_video(self, video_path: str) -> tuple[bool, str, QualityMetrics]:
        """
        Filter video based on quality metrics.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (is_accepted, reason, metrics)
        """
        try:
            metrics = self.calculate_quality_metrics(video_path)
            
            if metrics.overall_quality == "low":
                return False, "Low overall quality", metrics
            
            if metrics.blur_score < 50:
                return False, "Too blurry", metrics
            
            if metrics.motion_score < self.min_motion_score:
                return False, "Insufficient motion", metrics
            
            return True, "Accepted", metrics
        
        except Exception as e:
            return False, f"Error: {str(e)}", None
    
    def batch_filter_videos(
        self,
        video_paths: List[str]
    ) -> Dict[str, tuple[bool, str, Optional[QualityMetrics]]]:
        """
        Filter multiple videos.
        
        Args:
            video_paths: List of video paths
            
        Returns:
            Dictionary mapping video paths to filter results
        """
        results = {}
        
        for video_path in video_paths:
            results[video_path] = self.filter_video(video_path)
        
        return results
