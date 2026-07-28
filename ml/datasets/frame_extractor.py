"""
Frame Extractor for INCLUDE dataset.

Extracts frames from videos with temporal resampling.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Generator
from dataclasses import dataclass


@dataclass
class FrameExtractionConfig:
    """Configuration for frame extraction."""
    target_fps: float = 30.0
    target_size: Optional[tuple] = None
    enhance_frames: bool = True
    normalize: bool = True
    skip_duplicates: bool = True
    duplicate_threshold: float = 0.95


class FrameExtractor:
    """Extracts frames from videos with temporal resampling."""
    
    def __init__(self, config: Optional[FrameExtractionConfig] = None):
        """
        Initialize frame extractor.
        
        Args:
            config: Frame extraction configuration
        """
        self.config = config or FrameExtractionConfig()
    
    def extract_frames(
        self,
        video_path: str,
        max_frames: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of frames
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices for target FPS
        if original_fps > 0:
            frame_indices = self._calculate_frame_indices(
                total_frames,
                original_fps,
                self.config.target_fps
            )
        else:
            frame_indices = list(range(total_frames))
        
        # Limit frames if specified
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]
        
        frames = []
        prev_frame = None
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Skip duplicate frames if enabled
            if self.config.skip_duplicates and prev_frame is not None:
                if self._are_frames_similar(frame, prev_frame, self.config.duplicate_threshold):
                    continue
            
            # Preprocess frame
            if self.config.enhance_frames:
                frame = self._enhance_frame(frame)
            
            if self.config.normalize:
                frame = self._normalize_frame(frame)
            
            if self.config.target_size is not None:
                frame = cv2.resize(frame, self.config.target_size)
            
            frames.append(frame)
            prev_frame = frame.copy()
        
        cap.release()
        
        return frames
    
    def extract_frames_generator(
        self,
        video_path: str,
        max_frames: Optional[int] = None
    ) -> Generator[np.ndarray, None, None]:
        """
        Extract frames from video as a generator (memory efficient).
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            
        Yields:
            Frames one at a time
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_indices = self._calculate_frame_indices(
            total_frames,
            original_fps,
            self.config.target_fps
        )
        
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]
        
        prev_frame = None
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            if self.config.skip_duplicates and prev_frame is not None:
                if self._are_frames_similar(frame, prev_frame, self.config.duplicate_threshold):
                    continue
            
            if self.config.enhance_frames:
                frame = self._enhance_frame(frame)
            
            if self.config.normalize:
                frame = self._normalize_frame(frame)
            
            if self.config.target_size is not None:
                frame = cv2.resize(frame, self.config.target_size)
            
            yield frame
            prev_frame = frame.copy()
        
        cap.release()
    
    def _calculate_frame_indices(
        self,
        total_frames: int,
        original_fps: float,
        target_fps: float
    ) -> List[int]:
        """
        Calculate frame indices for temporal resampling.
        
        Args:
            total_frames: Total number of frames in video
            original_fps: Original FPS
            target_fps: Target FPS
            
        Returns:
            List of frame indices to extract
        """
        if original_fps <= 0:
            return list(range(total_frames))
        
        # Calculate sampling interval
        interval = original_fps / target_fps
        
        # Generate frame indices
        indices = []
        current_idx = 0
        
        while current_idx < total_frames:
            indices.append(int(current_idx))
            current_idx += interval
        
        return indices
    
    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhance frame quality."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            enhanced = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        else:
            enhanced = cv2.equalizeHist(frame)
        
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=10)
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced
    
    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame to [0, 1] range."""
        return frame.astype(np.float32) / 255.0
    
    def _are_frames_similar(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        threshold: float
    ) -> bool:
        """
        Check if two frames are similar using structural similarity.
        
        Args:
            frame1: First frame
            frame2: Second frame
            threshold: Similarity threshold
            
        Returns:
            True if frames are similar
        """
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor((frame1 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor((frame2 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray1 = (frame1 * 255).astype(np.uint8)
            gray2 = (frame2 * 255).astype(np.uint8)
        
        # Calculate structural similarity
        try:
            from skimage.metrics import structural_similarity as ssim
            similarity = ssim(gray1, gray2)
            return similarity > threshold
        except ImportError:
            # Fallback to simple MSE
            mse = np.mean((gray1 - gray2) ** 2)
            return mse < (1 - threshold) * 1000
    
    def save_frames(
        self,
        frames: List[np.ndarray],
        output_dir: str,
        prefix: str = "frame"
    ) -> List[str]:
        """
        Save frames to disk.
        
        Args:
            frames: List of frames to save
            output_dir: Output directory
            prefix: Filename prefix
            
        Returns:
            List of saved frame paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for idx, frame in np.array(frames):
            # Convert back to uint8 if normalized
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = (frame * 255).astype(np.uint8)
            
            # Convert BGR to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            filename = f"{prefix}_{idx:04d}.png"
            frame_path = output_path / filename
            cv2.imwrite(str(frame_path), frame)
            saved_paths.append(str(frame_path))
        
        return saved_paths
