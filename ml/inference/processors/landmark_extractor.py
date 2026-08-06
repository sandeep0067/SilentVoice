"""
Landmark Extractor using MediaPipe.

Extracts hand and facial landmarks from video frames for ISL recognition.
"""

import cv2
import numpy as np
import mediapipe as mp
try:
    import mediapipe.python.solutions as mp_solutions
except ImportError:
    mp_solutions = mp.solutions
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class LandmarkExtractionConfig:
    """Configuration for landmark extraction."""
    # Hands configuration
    max_num_hands: int = 2
    model_complexity: int = 1  # 0=Lite, 1=Full, 2=Heavy
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    
    # Face configuration
    max_num_faces: int = 1
    refine_landmarks: bool = True
    
    # Pose configuration (optional)
    enable_pose: bool = False
    model_complexity_pose: int = 1
    min_detection_confidence_pose: float = 0.5
    
    # Output configuration
    include_z_coordinates: bool = True
    normalize_coordinates: bool = True
    use_relative_coordinates: bool = False


class LandmarkExtractor:
    """Extracts landmarks using MediaPipe."""
    
    # Key facial landmark indices (lips, eyes, eyebrows)
    FACIAL_KEY_INDICES = [
        13, 14,  # Upper lip, lower lip
        61, 146,  # Inner lip corners
        159, 145,  # Outer lip corners
        33, 263,  # Eye corners
        291,  # Right eye corner
        133,  # Left eye corner
    ]
    
    def __init__(self, config: Optional[LandmarkExtractionConfig] = None):
        """
        Initialize landmark extractor.
        
        Args:
            config: Landmark extraction configuration
        """
        self.config = config or LandmarkExtractionConfig()
        
        # Initialize MediaPipe solutions
        self.mp_hands = mp_solutions.hands
        self.mp_face_mesh = mp_solutions.face_mesh
        self.mp_pose = mp_solutions.pose if self.config.enable_pose else None
        self.mp_drawing = mp_solutions.drawing_utils
        
        # Initialize detectors
        self.hands_detector = None
        self.face_detector = None
        self.pose_detector = None
        
        self._initialize_detectors()
    
    def _initialize_detectors(self) -> None:
        """Initialize MediaPipe detectors."""
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.max_num_hands,
            model_complexity=self.config.model_complexity,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence
        )
        
        self.face_detector = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.config.max_num_faces,
            refine_landmarks=self.config.refine_landmarks,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence
        )
        
        if self.config.enable_pose:
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.config.model_complexity_pose,
                smooth_landmarks=True,
                min_detection_confidence=self.config.min_detection_confidence_pose,
                min_tracking_confidence=self.config.min_tracking_confidence
            )
    
    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract landmarks from a single frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Combined landmarks array or None if extraction fails
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract hand landmarks
        hand_landmarks = self._extract_hand_landmarks(rgb_frame)
        
        # Extract facial landmarks
        facial_landmarks = self._extract_facial_landmarks(rgb_frame)
        
        # Extract pose landmarks (optional)
        pose_landmarks = self._extract_pose_landmarks(rgb_frame) if self.config.enable_pose else None
        
        # Combine landmarks
        combined = self._combine_landmarks(hand_landmarks, facial_landmarks, pose_landmarks)
        
        return combined
    
    def _extract_hand_landmarks(self, rgb_frame: np.ndarray) -> List[List[float]]:
        """
        Extract hand landmarks.
        
        Args:
            rgb_frame: RGB frame
            
        Returns:
            List of hand landmarks (2 hands × 21 landmarks × 3 coordinates)
        """
        results = self.hands_detector.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            # Return zeros if no hands detected
            return [[[0.0, 0.0, 0.0] for _ in range(21)] for _ in range(2)]
        
        hand_landmarks = []
        
        for hand_lm in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_lm.landmark:
                coords = [lm.x, lm.y]
                if self.config.include_z_coordinates:
                    coords.append(lm.z)
                landmarks.append(coords)
            hand_landmarks.append(landmarks)
        
        # Pad to 2 hands if fewer detected
        while len(hand_landmarks) < 2:
            hand_landmarks.append([[0.0, 0.0, 0.0] for _ in range(21)])
        
        return hand_landmarks
    
    def _extract_facial_landmarks(self, rgb_frame: np.ndarray) -> List[List[float]]:
        """
        Extract key facial landmarks.
        
        Args:
            rgb_frame: RGB frame
            
        Returns:
            List of facial landmarks (10 landmarks × 3 coordinates)
        """
        results = self.face_detector.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return [[0.0, 0.0, 0.0] for _ in range(len(self.FACIAL_KEY_INDICES))]
        
        face_lm = results.multi_face_landmarks[0]
        facial_landmarks = []
        
        for idx in self.FACIAL_KEY_INDICES:
            lm = face_lm.landmark[idx]
            coords = [lm.x, lm.y]
            if self.config.include_z_coordinates:
                coords.append(lm.z)
            facial_landmarks.append(coords)
        
        return facial_landmarks
    
    def _extract_pose_landmarks(self, rgb_frame: np.ndarray) -> Optional[List[List[float]]]:
        """
        Extract pose landmarks (optional).
        
        Args:
            rgb_frame: RGB frame
            
        Returns:
            List of pose landmarks or None
        """
        if self.pose_detector is None:
            return None
        
        results = self.pose_detector.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        pose_landmarks = []
        for lm in results.pose_landmarks.landmark:
            coords = [lm.x, lm.y]
            if self.config.include_z_coordinates:
                coords.append(lm.z)
            pose_landmarks.append(coords)
        
        return pose_landmarks
    
    def _combine_landmarks(
        self,
        hand_landmarks: List[List[List[float]]],
        facial_landmarks: List[List[float]],
        pose_landmarks: Optional[List[List[float]]]
    ) -> np.ndarray:
        """
        Combine all landmarks into a single array.
        
        Args:
            hand_landmarks: Hand landmarks
            facial_landmarks: Facial landmarks
            pose_landmarks: Pose landmarks (optional)
            
        Returns:
            Combined landmarks array
        """
        combined = []
        
        # Add hand landmarks (2 hands × 21 landmarks)
        for hand in hand_landmarks:
            combined.extend(hand)
        
        # Add facial landmarks (10 landmarks)
        combined.extend(facial_landmarks)
        
        # Add pose landmarks if enabled
        if pose_landmarks is not None:
            combined.extend(pose_landmarks)
        
        # Convert to numpy array
        landmarks_array = np.array(combined, dtype=np.float32)
        
        # Apply normalization if enabled
        if self.config.normalize_coordinates:
            landmarks_array = self._normalize_landmarks(landmarks_array)
        
        # Apply relative coordinates if enabled
        if self.config.use_relative_coordinates:
            landmarks_array = self._make_relative(landmarks_array)
        
        # Flatten per-frame landmarks (N_landmarks, 3) into a 1D feature vector
        # so sequences stack cleanly as (T, D) for downstream augmentation/training
        landmarks_array = landmarks_array.flatten()
        
        return landmarks_array
    
    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks to [0, 1] range.
        
        Args:
            landmarks: Landmarks array
            
        Returns:
            Normalized landmarks
        """
        # MediaPipe already provides normalized coordinates
        # This is a placeholder for additional normalization if needed
        return landmarks
    
    def _make_relative(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Make coordinates relative to wrist (first hand landmark).
        
        Args:
            landmarks: Landmarks array
            
        Returns:
            Relative coordinates
        """
        if len(landmarks) < 21:
            return landmarks
        
        # Use first hand's wrist as reference
        wrist = landmarks[0]
        
        # Subtract wrist position from all landmarks
        relative_landmarks = landmarks - wrist
        
        return relative_landmarks
    
    def extract_landmarks_batch(
        self,
        frames: List[np.ndarray]
    ) -> List[Optional[np.ndarray]]:
        """
        Extract landmarks from multiple frames.
        
        Args:
            frames: List of frames
            
        Returns:
            List of landmarks arrays
        """
        landmarks_list = []
        
        for frame in frames:
            landmarks = self.extract_landmarks(frame)
            landmarks_list.append(landmarks)
        
        return landmarks_list
    
    def visualize_landmarks(
        self,
        frame: np.ndarray,
        landmarks: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Visualize landmarks on frame.
        
        Args:
            frame: Input frame
            landmarks: Landmarks to visualize (if None, re-extract)
            
        Returns:
            Frame with landmarks drawn
        """
        if landmarks is None:
            # Re-extract landmarks for visualization
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands_detector.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_lm,
                        self.mp_hands.HAND_CONNECTIONS
                    )
            
            face_results = self.face_detector.process(rgb_frame)
            if face_results.multi_face_landmarks:
                for face_lm in face_results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        face_lm,
                        self.mp_face_mesh.FACEMESH_TESSELATION
                    )
        else:
            # Visualize from provided landmarks
            # This would require reconstructing MediaPipe structures
            # For now, just return the original frame
            pass
        
        return frame
    
    def get_landmark_dimensions(self) -> int:
        """
        Get the dimension of the landmark vector.
        
        Returns:
            Number of dimensions in landmark vector
        """
        # Hands: 2 hands × 21 landmarks × 3 coords = 126
        # Face: 10 landmarks × 3 coords = 30
        # Pose: 33 landmarks × 3 coords = 99 (if enabled)
        
        hand_dim = 2 * 21 * 3 if self.config.include_z_coordinates else 2 * 21 * 2
        face_dim = 10 * 3 if self.config.include_z_coordinates else 10 * 2
        pose_dim = 33 * 3 if (self.config.enable_pose and self.config.include_z_coordinates) else 0
        
        return hand_dim + face_dim + pose_dim
    
    def close(self) -> None:
        """Release MediaPipe resources."""
        if self.hands_detector:
            self.hands_detector.close()
        if self.face_detector:
            self.face_detector.close()
        if self.pose_detector:
            self.pose_detector.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
