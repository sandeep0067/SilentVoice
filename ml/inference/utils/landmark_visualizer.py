"""
Landmark Visualizer for debugging and analysis.

Visualizes landmarks on frames for debugging and quality assessment.
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class VisualizationConfig:
    """Configuration for landmark visualization."""
    draw_hands: bool = True
    draw_face: bool = True
    draw_pose: bool = False
    draw_connections: bool = True
    landmark_color: Tuple[int, int, int] = (0, 255, 0)
    connection_color: Tuple[int, int, int] = (255, 0, 0)
    landmark_thickness: int = 2
    connection_thickness: int = 2
    show_confidence: bool = False


class LandmarkVisualizer:
    """Visualizes landmarks on frames."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize landmark visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def visualize_landmarks(
        self,
        frame: np.ndarray,
        hand_landmarks: Optional[List] = None,
        face_landmarks: Optional[List] = None,
        pose_landmarks: Optional[List] = None
    ) -> np.ndarray:
        """
        Visualize landmarks on frame.
        
        Args:
            frame: Input frame
            hand_landmarks: Hand landmarks from MediaPipe
            face_landmarks: Face landmarks from MediaPipe
            pose_landmarks: Pose landmarks from MediaPipe
            
        Returns:
            Frame with landmarks drawn
        """
        vis_frame = frame.copy()
        
        if self.config.draw_hands and hand_landmarks:
            self._draw_hand_landmarks(vis_frame, hand_landmarks)
        
        if self.config.draw_face and face_landmarks:
            self._draw_face_landmarks(vis_frame, face_landmarks)
        
        if self.config.draw_pose and pose_landmarks:
            self._draw_pose_landmarks(vis_frame, pose_landmarks)
        
        return vis_frame
    
    def _draw_hand_landmarks(
        self,
        frame: np.ndarray,
        hand_landmarks: List
    ) -> None:
        """Draw hand landmarks on frame."""
        mp_hands = mp.solutions.hands
        
        for hand_lm in hand_landmarks:
            if self.config.draw_connections:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=self.config.landmark_color,
                        thickness=self.config.landmark_thickness
                    ),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=self.config.connection_color,
                        thickness=self.config.connection_thickness
                    )
                )
            else:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_lm,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=self.config.landmark_color,
                        thickness=self.config.landmark_thickness
                    )
                )
    
    def _draw_face_landmarks(
        self,
        frame: np.ndarray,
        face_landmarks: List
    ) -> None:
        """Draw face landmarks on frame."""
        mp_face_mesh = mp.solutions.face_mesh
        
        for face_lm in face_landmarks:
            if self.config.draw_connections:
                self.mp_drawing.draw_landmarks(
                    frame,
                    face_lm,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_mesh_face_mesh_tesselation_style()
                )
            else:
                self.mp_drawing.draw_landmarks(
                    frame,
                    face_lm,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=self.config.landmark_color,
                        thickness=self.config.landmark_thickness,
                        circle_radius=1
                    )
                )
    
    def _draw_pose_landmarks(
        self,
        frame: np.ndarray,
        pose_landmarks: List
    ) -> None:
        """Draw pose landmarks on frame."""
        mp_pose = mp.solutions.pose
        
        for pose_lm in pose_landmarks:
            if self.config.draw_connections:
                self.mp_drawing.draw_landmarks(
                    frame,
                    pose_lm,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=self.config.landmark_color,
                        thickness=self.config.landmark_thickness
                    ),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=self.config.connection_color,
                        thickness=self.config.connection_thickness
                    )
                )
            else:
                self.mp_drawing.draw_landmarks(
                    frame,
                    pose_lm,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=self.config.landmark_color,
                        thickness=self.config.landmark_thickness
                    )
                )
    
    def create_landmark_overlay(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        landmark_type: str = 'hands'
    ) -> np.ndarray:
        """
        Create overlay from landmark array.
        
        Args:
            frame: Input frame
            landmarks: Landmark array
            landmark_type: Type of landmarks ('hands', 'face', 'pose')
            
        Returns:
            Frame with landmark overlay
        """
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        if landmark_type == 'hands':
            # Draw hand landmarks from array
            # Assumes landmarks are in format: [hand1_landmarks, hand2_landmarks]
            for hand_idx in range(2):
                start_idx = hand_idx * 21 * 3
                hand_lm = landmarks[start_idx:start_idx + 21 * 3].reshape(21, 3)
                
                for lm in hand_lm:
                    x = int(lm[0] * width)
                    y = int(lm[1] * height)
                    cv2.circle(
                        overlay,
                        (x, y),
                        3,
                        self.config.landmark_color,
                        -1
                    )
        
        elif landmark_type == 'face':
            # Draw facial landmarks from array
            # Assumes 10 key facial landmarks
            for lm in landmarks.reshape(-1, 3):
                x = int(lm[0] * width)
                y = int(lm[1] * height)
                cv2.circle(
                    overlay,
                    (x, y),
                    2,
                    self.config.landmark_color,
                    -1
                )
        
        return overlay
    
    def save_visualization(
        self,
        frame: np.ndarray,
        output_path: str
    ) -> None:
        """
        Save visualization to file.
        
        Args:
            frame: Frame to save
            output_path: Output file path
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, frame)
    
    def create_comparison(
        self,
        original_frame: np.ndarray,
        visualized_frame: np.ndarray,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Create side-by-side comparison of original and visualized frames.
        
        Args:
            original_frame: Original frame
            visualized_frame: Frame with landmarks
            output_path: Optional path to save comparison
            
        Returns:
            Comparison image
        """
        h, w = original_frame.shape[:2]
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        
        comparison[:, :w] = original_frame
        comparison[:, w:] = visualized_frame
        
        # Add labels
        cv2.putText(
            comparison,
            "Original",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        cv2.putText(
            comparison,
            "Landmarks",
            (w + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        if output_path:
            self.save_visualization(comparison, output_path)
        
        return comparison
