"""
Hand Landmark Extraction using MediaPipe Hands.

This module provides functionality to extract hand landmarks from webcam frames
using MediaPipe Hands, matching the exact configuration used during training.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os


class HandLandmarkExtractor:
    """
    Extracts hand landmarks from frames using MediaPipe Hands.
    
    Uses the exact configuration from training:
    - max_num_hands=2 (for two-hand support)
    - static_image_mode=False
    - min_detection_confidence=0.5
    - min_tracking_confidence=0.5
    """
    
    # Configuration constants
    MAX_NUM_HANDS = 2
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    MODEL_FILENAME = "hand_landmarker.task"
    
    def __init__(self):
        """Initialize the MediaPipe Hands detector."""
        try:
            # Try old API first (compatible with older MediaPipe versions)
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                max_num_hands=self.MAX_NUM_HANDS,
                static_image_mode=False,
                min_detection_confidence=self.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=self.MIN_TRACKING_CONFIDENCE
            )
            self.use_new_api = False
            print("Using MediaPipe Solutions API (old)")
        except (AttributeError, Exception) as e:
            # Use new MediaPipe Tasks API (for newer MediaPipe versions)
            print(f"Old API not available ({e}), using new MediaPipe Tasks API")
            self._initialize_new_api()
        
    def _initialize_new_api(self):
        """Initialize the new MediaPipe Tasks API."""
        # Download model if not present
        if not os.path.exists(self.MODEL_FILENAME):
            print("Downloading hand landmarker model...")
            try:
                urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_FILENAME)
                print("Model downloaded successfully")
            except Exception as download_error:
                print(f"Failed to download model: {download_error}")
                raise
        
        try:
            base_options = python.BaseOptions(model_asset_path=self.MODEL_FILENAME)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=self.MAX_NUM_HANDS,
                min_hand_detection_confidence=self.MIN_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=self.MIN_TRACKING_CONFIDENCE
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self.use_new_api = True
            print("MediaPipe Tasks API initialized successfully")
        except Exception as api_error:
            print(f"Failed to initialize new API: {api_error}")
            raise
        
    def extract_landmarks(self, frame: np.ndarray) -> Tuple[Optional[List[np.ndarray]], Optional[object]]:
        """
        Extract hand landmarks from a BGR frame.
        
        Args:
            frame: Input frame in BGR format (as from cv2.VideoCapture)
            
        Returns:
            Tuple of (landmarks_list, results):
            - landmarks_list: List of landmark arrays (one per hand), each of shape (63,)
                            with x,y,z coordinates for 21 landmarks, or None if no hand detected
            - results: Raw MediaPipe results object for visualization/drawing,
                      or None if no hand detected
        """
        if self.use_new_api:
            # Use new MediaPipe Tasks API
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = self.detector.detect(mp_image)
            
            if not results.hand_landmarks:
                return None, None
            
            # Extract landmarks from all detected hands
            landmarks_list = []
            for hand_landmarks in results.hand_landmarks:
                landmarks = []
                for lm in hand_landmarks:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks_array = np.array(landmarks, dtype=np.float32)
                landmarks_list.append(landmarks_array)
            
            return landmarks_list, results
        else:
            # Use old API
            # Convert BGR to RGB (MediaPipe expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe Hands
            results = self.hands.process(frame_rgb)
            
            # Check if any hands were detected
            if not results.multi_hand_landmarks:
                return None, None
            
            # Extract landmarks from all detected hands
            landmarks_list = []
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks_array = np.array(landmarks, dtype=np.float32)
                landmarks_list.append(landmarks_array)
            
            return landmarks_list, results
    
    def draw_landmarks(self, frame: np.ndarray, results: object) -> np.ndarray:
        """
        Draw hand landmarks on the frame for visualization.
        
        Args:
            frame: Input frame in BGR format
            results: MediaPipe results object from extract_landmarks
            
        Returns:
            Frame with landmarks drawn
        """
        if results is None:
            return frame
        
        # Create a copy to avoid modifying the original
        annotated_frame = frame.copy()
        
        if self.use_new_api:
            # New API - results.hand_landmarks is a list of NormalizedLandmarkList
            if not results.hand_landmarks:
                return annotated_frame
            
            # For new API, we need to manually draw since drawing_utils may not be available
            # Simple landmark drawing for new API
            for hand_landmarks in results.hand_landmarks:
                h, w, _ = annotated_frame.shape
                for lm in hand_landmarks:
                    # Convert normalized coordinates to pixel coordinates
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
        else:
            # Old API - results.multi_hand_landmarks exists
            if not results.multi_hand_landmarks:
                return annotated_frame
            
            # Draw landmarks using MediaPipe's drawing utilities
            try:
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles
                
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
            except Exception as e:
                print(f"Drawing utilities not available: {e}")
                # Fallback to simple drawing
                h, w, _ = annotated_frame.shape
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
        
        return annotated_frame
    
    def close(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'hands'):
            self.hands.close()
    
    def __del__(self):
        """Clean up on deletion."""
        self.close()


def extract_landmarks(frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[object]]:
    """
    Convenience function to extract landmarks from a frame.
    
    This creates a temporary HandLandmarkExtractor instance and extracts landmarks.
    For better performance with video streams, create a HandLandmarkExtractor instance
    and call its extract_landmarks method directly.
    
    Args:
        frame: Input frame in BGR format
        
    Returns:
        Tuple of (landmarks, results) as described in HandLandmarkExtractor.extract_landmarks
    """
    extractor = HandLandmarkExtractor()
    landmarks, results = extractor.extract_landmarks(frame)
    extractor.close()
    return landmarks, results


if __name__ == '__main__':
    # Quick test of the landmark extractor
    print("HandLandmarkExtractor module loaded successfully")
    print("Use test_landmarks.py to test with webcam")
