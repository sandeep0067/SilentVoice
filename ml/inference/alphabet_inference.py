"""
Single-frame inference for ASL Alphabet Classifier.

Provides lightweight inference for static alphabet signs using MediaPipe Hands
and the MLP alphabet classifier. No temporal processing needed.
"""

import cv2
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
import mediapipe as mp

from ml.models.mlp_alphabet_classifier import AlphabetMLP, AlphabetMLPConfig


logger = logging.getLogger(__name__)


@dataclass
class AlphabetInferenceResult:
    """Container for alphabet inference results."""
    predicted_class: int
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: float
    inference_time_ms: float
    hand_detected: bool


@dataclass
class AlphabetInferenceConfig:
    """Configuration for alphabet inference."""
    # Model settings
    checkpoint_path: str = 'ml/models/alphabet_checkpoints/best_model.pt'
    config_path: str = 'ml/models/alphabet_checkpoints/model_config.json'
    
    # MediaPipe Hands settings
    max_num_hands: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    
    # Confidence threshold
    confidence_threshold: float = 0.5
    
    # Display settings
    display_landmarks: bool = True
    display_predictions: bool = True
    display_confidence: bool = True
    
    # Camera settings
    camera_id: int = 0
    camera_width: int = 640
    camera_height: int = 480


class AlphabetInferencePipeline:
    """
    Single-frame inference pipeline for ASL alphabet classification.
    
    Extracts hand landmarks from a single frame and predicts the alphabet letter.
    No temporal processing or sliding window needed.
    """

    def __init__(self, config: Optional[AlphabetInferenceConfig] = None):
        """
        Initialize alphabet inference pipeline.

        Args:
            config: AlphabetInferenceConfig configuration
        """
        self.config = config or AlphabetInferenceConfig()
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=self.config.max_num_hands,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence
        )
        
        # Load model
        self.model = None
        self.device = None
        self.label_to_idx = {}
        self.idx_to_label = {}
        self._load_model()
        
        logger.info("Alphabet inference pipeline initialized")

    def _load_model(self):
        """Load the trained alphabet classifier model."""
        checkpoint_path = Path(self.config.checkpoint_path)
        config_path = Path(self.config.config_path)
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model config
        if config_path.exists():
            model_config = AlphabetMLPConfig.load_config(config_path)
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            model_config = AlphabetMLPConfig()
        
        # Create model
        self.model = AlphabetMLP(model_config)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load label mapping if available in checkpoint
        if 'label_mapping' in checkpoint:
            self.idx_to_label = checkpoint['label_mapping']
            self.label_to_idx = {v: k for k, v in self.idx_to_label.items()}
        else:
            # Default A-Z + SPACE + DELETE + NOTHING
            default_labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['SPACE', 'DELETE', 'NOTHING']
            self.idx_to_label = {i: label for i, label in enumerate(default_labels)}
            self.label_to_idx = {label: i for i, label in self.idx_to_label.items()}
        
        logger.info(f"Model loaded from {checkpoint_path}")
        logger.info(f"Number of classes: {len(self.idx_to_label)}")

    def extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from a single image.

        Args:
            image: Input image (BGR format)

        Returns:
            Landmark array of shape (63,) or None if no hand detected
        """
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Hands
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Extract landmarks from the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract 21 landmarks × 3 coordinates (x, y, z)
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        
        return np.array(landmarks, dtype=np.float32)

    def predict(self, landmarks: np.ndarray) -> AlphabetInferenceResult:
        """
        Predict alphabet class from hand landmarks.

        Args:
            landmarks: Landmark array of shape (63,)

        Returns:
            AlphabetInferenceResult with prediction
        """
        start_time = time.time()
        
        # Convert to tensor
        landmarks_tensor = torch.FloatTensor(landmarks).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(landmarks_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Get results
        predicted_class = predicted.item()
        confidence_score = confidence.item()
        predicted_label = self.idx_to_label.get(predicted_class, f"Class_{predicted_class}")
        
        # Create probability dictionary
        probs_dict = {}
        for idx, prob in enumerate(probabilities[0].cpu().numpy()):
            label = self.idx_to_label.get(idx, f"Class_{idx}")
            probs_dict[label] = float(prob)
        
        return AlphabetInferenceResult(
            predicted_class=predicted_class,
            predicted_label=predicted_label,
            confidence=confidence_score,
            probabilities=probs_dict,
            timestamp=time.time(),
            inference_time_ms=inference_time,
            hand_detected=True
        )

    def process_frame(self, image: np.ndarray) -> Tuple[AlphabetInferenceResult, Optional[np.ndarray]]:
        """
        Process a single frame and return prediction.

        Args:
            image: Input image (BGR format)

        Returns:
            Tuple of (inference_result, annotated_image)
        """
        # Extract landmarks
        landmarks = self.extract_landmarks(image)
        
        if landmarks is None:
            # No hand detected
            result = AlphabetInferenceResult(
                predicted_class=-1,
                predicted_label="NO_HAND",
                confidence=0.0,
                probabilities={},
                timestamp=time.time(),
                inference_time_ms=0.0,
                hand_detected=False
            )
            return result, image
        
        # Predict
        result = self.predict(landmarks)
        
        # Annotate image
        annotated = self._annotate_image(image, result)
        
        return result, annotated

    def _annotate_image(self, image: np.ndarray, result: AlphabetInferenceResult) -> np.ndarray:
        """
        Annotate image with prediction results.

        Args:
            image: Input image
            result: Inference result

        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        if not result.hand_detected:
            cv2.putText(
                annotated,
                "No hand detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )
            return annotated
        
        # Display prediction
        if self.config.display_predictions:
            label_text = f"Letter: {result.predicted_label}"
            cv2.putText(
                annotated,
                label_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )
        
        # Display confidence
        if self.config.display_confidence:
            conf_text = f"Confidence: {result.confidence:.2f}"
            cv2.putText(
                annotated,
                conf_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )
        
        # Display inference time
        time_text = f"Time: {result.inference_time_ms:.1f}ms"
        cv2.putText(
            annotated,
            time_text,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        return annotated

    def run_webcam(self):
        """
        Run real-time inference from webcam.
        """
        # Initialize camera
        cap = cv2.VideoCapture(self.config.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        
        logger.info(f"Starting webcam inference (camera {self.config.camera_id})")
        logger.info("Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read from camera")
                    break
                
                # Process frame
                result, annotated = self.process_frame(frame)
                
                # Display
                cv2.imshow('ASL Alphabet Recognition', annotated)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Webcam inference stopped")

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'hands'):
            self.hands.close()


def main():
    """Main function for standalone inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ASL Alphabet single-frame inference')
    parser.add_argument('--checkpoint', type=str, 
                        default='ml/models/alphabet_checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str,
                        default='ml/models/alphabet_checkpoints/model_config.json',
                        help='Path to model config')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera ID')
    parser.add_argument('--width', type=int, default=640,
                        help='Camera width')
    parser.add_argument('--height', type=int, default=480,
                        help='Camera height')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Minimum confidence threshold')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create pipeline
    config = AlphabetInferenceConfig(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        camera_id=args.camera,
        camera_width=args.width,
        camera_height=args.height,
        confidence_threshold=args.confidence
    )
    
    pipeline = AlphabetInferencePipeline(config)
    
    # Run webcam inference
    pipeline.run_webcam()


if __name__ == '__main__':
    main()
