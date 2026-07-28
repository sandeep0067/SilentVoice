"""
MediaPipe Holistic Feature Extractor for ISL Recognition.

Extracts comprehensive multi-modal features including hands, face, and pose landmarks
for Indian Sign Language (ISL) recognition. Uses MediaPipe Holistic for unified detection.

Recommendation for ISL Recognition:
- Hand landmarks (Left & Right): ESSENTIAL (126 features) - Primary source of gesture & finger information.
- Face landmarks (Key Facial Subset): HIGHLY RECOMMENDED (120 features for 40 keypoints) - Facial expressions (Non-Manual Features / NMFs) express emotion, questions, and disambiguate homophenous signs.
- Upper Body Pose landmarks: RECOMMENDED (33 features for 11 keypoints) - Upper body pose provides spatial context relative to shoulders/head.

Total Default Feature Dimension: 279 dimensions per frame (float32).
"""

import cv2
import logging
import numpy as np
import mediapipe as mp
try:
    import mediapipe.python.solutions as mp_solutions
except ImportError:
    mp_solutions = mp.solutions
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class LandmarkType(Enum):
    """Types of landmarks that can be extracted."""
    HANDS = "hands"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"
    FACE = "face"
    POSE = "pose"


@dataclass
class HolisticExtractionConfig:
    """Configuration for holistic feature extraction."""
    # Model complexity (0=Lite, 1=Full, 2=Heavy)
    model_complexity: int = 1
    
    # Detection confidence thresholds
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    
    # Feature selection flags
    enable_hands: bool = True
    enable_face: bool = True
    enable_pose: bool = True
    
    # Output configuration
    include_z_coordinates: bool = True
    normalize_coordinates: bool = True
    use_relative_coordinates: bool = True  # Shift wrist/reference point to origin for position invariance
    scale_normalization: bool = True      # Scale landmarks by reference distance (e.g. shoulder width or palm size)
    
    # Face landmark selection
    use_all_face_landmarks: bool = False  # If False, use key facial subset (40 landmarks)
    face_landmark_subset: Optional[List[int]] = None  # Custom subset of indices
    
    # Pose landmark selection
    use_upper_body_only: bool = True  # Only use upper body pose landmarks (indices 0-10)
    
    # Missing detection handling
    fill_missing_detections: bool = True
    missing_fill_value: float = 0.0
    interpolate_missing: bool = True
    
    # Confidence filtering
    use_confidence_filtering: bool = True
    min_landmark_confidence: float = 0.3
    
    @classmethod
    def get_default_isl_config(cls) -> 'HolisticExtractionConfig':
        """
        Get default configuration optimized for ISL recognition.
        
        Returns:
            Configuration with hands, key face landmarks, and upper body pose enabled.
        """
        return cls(
            model_complexity=1,
            enable_hands=True,
            enable_face=True,
            enable_pose=True,
            include_z_coordinates=True,
            normalize_coordinates=True,
            use_relative_coordinates=True,
            scale_normalization=True,
            use_all_face_landmarks=False,
            face_landmark_subset=None,  # Uses DEFAULT_FACE_INDICES (40 keypoints)
            use_upper_body_only=True,
            fill_missing_detections=True,
            interpolate_missing=True
        )
    
    @classmethod
    def get_hands_only_config(cls) -> 'HolisticExtractionConfig':
        """
        Get configuration for hands-only extraction (126 dims).
        
        Returns:
            Configuration with only left and right hand landmarks enabled.
        """
        return cls(
            model_complexity=1,
            enable_hands=True,
            enable_face=False,
            enable_pose=False,
            include_z_coordinates=True,
            normalize_coordinates=True,
            use_relative_coordinates=True
        )
    
    @classmethod
    def get_comprehensive_config(cls) -> 'HolisticExtractionConfig':
        """
        Get comprehensive configuration with all features enabled (Full face mesh + full pose).
        
        Returns:
            Configuration with all 468 face landmarks and 33 pose landmarks.
        """
        return cls(
            model_complexity=1,
            enable_hands=True,
            enable_face=True,
            enable_pose=True,
            include_z_coordinates=True,
            normalize_coordinates=True,
            use_relative_coordinates=False,
            use_all_face_landmarks=True,
            use_upper_body_only=False
        )


@dataclass
class FeatureVector:
    """Container for extracted feature vector with metadata."""
    landmarks: np.ndarray  # Shape: (N, 2) or (N, 3)
    landmark_type: LandmarkType
    confidence: float
    timestamp: float = 0.0
    is_valid: bool = True
    missing_landmarks: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'landmarks': self.landmarks.tolist(),
            'landmark_type': self.landmark_type.value,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'is_valid': self.is_valid,
            'missing_landmarks': self.missing_landmarks,
        }


@dataclass
class HolisticFeatures:
    """Container for all holistic features extracted from a single frame."""
    left_hand_features: Optional[FeatureVector] = None
    right_hand_features: Optional[FeatureVector] = None
    face_features: Optional[FeatureVector] = None
    pose_features: Optional[FeatureVector] = None
    combined_features: Optional[np.ndarray] = None  # Flattened 1D numpy array float32
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'left_hand_features': self.left_hand_features.to_dict() if self.left_hand_features else None,
            'right_hand_features': self.right_hand_features.to_dict() if self.right_hand_features else None,
            'face_features': self.face_features.to_dict() if self.face_features else None,
            'pose_features': self.pose_features.to_dict() if self.pose_features else None,
            'combined_features': self.combined_features.tolist() if self.combined_features is not None else None,
            'timestamp': self.timestamp,
        }
    
    def get_feature_vector(self) -> Optional[np.ndarray]:
        """Get combined 1D numpy feature vector."""
        return self.combined_features
    
    def is_valid(self) -> bool:
        """Check if any feature is valid."""
        if self.combined_features is None:
            return False
        return np.any(self.combined_features != 0)


class HolisticFeatureExtractor:
    """
    MediaPipe Holistic-based feature extractor for ISL recognition.
    
    Extracts hand, face, and pose landmarks using MediaPipe Holistic for unified
    and efficient multi-modal feature extraction.
    """
    
    # Selected key facial landmark indices (40 points) for lips, eyes, eyebrows, nose outline
    DEFAULT_FACE_INDICES = [
        # Lips contour (12 points)
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
        # Eyes contour (12 points)
        33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
        # Eyebrows (10 points)
        70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
        # Nose bridge & tip (6 points)
        1, 2, 98, 327, 168, 6
    ]
    
    # Upper body pose landmark indices (11 points: Head to Torso)
    UPPER_BODY_INDICES = list(range(11))  # Nose, eyes, ears, mouth, shoulders
    
    def __init__(self, config: Optional[HolisticExtractionConfig] = None):
        """
        Initialize holistic feature extractor.
        
        Args:
            config: Extraction configuration
        """
        self.config = config or HolisticExtractionConfig.get_default_isl_config()
        
        # Initialize MediaPipe Holistic
        self.mp_holistic = mp_solutions.holistic
        self.mp_drawing = mp_solutions.drawing_utils
        self.mp_drawing_styles = mp_solutions.drawing_styles
        
        # Initialize detector
        self.holistic_detector = None
        self._initialize_detector()
        
        # Set face landmark subset
        if self.config.face_landmark_subset is None:
            self.config.face_landmark_subset = self.DEFAULT_FACE_INDICES
    
    def _initialize_detector(self) -> None:
        """Initialize MediaPipe Holistic detector instance."""
        self.holistic_detector = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=self.config.model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence
        )
        
        logger.info(f"Initialized Holistic detector with model_complexity={self.config.model_complexity}")
    
    def extract_features(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0
    ) -> HolisticFeatures:
        """
        Extract holistic features from a single image frame (BGR format).
        
        Args:
            frame: Input frame (BGR uint8 format)
            timestamp: Optional frame timestamp in seconds
            
        Returns:
            HolisticFeatures object containing all extracted features
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None")
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Holistic
        results = self.holistic_detector.process(rgb_frame)
        
        # Extract features from Holistic output attributes
        left_hand_features = self._extract_single_hand_features(
            results.left_hand_landmarks, LandmarkType.LEFT_HAND, timestamp
        ) if self.config.enable_hands else None
        
        right_hand_features = self._extract_single_hand_features(
            results.right_hand_landmarks, LandmarkType.RIGHT_HAND, timestamp
        ) if self.config.enable_hands else None
        
        face_features = self._extract_face_features(
            results.face_landmarks, timestamp
        ) if self.config.enable_face else None
        
        pose_features = self._extract_pose_features(
            results.pose_landmarks, timestamp
        ) if self.config.enable_pose else None
        
        # Combine all modalities into a single 1D numpy vector
        combined = self._combine_features(
            left_hand_features, right_hand_features, face_features, pose_features
        )
        
        return HolisticFeatures(
            left_hand_features=left_hand_features,
            right_hand_features=right_hand_features,
            face_features=face_features,
            pose_features=pose_features,
            combined_features=combined,
            timestamp=timestamp
        )
    
    def _extract_single_hand_features(
        self,
        hand_landmarks_proto,
        hand_type: LandmarkType,
        timestamp: float
    ) -> FeatureVector:
        """
        Extract features for a single hand (left or right).
        
        Args:
            hand_landmarks_proto: MediaPipe NormalizedLandmarkList or None
            hand_type: LandmarkType.LEFT_HAND or RIGHT_HAND
            timestamp: Frame timestamp
            
        Returns:
            FeatureVector for the specified hand
        """
        coord_dim = 3 if self.config.include_z_coordinates else 2
        
        if hand_landmarks_proto is None:
            # Handle missing hand detection gracefully
            return FeatureVector(
                landmarks=np.full((21, coord_dim), self.config.missing_fill_value, dtype=np.float32),
                landmark_type=hand_type,
                confidence=0.0,
                timestamp=timestamp,
                is_valid=False,
                missing_landmarks=list(range(21))
            )
        
        landmarks = []
        missing = []
        confidences = []
        
        for idx, lm in enumerate(hand_landmarks_proto.landmark):
            coords = [lm.x, lm.y]
            if self.config.include_z_coordinates:
                coords.append(lm.z)
            
            # Check landmark visibility/confidence if present
            visibility = getattr(lm, 'visibility', 1.0)
            if self.config.use_confidence_filtering and visibility < self.config.min_landmark_confidence:
                if self.config.fill_missing_detections:
                    coords = [self.config.missing_fill_value] * coord_dim
                    missing.append(idx)
                else:
                    confidences.append(visibility)
                    landmarks.append(coords)
                    continue
            
            landmarks.append(coords)
            confidences.append(visibility)
        
        landmarks_array = np.array(landmarks, dtype=np.float32)
        
        # Apply relative coordinate normalization (make relative to wrist landmark at index 0)
        if self.config.use_relative_coordinates and len(landmarks_array) > 0 and 0 not in missing:
            wrist = landmarks_array[0].copy()
            landmarks_array = landmarks_array - wrist
            
            # Scale normalization (relative to hand size: wrist to middle finger MCP at index 9)
            if self.config.scale_normalization and len(landmarks_array) > 9:
                hand_size = np.linalg.norm(landmarks_array[9] - landmarks_array[0])
                if hand_size > 1e-6:
                    landmarks_array = landmarks_array / hand_size
        
        avg_confidence = float(np.mean(confidences)) if confidences else 1.0
        
        return FeatureVector(
            landmarks=landmarks_array,
            landmark_type=hand_type,
            confidence=avg_confidence,
            timestamp=timestamp,
            is_valid=len(missing) < 21,
            missing_landmarks=missing
        )
    
    def _extract_face_features(
        self,
        face_landmarks_proto,
        timestamp: float
    ) -> FeatureVector:
        """
        Extract features for face mesh (full 468 or 40-point key facial landmark subset).
        
        Args:
            face_landmarks_proto: MediaPipe face landmarks or None
            timestamp: Frame timestamp
            
        Returns:
            FeatureVector for facial landmarks
        """
        coord_dim = 3 if self.config.include_z_coordinates else 2
        indices = range(468) if self.config.use_all_face_landmarks else self.config.face_landmark_subset
        num_landmarks = len(indices)
        
        if face_landmarks_proto is None:
            return FeatureVector(
                landmarks=np.full((num_landmarks, coord_dim), self.config.missing_fill_value, dtype=np.float32),
                landmark_type=LandmarkType.FACE,
                confidence=0.0,
                timestamp=timestamp,
                is_valid=False,
                missing_landmarks=list(range(num_landmarks))
            )
        
        landmarks = []
        missing = []
        confidences = []
        
        for i, idx in enumerate(indices):
            if idx < len(face_landmarks_proto.landmark):
                lm = face_landmarks_proto.landmark[idx]
                coords = [lm.x, lm.y]
                if self.config.include_z_coordinates:
                    coords.append(lm.z)
                visibility = getattr(lm, 'visibility', 1.0)
                landmarks.append(coords)
                confidences.append(visibility)
            else:
                landmarks.append([self.config.missing_fill_value] * coord_dim)
                missing.append(i)
        
        landmarks_array = np.array(landmarks, dtype=np.float32)
        
        # Relative normalization to nose tip (index 1 in face mesh) if available
        if self.config.use_relative_coordinates and len(landmarks_array) > 0:
            ref_idx = 0 if self.config.use_all_face_landmarks else (
                self.config.face_landmark_subset.index(1) if 1 in self.config.face_landmark_subset else 0
            )
            reference_point = landmarks_array[ref_idx].copy()
            landmarks_array = landmarks_array - reference_point
        
        avg_confidence = float(np.mean(confidences)) if confidences else 1.0
        
        return FeatureVector(
            landmarks=landmarks_array,
            landmark_type=LandmarkType.FACE,
            confidence=avg_confidence,
            timestamp=timestamp,
            is_valid=len(missing) < num_landmarks,
            missing_landmarks=missing
        )
    
    def _extract_pose_features(
        self,
        pose_landmarks_proto,
        timestamp: float
    ) -> FeatureVector:
        """
        Extract pose features (upper body or full 33 landmarks).
        
        Args:
            pose_landmarks_proto: MediaPipe pose landmarks or None
            timestamp: Frame timestamp
            
        Returns:
            FeatureVector for pose landmarks
        """
        coord_dim = 3 if self.config.include_z_coordinates else 2
        indices = self.UPPER_BODY_INDICES if self.config.use_upper_body_only else list(range(33))
        num_landmarks = len(indices)
        
        if pose_landmarks_proto is None:
            return FeatureVector(
                landmarks=np.full((num_landmarks, coord_dim), self.config.missing_fill_value, dtype=np.float32),
                landmark_type=LandmarkType.POSE,
                confidence=0.0,
                timestamp=timestamp,
                is_valid=False,
                missing_landmarks=list(range(num_landmarks))
            )
        
        landmarks = []
        missing = []
        confidences = []
        
        for i, idx in enumerate(indices):
            if idx < len(pose_landmarks_proto.landmark):
                lm = pose_landmarks_proto.landmark[idx]
                coords = [lm.x, lm.y]
                if self.config.include_z_coordinates:
                    coords.append(lm.z)
                visibility = getattr(lm, 'visibility', 1.0)
                landmarks.append(coords)
                confidences.append(visibility)
            else:
                landmarks.append([self.config.missing_fill_value] * coord_dim)
                missing.append(i)
        
        landmarks_array = np.array(landmarks, dtype=np.float32)
        
        # Relative normalization to nose (index 0) if enabled
        if self.config.use_relative_coordinates and len(landmarks_array) > 0:
            reference_point = landmarks_array[0].copy()
            landmarks_array = landmarks_array - reference_point
        
        avg_confidence = float(np.mean(confidences)) if confidences else 1.0
        
        return FeatureVector(
            landmarks=landmarks_array,
            landmark_type=LandmarkType.POSE,
            confidence=avg_confidence,
            timestamp=timestamp,
            is_valid=len(missing) < num_landmarks,
            missing_landmarks=missing
        )
    
    def _combine_features(
        self,
        left_hand: Optional[FeatureVector],
        right_hand: Optional[FeatureVector],
        face: Optional[FeatureVector],
        pose: Optional[FeatureVector]
    ) -> np.ndarray:
        """
        Combine all active landmark feature vectors into a single 1D numpy array.
        
        Order of feature concatenation:
        1. Left Hand landmarks
        2. Right Hand landmarks
        3. Face landmarks
        4. Pose landmarks
        """
        parts = []
        
        if self.config.enable_hands:
            if left_hand is not None:
                parts.append(left_hand.landmarks.flatten())
            if right_hand is not None:
                parts.append(right_hand.landmarks.flatten())
        
        if self.config.enable_face and face is not None:
            parts.append(face.landmarks.flatten())
        
        if self.config.enable_pose and pose is not None:
            parts.append(pose.landmarks.flatten())
        
        if not parts:
            return np.array([], dtype=np.float32)
        
        return np.concatenate(parts, dtype=np.float32)
    
    def extract_features_batch(
        self,
        frames: List[np.ndarray],
        timestamps: Optional[List[float]] = None
    ) -> List[HolisticFeatures]:
        """
        Extract features from a sequence of frames.
        
        Args:
            frames: List of BGR frames
            timestamps: Optional list of timestamps
            
        Returns:
            List of HolisticFeatures for each frame
        """
        if timestamps is None:
            timestamps = [i / 30.0 for i in range(len(frames))]
        
        features_list = []
        for frame, ts in zip(frames, timestamps):
            feat = self.extract_features(frame, ts)
            features_list.append(feat)
            
        if self.config.interpolate_missing:
            features_list = self.interpolate_missing_features(features_list)
            
        return features_list
    
    def interpolate_missing_features(
        self,
        features_sequence: List[HolisticFeatures]
    ) -> List[HolisticFeatures]:
        """
        Interpolate missing features across a sequence of frames linearly.
        
        Args:
            features_sequence: Sequence of HolisticFeatures
            
        Returns:
            Sequence with missing values interpolated
        """
        if len(features_sequence) <= 1:
            return features_sequence
        
        # Extract combined feature matrices
        vectors = [f.get_feature_vector() for f in features_sequence]
        if any(v is None for v in vectors):
            return features_sequence
            
        feat_matrix = np.array(vectors, dtype=np.float32)  # Shape: (T, N_features)
        
        # Identify zero/invalid frames
        invalid_mask = np.all(feat_matrix == 0, axis=1)
        
        if not np.any(invalid_mask) or np.all(invalid_mask):
            return features_sequence
        
        valid_indices = np.where(~invalid_mask)[0]
        
        # Perform linear interpolation column by column
        for col in range(feat_matrix.shape[1]):
            valid_vals = feat_matrix[valid_indices, col]
            feat_matrix[:, col] = np.interp(
                np.arange(len(features_sequence)), valid_indices, valid_vals
            )
        
        # Update combined features in the sequence
        for idx, feat in enumerate(features_sequence):
            feat.combined_features = feat_matrix[idx]
            
        return features_sequence
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        Get the dimensions of each feature subset and total dimension.
        
        Returns:
            Dictionary with feature breakdown and 'total' count.
        """
        dims = {}
        coord_dim = 3 if self.config.include_z_coordinates else 2
        
        if self.config.enable_hands:
            dims['left_hand'] = 21 * coord_dim
            dims['right_hand'] = 21 * coord_dim
            dims['hands_total'] = 42 * coord_dim
        
        if self.config.enable_face:
            face_count = 468 if self.config.use_all_face_landmarks else len(self.config.face_landmark_subset)
            dims['face'] = face_count * coord_dim
        
        if self.config.enable_pose:
            pose_count = len(self.UPPER_BODY_INDICES) if self.config.use_upper_body_only else 33
            dims['pose'] = pose_count * coord_dim
        
        dims['total'] = sum(v for k, v in dims.items() if k not in ['hands_total'])
        return dims
    
    def close(self) -> None:
        """Release MediaPipe resources."""
        if self.holistic_detector:
            self.holistic_detector.close()
            self.holistic_detector = None
            logger.info("Closed Holistic detector")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
