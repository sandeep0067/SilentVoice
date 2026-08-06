"""
ASL Alphabet Dataset Preprocessing Script

This script preprocesses the Kaggle ASL Alphabet dataset for alphabet classification.
It handles image scanning, MediaPipe Hands landmark extraction, and train/val/test splitting.
"""

import argparse
import logging
import sys
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import cv2
import mediapipe as mp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ASLAlphabetPreprocessor:
    """
    Preprocessor for ASL Alphabet dataset.
    
    Extracts MediaPipe hand landmarks from images and creates train/val/test splits.
    """
    
    def __init__(
        self,
        dataset_root: str,
        output_root: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5
    ):
        """
        Initialize the preprocessor.
        
        Args:
            dataset_root: Path to ASL Alphabet dataset root
            output_root: Path to save preprocessed data
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
        """
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'processed_images': 0,
            'skipped_images': 0,
            'classes': {}
        }
        
        logger.info(f"Initialized ASL Alphabet preprocessor")
        logger.info(f"Dataset root: {self.dataset_root}")
        logger.info(f"Output root: {self.output_root}")
    
    def scan_dataset(self) -> Dict[str, List[Path]]:
        """
        Scan the dataset directory structure.
        
        Returns:
            Dictionary mapping class names to lists of image paths
        """
        logger.info("Scanning dataset structure...")
        
        class_images = defaultdict(list)
        
        # Expected structure: dataset_root/class_name/*.jpg
        for class_dir in self.dataset_root.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            logger.info(f"Found class: {class_name}")
            
            # Find all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            images = []
            for ext in image_extensions:
                images.extend(class_dir.glob(f'*{ext}'))
                images.extend(class_dir.glob(f'*{ext.upper()}'))
            
            if images:
                class_images[class_name] = sorted(images)
                self.stats['classes'][class_name] = len(images)
                logger.info(f"  {class_name}: {len(images)} images")
        
        self.stats['total_images'] = sum(len(imgs) for imgs in class_images.values())
        logger.info(f"Total images found: {self.stats['total_images']}")
        
        return dict(class_images)
    
    def extract_landmarks(self, image_path: Path) -> np.ndarray:
        """
        Extract hand landmarks from a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Landmark array of shape (63,) or None if no hand detected
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Failed to read image: {image_path}")
            return None
        
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
    
    def create_splits(
        self,
        class_images: Dict[str, List[Path]]
    ) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]], Dict[str, List[Path]]]:
        """
        Create stratified train/val/test splits.
        
        Args:
            class_images: Dictionary mapping class names to image paths
            
        Returns:
            Tuple of (train_images, val_images, test_images) dictionaries
        """
        train_images = defaultdict(list)
        val_images = defaultdict(list)
        test_images = defaultdict(list)
        
        for class_name, images in class_images.items():
            n_images = len(images)
            n_train = int(n_images * self.train_ratio)
            n_val = int(n_images * self.val_ratio)
            n_test = n_images - n_train - n_val
            
            # Shuffle images
            np.random.shuffle(images)
            
            # Split
            train_images[class_name] = images[:n_train]
            val_images[class_name] = images[n_train:n_train + n_val]
            test_images[class_name] = images[n_train + n_val:]
            
            logger.info(f"{class_name}: train={n_train}, val={n_val}, test={n_test}")
        
        return dict(train_images), dict(val_images), dict(test_images)
    
    def process_split(
        self,
        split_images: Dict[str, List[Path]],
        split_name: str
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Process a single split (train/val/test).
        
        Args:
            split_images: Dictionary mapping class names to image paths
            split_name: Name of the split ('train', 'val', 'test')
            
        Returns:
            Tuple of (landmarks_array, labels_list, image_paths_list)
        """
        logger.info(f"Processing {split_name} split...")
        
        all_landmarks = []
        all_labels = []
        all_paths = []
        
        for class_name, images in split_images.items():
            for image_path in images:
                landmarks = self.extract_landmarks(image_path)
                
                if landmarks is not None:
                    all_landmarks.append(landmarks)
                    all_labels.append(class_name)
                    all_paths.append(str(image_path))
                    self.stats['processed_images'] += 1
                else:
                    self.stats['skipped_images'] += 1
                    logger.debug(f"Skipped {image_path}: no hand detected")
        
        if all_landmarks:
            landmarks_array = np.stack(all_landmarks, axis=0)
            logger.info(f"Processed {len(all_landmarks)} images for {split_name}")
        else:
            landmarks_array = np.array([])
            logger.warning(f"No valid images processed for {split_name}")
        
        return landmarks_array, all_labels, all_paths
    
    def save_data(
        self,
        landmarks: np.ndarray,
        labels: List[str],
        paths: List[str],
        split_name: str
    ):
        """
        Save preprocessed data to disk.
        
        Args:
            landmarks: Landmark array of shape (N, 63)
            labels: List of class labels
            paths: List of image paths
            split_name: Name of the split
        """
        split_dir = self.output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Save landmarks
        landmarks_path = split_dir / 'landmarks.npy'
        np.save(landmarks_path, landmarks)
        
        # Save labels and metadata
        metadata = {
            'labels': labels,
            'image_paths': paths,
            'num_samples': len(labels),
            'feature_dim': 63,
            'num_classes': len(set(labels))
        }
        
        metadata_path = split_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {split_name} data to {split_dir}")
    
    def run(self):
        """
        Run the complete preprocessing pipeline.
        """
        logger.info("Starting ASL Alphabet preprocessing pipeline...")
        
        # Create output directory
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Scan dataset
        class_images = self.scan_dataset()
        
        if not class_images:
            logger.error("No classes found in dataset")
            return False
        
        # Create splits
        train_images, val_images, test_images = self.create_splits(class_images)
        
        # Process each split
        train_landmarks, train_labels, train_paths = self.process_split(train_images, 'train')
        val_landmarks, val_labels, val_paths = self.process_split(val_images, 'val')
        test_landmarks, test_labels, test_paths = self.process_split(test_images, 'test')
        
        # Save data
        self.save_data(train_landmarks, train_labels, train_paths, 'train')
        self.save_data(val_landmarks, val_labels, val_paths, 'val')
        self.save_data(test_landmarks, test_labels, test_paths, 'test')
        
        # Save statistics
        stats_path = self.output_root / 'preprocessing_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info("Preprocessing completed successfully!")
        logger.info(f"Total images: {self.stats['total_images']}")
        logger.info(f"Processed: {self.stats['processed_images']}")
        logger.info(f"Skipped: {self.stats['skipped_images']}")
        logger.info(f"Statistics saved to {stats_path}")
        
        return True
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'hands'):
            self.hands.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Preprocess ASL Alphabet dataset for alphabet classification'
    )
    
    parser.add_argument(
        '--dataset-root',
        type=str,
        required=True,
        help='Path to ASL Alphabet dataset root directory'
    )
    
    parser.add_argument(
        '--output-root',
        type=str,
        required=True,
        help='Path to output directory for preprocessed data'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Ratio of training data (default: 0.7)'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Ratio of validation data (default: 0.15)'
    )
    
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Ratio of test data (default: 0.15)'
    )
    
    parser.add_argument(
        '--max-num-hands',
        type=int,
        default=1,
        help='Maximum number of hands to detect (default: 1)'
    )
    
    parser.add_argument(
        '--min-detection-confidence',
        type=float,
        default=0.5,
        help='Minimum confidence for hand detection (default: 0.5)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.01:
        logger.error("Train, val, and test ratios must sum to 1.0")
        sys.exit(1)
    
    # Create preprocessor
    preprocessor = ASLAlphabetPreprocessor(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_num_hands=args.max_num_hands,
        min_detection_confidence=args.min_detection_confidence
    )
    
    # Run preprocessing
    success = preprocessor.run()
    
    if success:
        logger.info("Preprocessing completed successfully!")
        sys.exit(0)
    else:
        logger.error("Preprocessing failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
