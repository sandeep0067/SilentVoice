"""
Unit tests for MediaPipe Holistic feature extraction and sample saving.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np

from ml.inference.processors.holistic_feature_extractor import (
    HolisticFeatureExtractor,
    HolisticExtractionConfig,
    LandmarkType
)
from ml.inference.processors.sample_saver import SampleSaver, SampleMetadata


class TestHolisticFeatureExtractor(unittest.TestCase):

    def setUp(self):
        # Create synthetic test frames (RGB/BGR image arrays of 480x640x3)
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some color gradients / structures so OpenCV/MediaPipe doesn't fail on empty buffer
        self.test_frame[:, :, 0] = 120
        self.test_frame[:, :, 1] = 150
        self.test_frame[:, :, 2] = 200

    def test_default_config_dimensions(self):
        config = HolisticExtractionConfig.get_default_isl_config()
        extractor = HolisticFeatureExtractor(config)
        
        dims = extractor.get_feature_dimensions()
        
        # Check expected dimension calculations:
        # Left Hand: 21 * 3 = 63
        # Right Hand: 21 * 3 = 63
        # Face: 40 key points * 3 = 120
        # Pose: 11 upper body * 3 = 33
        # Total = 63 + 63 + 120 + 33 = 279
        self.assertEqual(dims['left_hand'], 63)
        self.assertEqual(dims['right_hand'], 63)
        self.assertEqual(dims['face'], 120)
        self.assertEqual(dims['pose'], 33)
        self.assertEqual(dims['total'], 279)
        extractor.close()

    def test_hands_only_config_dimensions(self):
        config = HolisticExtractionConfig.get_hands_only_config()
        extractor = HolisticFeatureExtractor(config)
        
        dims = extractor.get_feature_dimensions()
        self.assertEqual(dims['left_hand'], 63)
        self.assertEqual(dims['right_hand'], 63)
        self.assertEqual(dims['total'], 126)
        extractor.close()

    def test_extract_features_single_frame(self):
        config = HolisticExtractionConfig.get_default_isl_config()
        with HolisticFeatureExtractor(config) as extractor:
            features = extractor.extract_features(self.test_frame, timestamp=0.5)
            
            vec = features.get_feature_vector()
            self.assertIsNotNone(vec)
            self.assertIsInstance(vec, np.ndarray)
            self.assertEqual(vec.dtype, np.float32)
            self.assertEqual(vec.shape[0], 279)

    def test_missing_detection_handling(self):
        # Blank frame has no visible human hands/face/pose
        config = HolisticExtractionConfig.get_default_isl_config()
        with HolisticFeatureExtractor(config) as extractor:
            features = extractor.extract_features(self.test_frame)
            
            # Since no person is present in the blank frame, features should gracefully fallback to fill value 0.0
            self.assertFalse(features.left_hand_features.is_valid)
            self.assertFalse(features.right_hand_features.is_valid)
            self.assertEqual(len(features.left_hand_features.missing_landmarks), 21)
            self.assertEqual(features.get_feature_vector().shape[0], 279)

    def test_batch_extraction_and_interpolation(self):
        frames = [self.test_frame.copy() for _ in range(5)]
        config = HolisticExtractionConfig.get_default_isl_config()
        with HolisticFeatureExtractor(config) as extractor:
            features_seq = extractor.extract_features_batch(frames)
            
            self.assertEqual(len(features_seq), 5)
            for feat in features_seq:
                self.assertEqual(feat.get_feature_vector().shape[0], 279)

    def test_sample_saver_npz(self):
        temp_dir = tempfile.mkdtemp()
        try:
            saver = SampleSaver(temp_dir)
            
            # Create synthetic feature sequence (30 frames, 279 features)
            dummy_sequence = np.random.randn(30, 279).astype(np.float32)
            
            metadata = SampleMetadata(
                sample_id="test_sample_001",
                label="HELLO",
                num_frames=30,
                feature_dim=279,
                subject_id="SUB_01"
            )
            
            saved_path = saver.save_sample_npz(
                features=dummy_sequence,
                label="HELLO",
                sample_id="test_sample_001",
                split="train",
                metadata=metadata
            )
            
            self.assertTrue(os.path.exists(saved_path))
            
            # Load back and verify round-trip persistence
            loaded_feats, loaded_meta = saver.load_sample_npz(saved_path)
            
            self.assertEqual(loaded_feats.shape, (30, 279))
            self.assertEqual(loaded_feats.dtype, np.float32)
            self.assertEqual(loaded_meta['label'], "HELLO")
            self.assertEqual(loaded_meta['sample_id'], "test_sample_001")
            np.testing.assert_allclose(loaded_feats, dummy_sequence, rtol=1e-5)
            
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
