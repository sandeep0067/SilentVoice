"""
Unit tests for Landmark Data Augmentation Pipeline.
"""

import unittest
import numpy as np

from ml.inference.processors.data_augmentation import DataAugmenter, AugmentationConfig


class TestDataAugmenter(unittest.TestCase):

    def setUp(self):
        # Create synthetic landmark sequence of shape (T=30, D=279)
        np.random.seed(42)
        self.seq_len = 30
        self.feature_dim = 279
        self.test_sequence = np.random.uniform(0.1, 0.9, size=(self.seq_len, self.feature_dim)).astype(np.float32)
        
        # Add zero-padding to simulate missing hand detection on right hand (index 63..126)
        self.test_sequence[:, 63:126] = 0.0

    def test_disabled_config(self):
        config = AugmentationConfig.get_disabled_config()
        augmenter = DataAugmenter(config)
        aug_seq = augmenter.augment_sequence(self.test_sequence)
        
        np.testing.assert_allclose(aug_seq, self.test_sequence)

    def test_temporal_cropping(self):
        config = AugmentationConfig(
            enabled=True,
            enable_temporal_cropping=True,
            prob_temporal_cropping=1.0,
            enable_time_stretching=False,
            enable_gaussian_noise=False,
            enable_rotation=False,
            enable_scaling=False,
            enable_translation=False,
            enable_landmark_dropout=False
        )
        augmenter = DataAugmenter(config)
        aug_seq = augmenter.apply_temporal_cropping(self.test_sequence)
        
        self.assertEqual(aug_seq.shape, (self.seq_len, self.feature_dim))
        self.assertEqual(aug_seq.dtype, np.float32)

    def test_time_stretching(self):
        config = AugmentationConfig(
            enabled=True,
            enable_time_stretching=True,
            prob_time_stretching=1.0,
            enable_temporal_cropping=False,
            enable_gaussian_noise=False,
            enable_rotation=False,
            enable_scaling=False,
            enable_translation=False,
            enable_landmark_dropout=False
        )
        augmenter = DataAugmenter(config)
        aug_seq = augmenter.apply_time_stretching(self.test_sequence)
        
        self.assertEqual(aug_seq.shape, (self.seq_len, self.feature_dim))
        self.assertEqual(aug_seq.dtype, np.float32)

    def test_gaussian_noise(self):
        config = AugmentationConfig(
            enabled=True,
            enable_gaussian_noise=True,
            prob_gaussian_noise=1.0,
            gaussian_noise_std=0.01,
            enable_temporal_cropping=False,
            enable_time_stretching=False,
            enable_rotation=False,
            enable_scaling=False,
            enable_translation=False,
            enable_landmark_dropout=False
        )
        augmenter = DataAugmenter(config)
        aug_seq = augmenter.apply_gaussian_noise(self.test_sequence)
        
        self.assertEqual(aug_seq.shape, (self.seq_len, self.feature_dim))
        # Ensure zero-padded missing landmarks remained 0.0
        np.testing.assert_allclose(aug_seq[:, 63:126], 0.0)
        # Ensure non-zero landmarks received jitter
        self.assertFalse(np.allclose(aug_seq[:, 0:63], self.test_sequence[:, 0:63]))

    def test_rotation(self):
        config = AugmentationConfig(
            enabled=True,
            enable_rotation=True,
            prob_rotation=1.0,
            rotation_range_deg=10.0
        )
        augmenter = DataAugmenter(config)
        aug_seq = augmenter.apply_rotation(self.test_sequence, coord_dim=3)
        
        self.assertEqual(aug_seq.shape, (self.seq_len, self.feature_dim))
        # Missing landmarks remain 0.0
        np.testing.assert_allclose(aug_seq[:, 63:126], 0.0)

    def test_scaling(self):
        config = AugmentationConfig(
            enabled=True,
            enable_scaling=True,
            prob_scaling=1.0,
            scaling_range=(0.9, 1.1)
        )
        augmenter = DataAugmenter(config)
        aug_seq = augmenter.apply_scaling(self.test_sequence, coord_dim=3)
        
        self.assertEqual(aug_seq.shape, (self.seq_len, self.feature_dim))
        np.testing.assert_allclose(aug_seq[:, 63:126], 0.0)

    def test_translation(self):
        config = AugmentationConfig(
            enabled=True,
            enable_translation=True,
            prob_translation=1.0,
            translation_range=0.05
        )
        augmenter = DataAugmenter(config)
        aug_seq = augmenter.apply_translation(self.test_sequence, coord_dim=3)
        
        self.assertEqual(aug_seq.shape, (self.seq_len, self.feature_dim))
        np.testing.assert_allclose(aug_seq[:, 63:126], 0.0)

    def test_landmark_dropout(self):
        config = AugmentationConfig(
            enabled=True,
            enable_landmark_dropout=True,
            prob_landmark_dropout=1.0,
            dropout_rate=0.1
        )
        augmenter = DataAugmenter(config)
        aug_seq = augmenter.apply_landmark_dropout(self.test_sequence)
        
        self.assertEqual(aug_seq.shape, (self.seq_len, self.feature_dim))
        # Some elements in non-zero region should now be 0.0
        self.assertTrue(np.any(aug_seq[:, 0:63] == 0.0))

    def test_augment_with_original(self):
        config = AugmentationConfig()
        augmenter = DataAugmenter(config)
        
        batch = np.array([self.test_sequence, self.test_sequence], dtype=np.float32)
        labels = ["HELLO", "NAMASTE"]
        
        combined_seqs, combined_labels = augmenter.augment_with_original(batch, labels)
        
        self.assertEqual(combined_seqs.shape, (4, self.seq_len, self.feature_dim))
        self.assertEqual(len(combined_labels), 4)
        self.assertEqual(combined_labels, ["HELLO", "NAMASTE", "HELLO", "NAMASTE"])


if __name__ == '__main__':
    unittest.main()
