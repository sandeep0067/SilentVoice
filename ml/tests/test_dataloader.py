"""
Unit tests for ISL PyTorch DataLoader.
"""

import json
import shutil
import tempfile
import unittest
import numpy as np
import torch

from ml.datasets.dataloader import ISLDataset, get_dataloader


class TestISLDataLoader(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.seq_path = f"{self.temp_dir}/sequences.npy"
        self.lbl_path = f"{self.temp_dir}/labels.json"

        # Generate synthetic sequences (N=10, T=30, D=279)
        self.sequences = np.random.randn(10, 30, 279).astype(np.float32)
        self.labels = ["HELLO", "HELLO", "HELLO", "NAMASTE", "NAMASTE", "WATER", "WATER", "WATER", "HELP", "HELP"]

        np.save(self.seq_path, self.sequences)
        with open(self.lbl_path, "w") as f:
            json.dump(self.labels, f)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_dataset_loading(self):
        dataset = ISLDataset(self.seq_path, self.lbl_path)

        self.assertEqual(len(dataset), 10)
        self.assertEqual(dataset.get_num_classes(), 4)

        x, y = dataset[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertEqual(x.shape, (30, 279))
        self.assertIsInstance(y, torch.Tensor)

    def test_dataloader_batching(self):
        loader = get_dataloader(self.seq_path, self.lbl_path, batch_size=4, shuffle=False, num_workers=0)
        
        batch_x, batch_y = next(iter(loader))
        self.assertEqual(batch_x.shape, (4, 30, 279))
        self.assertEqual(batch_y.shape, (4, 1))


if __name__ == '__main__':
    unittest.main()
