"""
Unit tests for BiLSTM Baseline Model architecture.
"""

import os
import shutil
import tempfile
import unittest
import torch

from ml.models.bilstm_baseline import BiLSTMBaseline, BiLSTMConfig
from ml.training.checkpoint import CheckpointManager


class TestBiLSTMBaseline(unittest.TestCase):

    def setUp(self):
        self.config = BiLSTMConfig(
            input_dim=279,
            hidden_dim=64,
            num_layers=2,
            num_classes=10,
            dropout=0.2,
            bidirectional=True,
            pooling_type='mean'
        )
        self.model = BiLSTMBaseline(self.config)

    def test_forward_fixed_length(self):
        batch_size = 4
        seq_len = 30
        x = torch.randn(batch_size, seq_len, self.config.input_dim)
        
        logits = self.model(x)
        self.assertEqual(logits.shape, (batch_size, self.config.num_classes))

    def test_forward_variable_length(self):
        batch_size = 4
        seq_len = 30
        x = torch.randn(batch_size, seq_len, self.config.input_dim)
        lengths = torch.tensor([30, 25, 20, 15], dtype=torch.long)

        logits = self.model(x, lengths=lengths)
        self.assertEqual(logits.shape, (batch_size, self.config.num_classes))

    def test_pooling_types(self):
        for pooling in ['mean', 'max', 'last', 'attention']:
            cfg = BiLSTMConfig(
                input_dim=279,
                hidden_dim=32,
                num_classes=5,
                pooling_type=pooling
            )
            model = BiLSTMBaseline(cfg)
            x = torch.randn(2, 20, 279)
            logits = model(x)
            self.assertEqual(logits.shape, (2, 5))

    def test_checkpoint_save_and_load(self):
        temp_dir = tempfile.mkdtemp()
        try:
            ckpt_mgr = CheckpointManager(temp_dir)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

            saved_path = ckpt_mgr.save_checkpoint(
                model=self.model,
                optimizer=optimizer,
                epoch=5,
                metrics={'val_loss': 0.5},
                is_best=True
            )

            self.assertTrue(os.path.exists(saved_path))
            best_path = os.path.join(temp_dir, "best_model.pt")
            self.assertTrue(os.path.exists(best_path))

            # Load into new model
            new_model = BiLSTMBaseline(self.config)
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
            metadata = ckpt_mgr.load_checkpoint(saved_path, new_model, new_optimizer)

            self.assertEqual(metadata['epoch'], 5)
            self.assertEqual(metadata['metrics']['val_loss'], 0.5)

            # Test predictions match
            x = torch.randn(2, 15, 279)
            self.model.eval()
            new_model.eval()

            with torch.no_grad():
                out1 = self.model(x)
                out2 = new_model(x)
                torch.testing.assert_close(out1, out2)

        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
