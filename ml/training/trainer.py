"""
Baseline Model Trainer for ISL Sequence Recognition.

Handles training and validation loops with clean separation from model definition,
supporting variable-length sequences, early stopping, checkpointing, and mixed precision.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

from ml.training.early_stopping import EarlyStopping
from ml.training.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class BaselineTrainer:
    """
    Manages model training, evaluation, checkpointing, and early stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        scheduler: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        checkpoint_dir: str = 'ml/models/checkpoints',
        log_dir: str = 'ml/models/logs',
        early_stopping_patience: int = 10,
        use_amp: bool = True,
        resume_from: Optional[str] = None
    ):
        """
        Initialize BaselineTrainer.

        Args:
            model: PyTorch classification model (e.g. BiLSTMBaseline)
            optimizer: PyTorch optimizer (e.g. AdamW)
            criterion: Loss function (defaults to CrossEntropyLoss)
            scheduler: Optional learning rate scheduler
            device: Training device ('cuda' or 'cpu')
            checkpoint_dir: Directory path to save model checkpoints
            log_dir: Directory path for TensorBoard logs
            early_stopping_patience: Early stopping patience epochs
            use_amp: Enable Automatic Mixed Precision (AMP) when CUDA is available
            resume_from: Path to checkpoint to resume training from
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.scheduler = scheduler

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        logger.info(f"Initialized trainer on device: {self.device}")

        # Checkpoint manager & Early stopping
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.early_stopping = EarlyStopping(patience=early_stopping_patience, mode='min')

        # Mixed Precision Setup
        self.use_amp = use_amp and (self.device.type == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        if self.use_amp:
            logger.info("Automatic Mixed Precision (AMP) enabled for CUDA training.")

        # TensorBoard Setup
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir)
            logger.info(f"TensorBoard logs will be saved to {self.log_dir}")
        else:
            self.writer = None
            logger.warning("TensorBoard not available. Logging to TensorBoard disabled.")

        # Resume training
        self.start_epoch = 1
        if resume_from:
            self._resume_training(resume_from)

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train model for one epoch.

        Args:
            train_loader: PyTorch DataLoader for training split

        Returns:
            Tuple of (epoch_loss, epoch_accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            # Unpack batch (support tuple of (x, y) or (x, y, lengths))
            if len(batch) == 3:
                x, y, lengths = batch
                x, y, lengths = x.to(self.device), y.to(self.device), lengths.to(self.device)
            else:
                x, y = batch
                lengths = None
                x, y = x.to(self.device), y.to(self.device)

            if y.ndim > 1:
                y = y.squeeze(-1)

            self.optimizer.zero_grad()

            # Forward pass with AMP autocast
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    logits = self.model(x, lengths=lengths)
                    loss = self.criterion(logits, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(x, lengths=lengths)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

            # Track metrics
            batch_size = x.size(0)
            running_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == y).item()
            total_samples += batch_size

        epoch_loss = running_loss / max(1, total_samples)
        epoch_acc = (correct_predictions / max(1, total_samples)) * 100.0
        return epoch_loss, epoch_acc

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate model on validation split.

        Args:
            val_loader: PyTorch DataLoader for validation split

        Returns:
            Tuple of (val_loss, val_accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    x, y, lengths = batch
                    x, y, lengths = x.to(self.device), y.to(self.device), lengths.to(self.device)
                else:
                    x, y = batch
                    lengths = None
                    x, y = x.to(self.device), y.to(self.device)

                if y.ndim > 1:
                    y = y.squeeze(-1)

                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = self.model(x, lengths=lengths)
                        loss = self.criterion(logits, y)
                else:
                    logits = self.model(x, lengths=lengths)
                    loss = self.criterion(logits, y)

                batch_size = x.size(0)
                running_loss += loss.item() * batch_size
                preds = torch.argmax(logits, dim=1)
                correct_predictions += torch.sum(preds == y).item()
                total_samples += batch_size

        val_loss = running_loss / max(1, total_samples)
        val_acc = (correct_predictions / max(1, total_samples)) * 100.0
        return val_loss, val_acc

    def _resume_training(self, checkpoint_path: str) -> None:
        """
        Resume training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"Resuming training from {checkpoint_path}")
        metadata = self.checkpoint_manager.load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler,
            device=self.device
        )
        self.start_epoch = metadata.get('epoch', 0) + 1
        logger.info(f"Resuming from epoch {self.start_epoch}")

        if self.early_stopping.best_score is not None:
            self.early_stopping.best_score = metadata.get('metrics', {}).get('val_loss', float('inf'))
            logger.info(f"Restored early stopping best score: {self.early_stopping.best_score}")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50
    ) -> Dict[str, List[float]]:
        """
        Full training & evaluation execution loop across multiple epochs.

        Args:
            train_loader: Training split DataLoader
            val_loader: Validation split DataLoader
            epochs: Max number of epochs to train

        Returns:
            History dictionary with lists of metrics per epoch
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        best_val_loss = float('inf')
        global_step = 0

        for epoch in range(self.start_epoch, epochs + 1):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            elapsed = time.time() - start_time

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
                self.writer.flush()

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metrics={'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc},
                is_best=is_best
            )

            logger.info(
                f"Epoch [{epoch:03d}/{epochs:03d}] ({elapsed:.1f}s) - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% "
                f"{'[BEST]' if is_best else ''}"
            )

            # Check Early Stopping
            if self.early_stopping.step(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}. Stopping training.")
                break

        if self.writer is not None:
            self.writer.close()
            logger.info(f"Training completed. TensorBoard logs saved to {self.log_dir}")
        else:
            logger.info("Training completed.")

        return history
