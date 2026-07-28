"""
Checkpoint Manager Utility for PyTorch Models.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages saving and loading of model checkpoints, optimizer state, and training metadata.
    """

    def __init__(self, checkpoint_dir: Union[str, Path]):
        """
        Initialize CheckpointManager.

        Args:
            checkpoint_dir: Directory where checkpoint files will be stored
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        filename: Optional[str] = None
    ) -> Path:
        """
        Save checkpoint file containing model state_dict and optimizer states.

        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch number
            metrics: Dictionary of metric values (e.g. val_loss, val_acc)
            is_best: Whether this checkpoint achieves the best validation metric
            filename: Optional custom checkpoint filename

        Returns:
            Path to saved checkpoint file
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:03d}.pt"

        save_path = self.checkpoint_dir / filename
        last_path = self.checkpoint_dir / "last_model.pt"

        # Unwrap DataParallel or DDP wrapper if present
        raw_model = model.module if hasattr(model, 'module') else model

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics or {},
            'model_config': getattr(raw_model, 'config', None).to_dict() if hasattr(getattr(raw_model, 'config', None), 'to_dict') else None
        }

        torch.save(checkpoint, save_path)
        torch.save(checkpoint, last_path)
        logger.info(f"Saved checkpoint to {save_path}")

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model checkpoint to {best_path}")

        return save_path

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Union[str, torch.device] = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load weights and state from a checkpoint file into model and optimizer.

        Args:
            checkpoint_path: Path to .pt checkpoint file
            model: PyTorch model to restore weights into
            optimizer: Optional optimizer to restore state into
            scheduler: Optional scheduler to restore state into
            device: Map storage location device ('cpu' or 'cuda')

        Returns:
            Dictionary containing checkpoint metadata (epoch, metrics, config)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        raw_model = model.module if hasattr(model, 'module') else model
        raw_model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Successfully loaded checkpoint from {checkpoint_path} (epoch {checkpoint.get('epoch', 0)})")
        return checkpoint
