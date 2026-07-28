"""
Early Stopping Utility for PyTorch Model Training.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Monitors a metric and signals when training should stop early.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min'
    ):
        """
        Initialize EarlyStopping.

        Args:
            patience: Number of epochs to wait without improvement before stopping
            min_delta: Minimum change in monitored metric to qualify as an improvement
            mode: 'min' if monitored metric should decrease (e.g. loss), 'max' if it should increase (e.g. accuracy)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()

        if self.mode not in ['min', 'max']:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")

        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def step(self, current_val: float) -> bool:
        """
        Update early stopping tracker with latest validation metric.

        Args:
            current_val: Monitored metric value for current epoch

        Returns:
            True if training should stop early, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_val
            return False

        if self.mode == 'min':
            improved = current_val < (self.best_score - self.min_delta)
        else:
            improved = current_val > (self.best_score + self.min_delta)

        if improved:
            logger.info(f"EarlyStopping: Metric improved from {self.best_score:.4f} to {current_val:.4f}")
            self.best_score = current_val
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience} (best: {self.best_score:.4f}, current: {current_val:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping triggered.")

        return self.early_stop

    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
