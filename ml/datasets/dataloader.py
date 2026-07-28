"""
PyTorch DataLoader for ISL dataset.

Loads preprocessed sequences for model training.
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple
import json


class ISLDataset(Dataset):
    """Indian Sign Language Dataset for training."""
    
    def __init__(
        self,
        sequences_path: str,
        labels_path: Optional[str] = None,
        transform=None
    ):
        """
        Initialize ISL dataset.
        
        Args:
            sequences_path: Path to sequences .npy file
            labels_path: Path to labels .json file
            transform: Optional transform to apply
        """
        self.sequences_path = Path(sequences_path)
        self.transform = transform
        
        # Load sequences
        self.sequences = np.load(self.sequences_path)
        
        # Load labels
        if labels_path:
            with open(labels_path, 'r') as f:
                self.labels = json.load(f)
        else:
            # Try to find labels.json in same directory
            labels_path = self.sequences_path.parent / 'labels.json'
            if labels_path.exists():
                with open(labels_path, 'r') as f:
                    self.labels = json.load(f)
            else:
                self.labels = ['unknown'] * len(self.sequences)
        
        # Create label to index mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Convert labels to indices
        self.label_indices = [self.label_to_idx[label] for label in self.labels]
    
    def __len__(self):
        """Return dataset size."""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get item from dataset.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (sequence, label_index)
        """
        sequence = self.sequences[idx]
        label_idx = self.label_indices[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
        
        return torch.FloatTensor(sequence), torch.LongTensor([label_idx])
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        return len(self.label_to_idx)
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset."""
        from collections import Counter
        
        label_counts = Counter(self.labels)
        total_samples = len(self.labels)
        
        weights = []
        for label in sorted(self.label_to_idx.keys()):
            weight = total_samples / (len(self.label_to_idx) * label_counts[label])
            weights.append(weight)
        
        return torch.FloatTensor(weights)


def get_dataloader(
    sequences_path: str,
    labels_path: Optional[str] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a DataLoader for the ISL dataset.
    
    Args:
        sequences_path: Path to sequences .npy file
        labels_path: Path to labels .json file
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        DataLoader instance
    """
    dataset = ISLDataset(sequences_path, labels_path)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
