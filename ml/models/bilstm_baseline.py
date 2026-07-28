"""
Bidirectional LSTM Baseline Model for ISL Recognition.

PyTorch implementation of a BiLSTM architecture for sequence classification
of 2D/3D landmark feature vectors with support for variable-length sequences.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class BiLSTMConfig:
    """Hyperparameters and configuration for BiLSTM Baseline Model."""
    input_dim: int = 279          # Default MediaPipe Holistic 279 feature dimensions
    hidden_dim: int = 128         # LSTM hidden state size per direction
    num_layers: int = 2           # Number of stacked LSTM layers
    num_classes: int = 25         # Number of gesture classes
    dropout: float = 0.3          # Dropout probability
    bidirectional: bool = True    # Bidirectional LSTM
    pooling_type: str = 'mean'    # Temporal pooling: 'mean', 'max', 'last', 'attention'
    projection_dim: Optional[int] = 128  # Optional linear input projection before LSTM

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'BiLSTMConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class AttentionPooling(nn.Module):
    """Attention-based temporal pooling over LSTM output sequences."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, D)
            mask: Optional boolean mask of shape (B, T) where True indicates valid frame

        Returns:
            Pooled tensor of shape (B, D)
        """
        scores = self.attn(x).squeeze(-1)  # Shape: (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # Shape: (B, T, 1)
        return torch.sum(x * weights, dim=1)


class BiLSTMBaseline(nn.Module):
    """
    Bidirectional LSTM Baseline Classification Model.
    """

    def __init__(self, config: Optional[BiLSTMConfig] = None):
        """
        Initialize BiLSTM Baseline Model.

        Args:
            config: BiLSTMConfig hyperparameters object
        """
        super().__init__()
        self.config = config or BiLSTMConfig()
        
        # 1. Input Projection Layer (optional reduction / embedding)
        if self.config.projection_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.config.input_dim, self.config.projection_dim),
                nn.LayerNorm(self.config.projection_dim),
                nn.ReLU(),
                nn.Dropout(self.config.dropout)
            )
            lstm_input_dim = self.config.projection_dim
        else:
            self.projection = nn.Identity()
            lstm_input_dim = self.config.input_dim

        # 2. Bidirectional LSTM Backbone
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
            bidirectional=self.config.bidirectional
        )

        lstm_out_dim = self.config.hidden_dim * (2 if self.config.bidirectional else 1)

        # 3. Temporal Pooling Module
        if self.config.pooling_type == 'attention':
            self.pooling = AttentionPooling(lstm_out_dim)
        else:
            self.pooling = None

        # 4. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.BatchNorm1d(lstm_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(lstm_out_dim // 2, self.config.num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of BiLSTM Baseline Model.

        Args:
            x: Input tensor of shape (B, T, D) where B=batch size, T=sequence length, D=features
            lengths: Optional 1D tensor of shape (B,) containing valid frame lengths per sequence

        Returns:
            Logits tensor of shape (B, num_classes)
        """
        B, T, D = x.shape

        # 1. Input Projection
        x_proj = self.projection(x)  # Shape: (B, T, lstm_input_dim)

        # 2. Variable-length sequence handling with PackPaddedSequence
        if lengths is not None:
            # Move lengths to CPU for pack_padded_sequence if needed
            lengths_cpu = lengths.cpu().to(torch.int64)
            # Ensure lengths are at least 1
            lengths_cpu = torch.clamp(lengths_cpu, min=1, max=T)

            packed_x = pack_padded_sequence(
                x_proj,
                lengths_cpu,
                batch_first=True,
                enforce_sorted=False
            )
            packed_out, (hn, cn) = self.lstm(packed_x)
            lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)
        else:
            lstm_out, (hn, cn) = self.lstm(x_proj)

        # 3. Temporal Pooling over time dimension T
        if self.config.pooling_type == 'attention':
            mask = None
            if lengths is not None:
                mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            pooled = self.pooling(lstm_out, mask=mask)
        elif self.config.pooling_type == 'max':
            if lengths is not None:
                mask = (torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)
                lstm_out_masked = lstm_out.masked_fill(~mask, float('-inf'))
                pooled, _ = torch.max(lstm_out_masked, dim=1)
            else:
                pooled, _ = torch.max(lstm_out, dim=1)
        elif self.config.pooling_type == 'last':
            if lengths is not None:
                idx = (lengths - 1).view(-1, 1, 1).expand(B, 1, lstm_out.size(-1)).to(x.device)
                pooled = lstm_out.gather(1, idx).squeeze(1)
            else:
                pooled = lstm_out[:, -1, :]
        else:  # 'mean' default pooling
            if lengths is not None:
                mask = (torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)
                lstm_out_masked = lstm_out * mask.float()
                pooled = torch.sum(lstm_out_masked, dim=1) / torch.clamp(lengths.unsqueeze(1).float(), min=1.0)
            else:
                pooled = torch.mean(lstm_out, dim=1)

        # 4. Classifier Head
        logits = self.classifier(pooled)
        return logits
