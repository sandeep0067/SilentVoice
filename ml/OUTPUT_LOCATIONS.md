# Model Training Output Locations

## Overview

This document specifies exactly where checkpoints, logs, and trained models are saved during the SilentVoice model training process.

## Directory Structure

```
ml/
├── models/
│   ├── checkpoints/           # Model checkpoints
│   ├── logs/                 # TensorBoard logs
│   └── training.log          # Training text log
├── datasets/
│   ├── raw/                  # Original INCLUDE dataset
│   └── processed/            # Preprocessed training data
│       ├── landmarks/        # Extracted landmarks
│       ├── sequences/        # Training sequences
│       ├── quality_reports/  # Quality validation reports
│       └── pipeline_statistics.json
```

## Checkpoint Locations

### Primary Checkpoint Directory
**Path**: `ml/models/checkpoints/`

### Checkpoint Files

#### Best Model
**File**: `ml/models/checkpoints/best_model.pt`
- **Description**: The model checkpoint with the highest validation accuracy
- **When Saved**: Automatically saved when validation accuracy improves
- **Use Case**: Final model for deployment and inference

#### Last Model
**File**: `ml/models/checkpoints/last_model.pt`
- **Description**: The most recent checkpoint (end of training or interruption)
- **When Saved**: Saved at the end of each epoch
- **Use Case**: Resuming training from interruption

#### Epoch Checkpoints
**Files**: `ml/models/checkpoints/checkpoint_epoch_N.pt`
- **Description**: Checkpoint at specific epoch N
- **When Saved**: Saved at the end of each epoch
- **Use Case**: Training analysis, rollback to specific epoch

### Checkpoint Contents

Each checkpoint file contains:
```python
{
    'epoch': int,                    # Current epoch number
    'model_state_dict': dict,       # Model parameters
    'optimizer_state_dict': dict,    # Optimizer state
    'scheduler_state_dict': dict,    # Learning rate scheduler state
    'train_loss': float,            # Training loss
    'val_loss': float,              # Validation loss
    'val_acc': float,               # Validation accuracy
    'model_config': dict            # Model configuration
}
```

### Loading a Checkpoint

```python
import torch
from ml.models.bilstm_baseline import BiLSTMBaseline, BiLSTMConfig

# Load checkpoint
checkpoint = torch.load('ml/models/checkpoints/best_model.pt')

# Recreate model with saved config
config = BiLSTMConfig.from_dict(checkpoint['model_config'])
model = BiLSTMBaseline(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Access training metrics
print(f"Epoch: {checkpoint['epoch']}")
print(f"Validation Accuracy: {checkpoint['val_acc']:.2f}%")
```

## Log Locations

### TensorBoard Logs
**Directory**: `ml/models/logs/`

**Contents**:
- TensorBoard event files for training visualization
- Metrics tracked per epoch (loss, accuracy, learning rate)
- Model graphs (if enabled)

**Viewing Logs**:
```bash
tensorboard --logdir ml/models/logs
# Open browser to http://localhost:6006
```

### Training Log
**File**: `ml/models/training.log`

**Contents**:
- Detailed training progress
- Epoch-by-epoch metrics
- Configuration parameters
- Error messages and warnings

**Viewing Log**:
```bash
# Real-time monitoring
tail -f ml/models/training.log

# View entire log
cat ml/models/training.log

# Search for errors
grep ERROR ml/models/training.log
```

## Preprocessed Data Locations

### Landmarks
**Directory**: `ml/datasets/processed/landmarks/`

**Files**: `*.npy` files (one per video)
- **Format**: NumPy arrays of shape (num_frames, feature_dim)
- **Feature Dim**: 279 (hands + face + pose landmarks)
- **Use Case**: Intermediate data for sequence generation

### Sequences
**Directory**: `ml/datasets/processed/sequences/`

**Subdirectories**:
- `train/` - Training sequences
- `val/` - Validation sequences  
- `test/` - Test sequences

**Files in each split**:
- `sequences.npy` - NumPy array of training sequences
- `labels.json` - JSON file with label mappings

**Sequence Format**:
- Shape: (num_sequences, sequence_length, feature_dim)
- Sequence Length: 30 frames (default)
- Feature Dim: 279 landmarks

### Quality Reports
**Directory**: `ml/datasets/processed/quality_reports/`

**Files**:
- `validation_report.json` - Dataset validation results
- Quality metrics per video
- Rejection reasons for filtered videos

### Pipeline Statistics
**File**: `ml/datasets/processed/pipeline_statistics.json`

**Contents**:
```json
{
    "total_videos": 100,
    "processed_videos": 95,
    "failed_videos": 3,
    "skipped_videos": 2,
    "quality_filtered": 5,
    "total_sequences": 1500,
    "train_sequences": 1050,
    "val_sequences": 225,
    "test_sequences": 225
}
```

## Configuration Files

### Training Configuration
**File**: `ml/experiments/configs/training_config.yaml`

**Contents**:
- Model hyperparameters
- Training parameters
- Data augmentation settings
- Checkpoint and logging settings

### Preprocessing Configuration
**File**: `ml/experiments/configs/preprocessing_config.yaml`

**Contents**:
- Video preprocessing settings
- Landmark extraction settings
- Sequence generation parameters
- Data split ratios

## Summary of Key Locations

| Item | Location | Purpose |
|------|----------|---------|
| **Best Model** | `ml/models/checkpoints/best_model.pt` | Final trained model for deployment |
| **Last Checkpoint** | `ml/models/checkpoints/last_model.pt` | Resume training from interruption |
| **Epoch Checkpoints** | `ml/models/checkpoints/checkpoint_epoch_N.pt` | Training analysis and rollback |
| **TensorBoard Logs** | `ml/models/logs/` | Training visualization |
| **Training Log** | `ml/models/training.log` | Detailed training progress |
| **Training Sequences** | `ml/datasets/processed/sequences/train/` | Training data |
| **Validation Sequences** | `ml/datasets/processed/sequences/val/` | Validation data |
| **Test Sequences** | `ml/datasets/processed/sequences/test/` | Test data |
| **Pipeline Statistics** | `ml/datasets/processed/pipeline_statistics.json` | Preprocessing summary |

## Accessing Outputs Programmatically

```python
from pathlib import Path

# Define paths
CHECKPOINT_DIR = Path('ml/models/checkpoints')
LOG_DIR = Path('ml/models/logs')
DATA_DIR = Path('ml/datasets/processed')

# Get best model
best_model_path = CHECKPOINT_DIR / 'best_model.pt'

# Get latest checkpoint
checkpoints = list(CHECKPOINT_DIR.glob('checkpoint_epoch_*.pt'))
latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)

# Get training sequences
train_sequences = DATA_DIR / 'sequences' / 'train' / 'sequences.npy'
train_labels = DATA_DIR / 'sequences' / 'train' / 'labels.json'

# Get pipeline statistics
stats_path = DATA_DIR / 'pipeline_statistics.json'
```

## Cleanup and Management

### Removing Old Checkpoints
```bash
# Keep only best and last model
cd ml/models/checkpoints
rm checkpoint_epoch_*.pt
```

### Clearing TensorBoard Logs
```bash
rm -rf ml/models/logs/*
```

### Removing Preprocessed Data
```bash
# Remove all preprocessed data
rm -rf ml/datasets/processed/*
```

## Backup Recommendations

### Essential Files to Backup
1. `ml/models/checkpoints/best_model.pt` - Best trained model
2. `ml/models/training.log` - Training history
3. `ml/datasets/processed/pipeline_statistics.json` - Data statistics
4. `ml/datasets/processed/sequences/` - Preprocessed training data

### Backup Command
```bash
# Create backup of essential files
tar -czf silentvoice_backup.tar.gz \
    ml/models/checkpoints/best_model.pt \
    ml/models/training.log \
    ml/datasets/processed/pipeline_statistics.json \
    ml/datasets/processed/sequences/
```

## Verification Checklist

After training, verify:
- [ ] `best_model.pt` exists in checkpoints directory
- [ ] `training.log` contains complete training history
- [ ] TensorBoard logs are viewable
- [ ] Preprocessed sequences are accessible
- [ ] Model can be loaded successfully
- [ ] Model performs as expected on validation set
