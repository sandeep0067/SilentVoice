# SilentVoice Model Training Guide

## Overview

This guide provides step-by-step instructions for training the SilentVoice ISL recognition model using the INCLUDE dataset.

## Prerequisites

1. **INCLUDE Dataset**: Downloaded and organized according to INCLUDE_DATASET_GUIDE.md
2. **Preprocessed Data**: Dataset preprocessed using the preprocessing script
3. **Environment**: Python virtual environment with all dependencies installed

## Step 1: Preprocess the INCLUDE Dataset

Before training, preprocess the INCLUDE dataset to extract landmarks and generate training sequences.

```bash
# Create metadata file (if not already created)
python ml/datasets/preprocess_include.py \
    --dataset-root ml/datasets/raw/INCLUDE \
    --output-root ml/datasets/processed \
    --metadata-path ml/datasets/raw/metadata.csv \
    --create-metadata

# Validate dataset structure
python ml/datasets/preprocess_include.py \
    --dataset-root ml/datasets/raw/INCLUDE \
    --output-root ml/datasets/processed \
    --metadata-path ml/datasets/raw/metadata.csv \
    --validate-only

# Run preprocessing
python ml/datasets/preprocess_include.py \
    --dataset-root ml/datasets/raw/INCLUDE \
    --output-root ml/datasets/processed \
    --metadata-path ml/datasets/raw/metadata.csv \
    --sequence-length 30 \
    --stride 15 \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

## Step 2: Verify Preprocessed Data

Check that preprocessing completed successfully:

```bash
# Check landmark files
ls ml/datasets/processed/landmarks/

# Check sequence files
ls ml/datasets/processed/sequences/train/
ls ml/datasets/processed/sequences/val/
ls ml/datasets/processed/sequences/test/

# View statistics
cat ml/datasets/processed/pipeline_statistics.json
```

Expected output structure:
```
ml/datasets/processed/
├── landmarks/
│   ├── 0001.npy
│   ├── 0002.npy
│   └── ...
├── sequences/
│   ├── train/
│   │   ├── sequences.npy
│   │   └── labels.json
│   ├── val/
│   │   ├── sequences.npy
│   │   └── labels.json
│   └── test/
│       ├── sequences.npy
│       └── labels.json
├── quality_reports/
│   └── validation_report.json
└── pipeline_statistics.json
```

## Step 3: Basic Training Command

Start training with default parameters:

```bash
python ml/train.py \
    --data_dir ml/datasets/processed \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-3
```

## Step 4: Advanced Training Commands

### Training with Custom Model Architecture

```bash
python ml/train.py \
    --data_dir ml/datasets/processed \
    --input_dim 279 \
    --hidden_dim 128 \
    --num_layers 2 \
    --num classes 25 \
    --dropout 0.3 \
    --bidirectional \
    --pooling_type mean
```

### Training with Learning Rate Scheduling

```bash
# Cosine annealing (recommended)
python ml/train.py \
    --data_dir ml/datasets/processed \
    --scheduler cosine \
    --scheduler_t_max 50

# Step decay
python ml/train.py \
    --data_dir ml/datasets/processed \
    --scheduler step \
    --scheduler_step_size 10 \
    --scheduler_gamma 0.1

# Reduce on plateau
python ml/train.py \
    --data_dir ml/datasets/processed \
    --scheduler plateau \
    --scheduler_gamma 0.1
```

### Training with Early Stopping

```bash
python ml/train.py \
    --data_dir ml/datasets/processed \
    --patience 10 \
    --epochs 100
```

### Training with Mixed Precision (GPU)

```bash
python ml/train.py \
    --data_dir ml/datasets/processed \
    --use_amp \
    --device cuda
```

### Training with Reproducibility

```bash
python ml/train.py \
    --data_dir ml/datasets/processed \
    --seed 42 \
    --deterministic
```

### Resuming from Checkpoint

```bash
python ml/train.py \
    --data_dir ml/datasets/processed \
    --resume_from ml/models/checkpoints/checkpoint_epoch_15.pt
```

## Step 5: Recommended Training Configuration

For the INCLUDE dataset, use this recommended configuration:

```bash
python ml/train.py \
    --data_dir ml/datasets/processed \
    --train_split train \
    --val_split val \
    --input_dim 279 \
    --hidden_dim 128 \
    --num_layers 2 \
    --num_classes 25 \
    --dropout 0.3 \
    --bidirectional \
    --pooling_type mean \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --patience 10 \
    --seed 42 \
    --scheduler cosine \
    --scheduler_t_max 50 \
    --device cuda \
    --use_amp \
    --num_workers 4 \
    --checkpoint_dir ml/models/checkpoints \
    --log_dir ml/models/logs \
    --log_level INFO
```

## Step 6: Monitor Training

### Using TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir ml/models/logs

# Open browser to http://localhost:6006
```

### Using Training Log

```bash
# View training log in real-time
tail -f ml/models/training.log
```

## Step 7: Training Output Locations

### Checkpoints
**Location**: `ml/models/checkpoints/`

**Files saved**:
- `checkpoint_epoch_N.pt` - Checkpoint at epoch N
- `best_model.pt` - Best model based on validation accuracy
- `last_model.pt` - Last checkpoint (for resuming)

**Checkpoint contents**:
```python
{
    'epoch': N,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_acc': val_acc,
    'model_config': config.to_dict()
}
```

### Logs
**Location**: `ml/models/logs/`

**Files saved**:
- TensorBoard event files (for visualization)
- Training metrics per epoch

**Training log**: `ml/models/training.log`

### Model Artifacts
**Location**: `ml/models/`

**Files saved**:
- `training.log` - Detailed training log
- `checkpoints/` - Model checkpoints
- `logs/` - TensorBoard logs

## Step 8: Loading Trained Model

After training, load the best model for inference:

```python
import torch
from ml.models.bilstm_baseline import BiLSTMBaseline, BiLSTMConfig

# Load checkpoint
checkpoint = torch.load('ml/models/checkpoints/best_model.pt')

# Recreate model
config = BiLSTMConfig.from_dict(checkpoint['model_config'])
model = BiLSTMBaseline(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for inference
```

## Step 9: Evaluating on Test Set

After training, evaluate on the test set:

```bash
python ml/evaluate.py \
    --checkpoint ml/models/checkpoints/best_model.pt \
    --data_dir ml/datasets/processed \
    --test_split test
```

## Training Tips

### Hyperparameter Tuning

**Learning Rate**: Start with 1e-3, try 1e-4 if training is unstable
**Batch Size**: 32 is good default, try 16 or 64 based on GPU memory
**Hidden Dimension**: 128-256 is typical for ISL recognition
**Dropout**: 0.3-0.5 to prevent overfitting
**Sequence Length**: 30 frames (1 second at 30 FPS) is standard

### Common Issues

**Out of Memory**: Reduce batch_size or hidden_dim
**Overfitting**: Increase dropout, add data augmentation
**Underfitting**: Increase model size, train longer
**Unstable Training**: Reduce learning rate, use gradient clipping

### Performance Optimization

**Use GPU**: Set `--device cuda` for 5-10x speedup
**Mixed Precision**: Set `--use_amp` for 2x speedup on GPU
**Data Loading**: Increase `--num_workers` for faster data loading
**Pin Memory**: Automatically enabled for GPU training

## Training Checklist

- [ ] INCLUDE dataset downloaded and organized
- [ ] Metadata file created and validated
- [ ] Dataset preprocessed successfully
- [ ] Preprocessed data verified
- [ ] Training configuration selected
- [ ] Training started with correct parameters
- [ ] TensorBoard monitoring set up
- [ ] Training progress monitored
- [ ] Best model checkpoint saved
- [ ] Model evaluated on test set
- [ ] Model documented and ready for deployment

## Next Steps

After successful training:
1. Evaluate model on test set
2. Analyze confusion matrix for class-wise performance
3. Optimize model using profiling tools
4. Deploy to FastAPI backend
5. Test with real webcam input
