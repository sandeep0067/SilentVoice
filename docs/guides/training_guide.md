# Model Training Guide

## Overview
Model training is performed on Google Colab to leverage GPU resources. Trained models are exported and used for local inference.

## Training Process

### 1. Data Preparation
- Place raw ISL video data in `ml/datasets/raw/`
- Run data preprocessing notebook: `ml/training/02_data_preprocessing.ipynb`
- Processed data will be in `ml/datasets/processed/`

### 2. Model Training
- Open `ml/training/04_model_training.ipynb` in Google Colab
- Configure experiment parameters in `ml/experiments/configs/`
- Run training cells
- Monitor training progress with TensorBoard

### 3. Model Evaluation
- Run `ml/training/06_model_evaluation.ipynb`
- Review metrics and confusion matrix
- Adjust hyperparameters if needed

### 4. Model Export
- Use `ml/training/07_model_export.ipynb` to export trained model
- Model will be saved to `ml/models/v{version}/`
- Update `ml/models/latest` symlink

### 5. Model Sync to Local
- Use `utilities/scripts/model_sync.py` to download models from Colab
- Or manually download from Google Drive

## Experiment Tracking
- Experiment configs are stored in `ml/experiments/configs/`
- Training runs are logged in `ml/experiments/runs/`
- Results comparison in `ml/experiments/results/`

## Model Versioning
- Each model version gets its own directory: `ml/models/v1.0.0/`
- Includes: `model.pth`, `config.json`, `metadata.json`, `metrics.json`
- Backend configuration points to specific model version
