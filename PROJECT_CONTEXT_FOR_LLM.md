# SilentVoice Project - Context for AI Assistant

## Project Overview

**Project Name**: SilentVoice - Indian Sign Language (ISL) Recognition System

**Objective**: Build a complete ISL recognition system with:
- ML pipeline for training ISL recognition models using the INCLUDE dataset
- FastAPI backend for model serving
- React frontend with webcam integration for real-time recognition
- Text-to-speech and translation capabilities

**Current Status**: Architecture complete, ready for model training on Google Colab

## Technology Stack

### ML Pipeline
- **Framework**: PyTorch
- **Model**: BiLSTM with attention pooling
- **Feature Extraction**: MediaPipe (hands, face, pose landmarks)
- **Dataset**: INCLUDE dataset for Indian Sign Language
- **Training**: Google Colab (GPU)

### Backend
- **Framework**: FastAPI
- **Features**: REST API, WebSocket, TTS integration, analytics
- **Authentication**: API key based
- **Rate Limiting**: Token bucket algorithm

### Frontend
- **Framework**: React with Vite
- **Styling**: TailwindCSS
- **Features**: Webcam integration, demo/live modes, prediction display, translation output

## Project Structure

```
silentvoice/
├── api/                    # PRIMARY - FastAPI backend (14 files)
│   ├── app.py              # FastAPI application entry point
│   ├── routes.py           # Complete API routes
│   ├── services.py         # Inference service layer
│   ├── config.py           # Configuration management
│   ├── schemas.py          # Pydantic schemas
│   ├── middleware.py       # Custom middleware
│   ├── auth.py             # API key authentication
│   ├── rate_limit.py       # Rate limiting
│   ├── websocket.py        # WebSocket handling
│   ├── analytics.py        # Prediction history tracking
│   └── tts.py              # Text-to-speech integration
├── ml/                     # PRIMARY - ML pipeline (59 Python files)
│   ├── datasets/           # Dataset preprocessing and loading
│   │   ├── preprocessing_pipeline.py  # Full preprocessing orchestrator
│   │   ├── preprocess_include.py       # INCLUDE dataset preprocessing
│   │   ├── dataloader.py               # PyTorch DataLoader
│   │   ├── sequence_generator.py       # Sequence generation with splits
│   │   ├── metadata_manager.py        # Video metadata management
│   │   ├── dataset_validator.py        # Dataset validation
│   │   ├── video_preprocessor.py       # Video preprocessing
│   │   ├── frame_extractor.py         # Frame extraction
│   │   └── quality_filter.py           # Quality filtering
│   ├── inference/          # Inference pipeline
│   │   ├── processors/
│   │   │   ├── landmark_extractor.py  # MediaPipe landmark extraction
│   │   │   ├── sequence_builder.py    # Sequence building
│   │   │   └── data_augmentation.py    # Data augmentation
│   │   ├── realtime/                  # Real-time inference
│   │   └── utils/                     # Inference utilities
│   ├── models/             # Model definitions
│   │   ├── bilstm_baseline.py         # BiLSTM model
│   │   ├── optimized_model.py         # Model optimization
│   │   └── checkpoints/               # Model checkpoints
│   ├── training/           # Training pipeline
│   │   ├── trainer.py                 # Training with checkpointing
│   │   ├── early_stopping.py          # Early stopping
│   │   └── checkpoint.py              # Checkpoint management
│   ├── evaluation/         # Model evaluation
│   │   ├── metrics.py                # Evaluation metrics
│   │   └── visualization.py           # Visualization tools
│   ├── translation/        # Translation services
│   │   ├── translator.py              # Translation service
│   │   ├── label_mapping.py           # Label mapping
│   │   └── prediction.py              # Prediction handling
│   ├── tts/                # Text-to-speech
│   │   ├── manager.py                 # TTS management
│   │   ├── piper_engine.py            # Piper TTS engine
│   │   └── fallback_engine.py         # Fallback TTS
│   ├── utils/              # ML utilities
│   │   ├── file_helpers.py            # File operations
│   │   ├── seed.py                    # Random seed
│   │   ├── landmark_helpers.py        # Landmark utilities
│   │   └── augmentation_helpers.py    # Augmentation utilities
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   ├── infer_realtime.py   # Real-time inference
│   ├── profile_inference.py # Profiling script
│   ├── optimize.py         # Hyperparameter optimization
│   ├── colab_training.ipynb # Google Colab training notebook
│   └── requirements.txt    # ML dependencies
├── web/                    # PRIMARY - React frontend (6,324 files)
│   ├── src/
│   │   ├── components/
│   │   │   ├── WebcamPreview.jsx      # Webcam with demo/live modes
│   │   │   ├── PredictionDisplay.jsx  # Prediction display
│   │   │   ├── TranslationOutput.jsx   # Multi-language translation
│   │   │   ├── PredictionHistory.jsx   # Prediction history
│   │   │   └── Header.jsx             # Header component
│   │   ├── services/
│   │   │   └── api.js                 # API integration
│   │   ├── App.jsx                    # Main application
│   │   ├── main.jsx                   # Entry point
│   │   └── index.css                  # Global styles
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
├── backend/                # DUPLICATE - Remove (incomplete alternative to api/)
├── frontend/               # DUPLICATE - Remove (incomplete alternative to web/)
├── translation/            # DUPLICATE - Remove (duplicate of ml/translation/)
├── tts/                    # DUPLICATE - Remove (duplicate of ml/tts/)
├── deployment/             # Deployment configurations
├── docs/                   # Documentation
├── utilities/              # Shared utilities
├── docker-compose.yml
├── .env.example
└── README.md
```

## Completed Work

### ML Pipeline (100% Complete)
- ✅ Complete preprocessing pipeline with MediaPipe landmark extraction
- ✅ Dataset loading with PyTorch DataLoader
- ✅ Sequence generation with train/val/test splits
- ✅ Data augmentation (spatial and temporal)
- ✅ BiLSTM baseline model with configurable architecture
- ✅ Training pipeline with checkpointing, early stopping, AMP
- ✅ Evaluation pipeline with metrics and visualization
- ✅ Translation service for multi-language output
- ✅ TTS integration (Piper and fallback engines)
- ✅ Hyperparameter optimization with Optuna
- ✅ Real-time inference pipeline
- ✅ INCLUDE dataset preprocessing script
- ✅ Google Colab training notebook

### Backend API (100% Complete)
- ✅ FastAPI application with middleware
- ✅ Complete REST API endpoints:
  - Health check
  - Model info
  - Single prediction
  - Batch prediction
  - Translation
  - File upload
  - Metrics
  - TTS
  - WebSocket for real-time predictions
  - Analytics (history, statistics, class filtering)
- ✅ Inference service layer
- ✅ API key authentication
- ✅ Rate limiting
- ✅ Error handling middleware
- ✅ Request ID tracking
- ✅ Prediction history analytics

### Frontend UI (80% Complete)
- ✅ React application with Vite
- ✅ Webcam integration with FPS monitoring
- ✅ Demo mode with mock predictions
- ✅ Live mode with API integration
- ✅ Prediction display with confidence visualization
- ✅ Translation output with multi-language support
- ✅ Prediction history
- ✅ API service layer
- ✅ TailwindCSS styling
- ⚠️ **Missing**: MediaPipe integration in browser for live mode

### Documentation (70% Complete)
- ✅ INCLUDE dataset guide
- ✅ Training guide
- ✅ Output locations guide
- ✅ Optimization guide
- ✅ Profiling summary
- ⚠️ **Missing**: API documentation, deployment guide, troubleshooting

## Current Issues

### Technical Debt
1. **Duplicate directories** (should be removed):
   - `backend/` - duplicate of `api/`
   - `frontend/` - duplicate of `web/`
   - `translation/` - duplicate of `ml/translation/`
   - `tts/` - duplicate of `ml/tts/`

2. **Frontend limitation**:
   - Live mode uses placeholder feature extraction (random features)
   - Needs MediaPipe JavaScript integration for real webcam inference

3. **Documentation gaps**:
   - No API usage examples
   - Missing deployment guide
   - No troubleshooting guide

## Training Workflow

### Google Colab Training
- **Notebook**: `ml/colab_training.ipynb`
- **Dataset**: INCLUDE dataset (upload to Google Drive)
- **Output**: Checkpoints saved to Google Drive
- **Status**: Ready to use

### Training Steps
1. Upload INCLUDE dataset to `My Drive/SilentVoice/datasets/INCLUDE/`
2. Upload required ML modules to Colab (listed in notebook)
3. Run preprocessing to extract landmarks
4. Train BiLSTM model with default hyperparameters
5. Evaluate on test set
6. Download best_model.pt to local project
7. Place in `ml/models/checkpoints/best_model.pt`

### Model Configuration
- **Architecture**: BiLSTM with attention pooling
- **Input**: 279-dimensional MediaPipe landmarks (hands + face)
- **Sequence length**: 30 frames
- **Hidden dim**: 128
- **Layers**: 2 bidirectional LSTM layers
- **Dropout**: 0.3
- **Pooling**: Mean pooling
- **Loss**: Cross-entropy
- **Optimizer**: Adam
- **Scheduler**: Cosine annealing
- **AMP**: Mixed precision training

## Next Steps (Priority Order)

### 1. Clean Up Duplicate Directories (HIGH PRIORITY)
Remove the following directories to eliminate confusion:
- `backend/` (incomplete, duplicate of `api/`)
- `frontend/` (incomplete, duplicate of `web/`)
- `translation/` (duplicate of `ml/translation/`)
- `tts/` (duplicate of `ml/tts/`)

### 2. Complete Google Colab Training (HIGH PRIORITY)
- Upload INCLUDE dataset to Google Drive
- Run the Colab notebook to train the model
- Download best_model.pt
- Place in `ml/models/checkpoints/best_model.pt`

### 3. Integrate MediaPipe in Frontend (MEDIUM PRIORITY)
- Add MediaPipe JavaScript library to web/
- Replace placeholder feature extraction with real MediaPipe
- Test live mode with webcam
- Add error handling for MediaPipe initialization

## Important Notes

### DO NOT
- Do not generate new features or files
- Do not modify frontend or backend architecture
- Do not change the ML pipeline structure
- Do not implement training locally (only on Colab)

### DO
- Focus on cleaning up duplicate directories
- Complete the Colab training workflow
- Integrate MediaPipe in frontend for live mode
- Add missing documentation
- Fix any bugs in existing code

### File Locations
- **Model checkpoints**: `ml/models/checkpoints/best_model.pt`
- **Training logs**: `ml/models/logs/`
- **Preprocessed data**: `ml/datasets/processed/`
- **API base URL**: `http://localhost:8000`
- **Frontend dev server**: `http://localhost:3000`

### Dependencies
- **ML**: See `ml/requirements.txt`
- **Frontend**: See `web/package.json`
- **Backend**: See `api/` imports

## Current Working State

- **ML Pipeline**: Complete and ready for training
- **Backend API**: Complete and production-ready
- **Frontend**: Demo mode works, live mode needs MediaPipe
- **Training**: Colab notebook ready, waiting for dataset upload
- **Model**: No trained model yet (best_model.pt does not exist)

## Communication Style

- Be direct and concise
- Focus on implementation over explanation
- Use existing code patterns
- Follow the established architecture
- Prioritize minimal, focused changes
- Test changes before committing
