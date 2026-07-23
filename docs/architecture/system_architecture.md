# SilentVoice System Architecture

## Overview
SilentVoice is a real-time Indian Sign Language (ISL) recognition system that translates sign language into spoken and written language.

## Architecture Principles
- **Separation of Concerns**: Clear boundaries between frontend, backend, ML training, and inference
- **Training-Inference Decoupling**: Training exclusively on Google Colab, inference runs locally
- **Modularity**: Each module is independently testable and replaceable
- **Scalability**: Architecture supports model updates without touching inference code
- **Production-Ready**: Environment-based configuration, proper error handling, logging

## System Components

### Frontend (React + TypeScript)
- Real-time camera capture and video streaming
- WebSocket connection for real-time inference results
- Display transcribed text and play TTS audio
- User settings and configuration management

### Backend API (FastAPI)
- RESTful API endpoints for inference requests
- WebSocket server for real-time communication
- Request validation and error handling
- Service orchestration (inference → translation → TTS)

### ML Pipeline (Training - Colab Only)
- Data preprocessing and augmentation
- Model architecture definition and training
- Hyperparameter tuning and cross-validation
- Model evaluation and metrics tracking
- Model export and versioning

### Inference Engine (Local)
- Frame extraction from video stream
- MediaPipe landmark extraction
- Sequence building and normalization
- Gesture classification using loaded PyTorch model
- Confidence scoring and filtering

### Translation Module
- ISL gesture-to-text mapping
- Grammar processing for natural language output
- Context-aware sentence construction
- Text enhancement and correction

### Text-to-Speech Module
- Speech synthesis from transcribed text
- Voice profile management
- Audio streaming to frontend
- Playback controls

## Data Flow

```
User Camera (Frontend)
    ↓ [Video Frames via WebSocket]
Backend API
    ↓ [Frame Processing]
Inference Engine
    ↓ [Landmarks → Sequences → Model Prediction]
Gesture Classifier
    ↓ [Gesture Labels]
Translation Module
    ↓ [Natural Language Text]
TTS Module
    ↓ [Audio Stream]
Frontend (Display & Audio Output)
```

## Technology Stack
- **Frontend**: React + TypeScript + Vite + Tailwind CSS + shadcn/ui
- **Backend**: FastAPI + Pydantic + WebSockets
- **ML Training**: PyTorch + Google Colab + MediaPipe
- **Inference**: PyTorch + MediaPipe + OpenCV
- **Translation**: Custom ISL dictionary + NLP processing
- **TTS**: pyttsx3 or gTTS (offline/online options)
- **Version Control**: Git + GitHub
