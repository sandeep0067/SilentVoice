# SilentVoice

A real-time Indian Sign Language (ISL) recognition system that translates sign language into spoken and written language.

## Project Overview

SilentVoice is an AI/ML-first project designed for national-level hackathons and portfolio development. It uses computer vision and deep learning to recognize ISL gestures in real-time and convert them to natural language text and speech.

## Tech Stack

- **Frontend**: React + TypeScript + Vite + Tailwind CSS + shadcn/ui
- **Backend**: FastAPI + Pydantic + WebSockets
- **ML Training**: PyTorch + Google Colab + MediaPipe
- **Inference**: PyTorch + MediaPipe + OpenCV
- **Translation**: Custom ISL dictionary + NLP processing
- **TTS**: pyttsx3/gTTS (offline/online options)
- **Version Control**: Git + GitHub

## Architecture

```
silentvoice/
├── frontend/           # React application
├── backend/            # FastAPI backend
├── ml/                 # ML training and inference
├── translation/        # ISL to text translation
├── tts/               # Text-to-speech synthesis
├── utilities/         # Shared utilities
├── docs/              # Documentation
└── deployment/        # Deployment configurations
```

## Quick Start

### Prerequisites
- Node.js 18+
- Python 3.10+
- Git

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd silentvoice
```

2. **Frontend setup**
```bash
cd frontend
npm install
npm run dev
```

3. **Backend setup**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

4. **ML module setup**
```bash
cd ml
pip install -r requirements.txt
```

## Project Structure

- **Training-inference decoupling**: Training happens on Google Colab, inference runs locally
- **Model versioning**: Trained models are versioned in `ml/models/`
- **Experiment tracking**: ML experiments are tracked in `ml/experiments/`
- **Modular design**: Each module can be developed and tested independently

## Documentation

Comprehensive documentation is available in the `docs/` directory:
- [System Architecture](docs/architecture/system_architecture.md)
- [Setup Guide](docs/guides/setup_guide.md)
- [Training Guide](docs/guides/training_guide.md)
- [Model Architecture](docs/ml/model_architecture.md)

## Development

### Frontend Development
```bash
cd frontend
npm run dev
```

### Backend Development
```bash
cd backend
uvicorn app.main:app --reload
```

### Model Training
Model training is performed on Google Colab. See [Training Guide](docs/guides/training_guide.md) for details.

## Contributing

This is a portfolio project. Contributions and suggestions are welcome.

## License

[Specify your license here]

## Acknowledgments

- MediaPipe for hand tracking
- PyTorch for deep learning
- FastAPI for backend API
- shadcn/ui for UI components
