# SilentVoice Setup Guide

## Prerequisites
- Node.js 18+ and npm
- Python 3.10+
- Git
- (Optional) Docker for containerized deployment

## Initial Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd silentvoice
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
The frontend will be available at `http://localhost:5173`

### 3. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
The backend API will be available at `http://localhost:8000`

### 4. ML Module Setup
```bash
cd ml
pip install -r requirements.txt
```

### 5. Translation Module Setup
```bash
cd translation
pip install -r requirements.txt
```

### 6. TTS Module Setup
```bash
cd tts
pip install -r requirements.txt
```

### 7. Utilities Setup
```bash
cd utilities
pip install -r requirements.txt
```

## Environment Variables
Copy `.env.example` to `.env` in the backend directory and configure:
- Backend host and port
- CORS origins
- ML model paths
- Translation dictionary paths
- TTS engine configuration

## Verification
1. Check frontend at `http://localhost:5173`
2. Check backend API docs at `http://localhost:8000/docs`
3. Run health check: `curl http://localhost:8000/health`

## Troubleshooting
- Ensure all Python virtual environments are activated
- Check that ports 5173 and 8000 are not in use
- Verify Node.js and Python versions meet requirements
