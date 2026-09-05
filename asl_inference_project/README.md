# ASL Alphabet Recognition Model - Local Setup

This project contains the setup for running the trained ASL alphabet recognition model locally.

## Project Structure

```
asl_inference_project/
├── models/
│   ├── best_model.pt          # Trained model checkpoint
│   └── model_config.json      # Model configuration
├── model.py                   # AlphabetMLP model definition
├── load_model.py              # Model loading and testing script
├── landmark_extractor.py      # Hand landmark extraction using MediaPipe
├── test_landmarks.py          # Webcam landmark extraction test
├── word_builder.py            # Stability-based word building class
├── tts_handler.py             # Text-to-speech handler with threading
├── main.py                    # Live inference with word building and TTS
├── requirements.txt           # Python dependencies
├── setup_venv.bat             # Virtual environment setup script (Windows)
└── README.md                  # This file
```

## Setup Instructions

### Step 1: Navigate to the project directory

```bash
cd asl_inference_project
```

### Step 2: Create and activate virtual environment

**Option A: Using the setup script (Windows)**
```bash
setup_venv.bat
```

**Option B: Manual setup**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate.bat

# Activate virtual environment (Linux/Mac)
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Test the model loading

```bash
python load_model.py
```

This will:
- Load the trained model from `models/best_model.pt`
- Load the model configuration from `models/model_config.json`
- Print the model architecture and parameter count
- Run a dummy forward pass with random 63-dim input
- Display the output and prediction

### Step 4: Test landmark extraction (optional)

```bash
python test_landmarks.py
```

This will:
- Open your webcam
- Extract hand landmarks from 10 frames
- Display the frame with landmarks drawn
- Print landmark shape and sample values to confirm (63,) output

### Step 5: Run live inference

```bash
python main.py
```

This will:
- Load the trained model
- Open your webcam
- Extract hand landmarks in real-time
- Classify ASL alphabet signs
- Display the predicted letter and confidence
- Build words using stability-based letter confirmation
- Display accumulated word buffer at bottom of screen
- Speak word buffer with text-to-speech
- Draw hand landmarks on the frame
- Press 'q' to quit, 'c' to clear word buffer, 's' to speak

## Expected Output

When you run `python load_model.py`, you should see output similar to:

```
============================================================
ASL Alphabet Recognition Model Loading and Testing
============================================================

1. Loading checkpoint from: models/best_model.pt
2. Using device: cpu
3. Loading config from: models/model_config.json
   Config: {'input_dim': 63, 'hidden_dims': [256, 128, 64], 'num_classes': 29, 'dropout': 0.3, 'use_batch_norm': True, 'activation': 'relu'}
4. Creating model architecture...
5. Loading model weights from checkpoint...
   Checkpoint keys: ['model_state_dict']
   ✓ Model state dict loaded successfully
6. Model Statistics:
   Total parameters: 47,493
   Trainable parameters: 47,493

7. Model Architecture:
AlphabetMLP(
  (network): Sequential(
    (0): Linear(in_features=63, out_features=256, bias=True)
    (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.3, inplace=False)
    (4): Linear(in_features=256, out_features=128, bias=True)
    (5): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): Dropout(p=0.3, inplace=False)
    (8): Linear(in_features=128, out_features=64, bias=True)
    (9): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU(inplace=True)
    (11): Dropout(p=0.3, inplace=False)
    (12): Linear(in_features=64, out_features=29, bias=True)
  )
)

8. Running dummy forward pass with random input...
   Input shape: torch.Size([1, 63])
   Output shape: torch.Size([1, 29])
   Output logits (first 5 classes): [...]
   Probabilities (first 5 classes): [...]
   Predicted class: X
   Confidence: 0.XXXX

============================================================
✓ Model loaded and tested successfully!
============================================================
```

## Model Details

- **Architecture**: AlphabetMLP (Feedforward Neural Network)
- **Input**: 63 features (21 hand landmarks × 3 coordinates from MediaPipe Hands)
- **Hidden Layers**: [256, 128, 64] with BatchNorm, ReLU, Dropout(0.3)
- **Output**: 29 classes (A-Z + 'del' + 'nothing' + 'space')
- **Checkpoint Format**: PyTorch state_dict under key 'model_state_dict'

## Dependencies

- torch>=2.0.0
- opencv-python>=4.8.0
- mediapipe>=0.10.0
- numpy>=1.24.0

## Next Steps

Once the model loads successfully, you can:
1. Run `python main.py` for real-time ASL alphabet recognition
2. Extend the main.py with word-building logic
3. Add additional features like gesture smoothing, text-to-speech, etc.

## Live Inference Features

The main.py script provides:
- Real-time hand landmark extraction using MediaPipe Hands
- Instant ASL alphabet classification using the trained model
- Stability-based word building to reduce prediction noise
- Text-to-speech output using pyttsx3 (offline-capable)
- Visual feedback with hand landmarks drawn on frame
- Predicted letter and confidence percentage overlay
- Stability progress bar showing letter confirmation progress
- Accumulated word buffer display at bottom of screen
- Speaking indicator when TTS is active
- GPU acceleration if available (CUDA), otherwise CPU
- Clean exit on 'q' key press with proper resource cleanup
- Manual word buffer clear with 'c' key
- Speak word buffer with 's' key

## Word Building Logic

The WordBuilder class implements stability-based word construction:
- Requires 5 consecutive frames with same prediction (configurable)
- Confidence threshold of 0.7 (configurable)
- Special characters handled:
  - 'space': Adds space to word buffer
  - 'del': Removes last character from word buffer
  - 'nothing': No action (no gesture class)
- Same letter must re-stabilize to be entered again (prevents spamming)
- Visual stability progress bar shows confirmation progress

## Text-to-Speech Features

The TTSHandler class provides offline text-to-speech functionality:
- Uses pyttsx3 (works without internet, cross-platform)
- Threaded execution to avoid blocking video loop
- Configurable auto-clear after speaking
- Visual "Speaking..." indicator during playback
- Voice selection and speech rate configuration
- Press 's' to speak the current word buffer

## Additional Dependencies

To enable text-to-speech functionality, install pyttsx3:

```bash
pip install pyttsx3
```

Note: pyttsx3 is included in requirements.txt, so it will be installed automatically when you run the setup script.
