# Model Architecture

## Current Architecture: CNN-LSTM

### Overview
The CNN-LSTM architecture combines spatial feature extraction (CNN) with temporal sequence modeling (LSTM) for ISL gesture recognition.

### Architecture Details

**Input Shape**: `[sequence_length, landmark_dim]`
- Sequence length: 30 frames
- Landmark dimension: 63 (21 hand landmarks × 3 hands + facial landmarks)

**CNN Layers**:
- Conv1D filters: [64, 128, 256]
- Kernel size: 3
- Activation: ReLU
- Dropout: 0.5

**LSTM Layers**:
- Hidden size: 128
- Number of layers: 2
- Bidirectional: True

**Output Layer**:
- Fully connected with softmax
- Number of classes: 26 (A-Z letters)

### Model Configuration
See `ml/models/v1.0.0/config.json` for detailed parameters.

### Future Architectures
- **Transformer**: Self-attention for long-range dependencies
- **3D CNN**: Direct video processing
- **Graph Neural Networks**: Leverage skeletal graph structure
