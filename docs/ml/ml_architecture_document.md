# SilentVoice ML Architecture Document

## Overview
This document provides a comprehensive architectural design for the Machine Learning pipeline of SilentVoice, an Indian Sign Language (ISL) recognition system. The design prioritizes real-time performance, accuracy, and production readiness.

---

## 1. Dataset Structure

### Recommended Dataset Organization

```
ml/datasets/
├── raw/
│   ├── videos/
│   │   ├── train/
│   │   │   ├── hello/
│   │   │   │   ├── subject_001_hello_001.mp4
│   │   │   │   ├── subject_001_hello_002.mp4
│   │   │   │   └── ...
│   │   │   ├── thank_you/
│   │   │   └── ...
│   │   ├── val/
│   │   └── test/
│   └── metadata.csv
├── processed/
│   ├── landmarks/
│   │   ├── train/
│   │   │   ├── hello/
│   │   │   │   ├── subject_001_hello_001_landmarks.npy
│   │   │   │   └── ...
│   │   │   └── ...
│   │   ├── val/
│   │   └── test/
│   └── sequences/
│       ├── train/
│       │   ├── sequences.npy
│       │   └── labels.npy
│       ├── val/
│       └── test/
└── augmented/
    ├── train/
    └── val/
```

### Metadata Structure (metadata.csv)

```csv
video_id,gesture,subject,video_path,duration,frame_count,quality,lighting_condition,background
subject_001_hello_001,hello,001,raw/videos/train/hello/subject_001_hello_001.mp4,2.5,60,high,uniform,plain
subject_001_hello_002,hello,001,raw/videos/train/hello/subject_001_hello_002.mp4,2.3,55,medium,variable,complex
...
```

### Design Decisions

**Subject-Based Splitting**: Use subject-based train/val/test splits (not random) to prevent data leakage and ensure generalization to new users. This is critical for real-world deployment where the system must recognize gestures from unseen individuals.

**Video Quality Metadata**: Track lighting conditions and background complexity to enable analysis of model performance across different environments. This helps identify failure modes and guide data collection efforts.

**Multiple Samples per Gesture**: Collect 5-10 samples per gesture per subject to capture natural variations in signing speed, style, and execution.

**Gesture Categories**: Organize gestures into semantic categories (greetings, courtesy, questions, etc.) to enable hierarchical analysis and potential category-specific fine-tuning.

---

## 2. Data Preprocessing Pipeline

### Pipeline Stages

#### Stage 1: Video Quality Filtering
```python
# Pseudocode
def filter_videos(video_path, min_duration=1.0, max_duration=5.0, min_fps=24):
    duration = get_video_duration(video_path)
    fps = get_video_fps(video_path)
    
    if duration < min_duration or duration > max_duration:
        return False
    if fps < min_fps:
        return False
    return True
```

**Design Decision**: Filter videos to 1-5 seconds duration and minimum 24 FPS. This ensures consistent temporal resolution while accommodating natural signing speed variations. Too short videos may not capture complete gestures; too long videos increase computational cost.

#### Stage 2: Frame Extraction
```python
# Pseudocode
def extract_frames(video_path, target_fps=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    # Temporal resampling to consistent FPS
    frames = resample_frames(frames, original_fps, target_fps)
    return frames
```

**Design Decision**: Resample all videos to 30 FPS for consistency. This standardizes temporal resolution across different recording conditions while maintaining smooth motion capture.

#### Stage 3: Frame Quality Enhancement
```python
# Pseudocode
def enhance_frame(frame):
    # Histogram equalization for lighting normalization
    frame = cv2.equalizeHist(frame)
    
    # Gaussian blur for noise reduction
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    
    # Contrast enhancement
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
    
    return frame
```

**Design Decision**: Apply mild enhancement rather than aggressive preprocessing. Over-processing can remove subtle hand movements and facial expressions that are important for ISL recognition. The goal is lighting normalization, not feature alteration.

#### Stage 4: Background Removal (Optional)
```python
# Pseudocode
def remove_background(frame, method='mediapipe'):
    if method == 'mediapipe':
        # Use MediaPipe segmentation
        mask = mediapipe_segmentation(frame)
        return frame * mask
    elif method == 'grabcut':
        # Use GrabCut algorithm
        mask = grabcut_segmentation(frame)
        return frame * mask
```

**Design Decision**: Background removal is optional and experimental. While it can reduce complexity, it may also introduce artifacts and increase processing time. For initial development, work with raw frames and evaluate if background removal improves performance.

---

## 3. Landmark Extraction Using MediaPipe

### MediaPipe Configuration

```python
# Recommended MediaPipe Hands Configuration
hands = mp.solutions.hands.Hands(
    static_image_mode=False,           # Video stream mode
    max_num_hands=2,                   # ISL often uses two hands
    model_complexity=1,                # Medium complexity (balance speed/accuracy)
    min_detection_confidence=0.5,       # Detection threshold
    min_tracking_confidence=0.5         # Tracking threshold
)

# MediaPipe Face Configuration
face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,              # Enable refined landmarks for lips/eyes
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# MediaPipe Pose Configuration (Optional)
pose = mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

### Landmark Selection Strategy

**Primary Landmarks (Essential)**:
- **Hands**: 21 landmarks per hand × 2 hands = 42 landmarks
  - Each landmark: (x, y, z) coordinates
  - Total: 42 × 3 = 126 dimensions

**Secondary Landmarks (Optional)**:
- **Face**: 468 landmarks (full mesh) or 10 key landmarks (lips, eyes, eyebrows)
  - Recommended: 10 key facial landmarks for expressions
  - Each landmark: (x, y, z) coordinates
  - Total: 10 × 3 = 30 dimensions

**Total Feature Dimensions**: 126 (hands) + 30 (face) = 156 dimensions per frame

### Landmark Extraction Process

```python
# Pseudocode
def extract_landmarks(frame):
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Extract hand landmarks
    hand_results = hands.process(rgb_frame)
    hand_landmarks = []
    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            hand_landmarks.append(landmarks)
    
    # Pad if fewer than 2 hands detected
    while len(hand_landmarks) < 2:
        hand_landmarks.append([[0, 0, 0]] * 21))
    
    # Extract facial landmarks
    face_results = face.process(rgb_frame)
    facial_landmarks = []
    
    if face_results.multi_face_landmarks:
        face_lm = face_results.multi_face_landmarks[0]
        # Select 10 key landmarks
        key_indices = [13, 14, 61, 146, 159, 145, 33, 263, 291, 33]
        for idx in key_indices:
            lm = face_lm.landmark[idx]
            facial_landmarks.append([lm.x, lm.y, lm.z])
    
    # Combine landmarks
    combined_landmarks = hand_landmarks[0] + hand_landmarks[1] + facial_landmarks
    
    return np.array(combined_landmarks)  # Shape: (156,)
```

### Design Decisions

**Two-Hand Detection**: ISL frequently uses both hands simultaneously. Configuring MediaPipe to detect up to 2 hands is essential. If only one hand is detected, pad with zeros to maintain consistent input dimensions.

**Z-Coordinate Inclusion**: MediaPipe provides normalized Z-coordinates representing depth. Including Z-coordinates helps distinguish overlapping fingers and adds 3D spatial context, improving accuracy for complex gestures.

**Facial Landmarks**: ISL incorporates facial expressions (e.g., questioning expressions). Including key facial landmarks (lips, eyes, eyebrows) captures these expressions without excessive computational overhead.

**Medium Model Complexity**: Use model_complexity=1 (medium) rather than 0 (lite) or 2 (full). This provides a good balance between accuracy and speed for real-time inference.

**Confidence Thresholds**: Set detection and tracking confidence to 0.5. Lower thresholds increase false positives; higher thresholds may miss subtle gestures. 0.5 is a empirically good starting point.

---

## 4. Sequence Generation

### Sequence Length Strategy

**Recommended Sequence Length**: 30 frames (1 second at 30 FPS)

**Rationale**:
- Most ISL gestures complete within 0.5-2 seconds
- 30 frames provides sufficient temporal context without excessive computational cost
- Shorter sequences may miss gesture transitions
- Longer sequences increase latency and memory requirements

### Sequence Generation Methods

#### Method 1: Fixed-Length Windowing (Recommended)
```python
# Pseudocode
def generate_fixed_sequences(landmarks, sequence_length=30, stride=15):
    sequences = []
    labels = []
    
    for i in range(0, len(landmarks) - sequence_length + 1, stride):
        sequence = landmarks[i:i + sequence_length]
        sequences.append(sequence)
        labels.append(gesture_label)
    
    return sequences, labels
```

**Design Decision**: Use stride=15 (50% overlap) to increase training samples and capture temporal variations. Overlapping windows help the model learn gesture invariance to temporal shifts.

#### Method 2: Dynamic-Length Padding (Alternative)
```python
# Pseudocode
def generate_dynamic_sequences(landmarks, max_length=30):
    if len(landmarks) < max_length:
        # Pad with zeros
        padding = np.zeros((max_length - len(landmarks), landmark_dim))
        sequence = np.vstack([landmarks, padding])
    else:
        # Truncate to max_length
        sequence = landmarks[:max_length]
    
    return sequence
```

**Design Decision**: Dynamic-length padding accommodates variable gesture durations but may introduce noise from zero-padding. Fixed-length windowing is preferred for consistency and performance.

### Temporal Normalization

```python
# Pseudocode
def normalize_sequence_temporal(sequence):
    # Option 1: No normalization (preserve temporal dynamics)
    # Option 2: Linear interpolation to fixed length
    # Option 3: Temporal scaling based on gesture speed
    
    # Recommended: No normalization for initial training
    # Preserves natural signing speed variations
    return sequence
```

**Design Decision**: Do not apply temporal normalization initially. ISL gestures have meaningful temporal dynamics (fast vs slow signing). Preserving these dynamics allows the model to learn speed-invariant representations naturally.

---

## 5. Data Augmentation

### Augmentation Techniques

#### Spatial Augmentations
```python
# Pseudocode
def spatial_augmentation(landmarks, augmentation_type):
    if augmentation_type == 'rotation':
        # Random rotation in 2D plane
        angle = np.random.uniform(-15, 15)
        landmarks = rotate_landmarks(landmarks, angle)
    
    elif augmentation_type == 'scaling':
        # Random scaling
        scale = np.random.uniform(0.9, 1.1)
        landmarks = scale_landmarks(landmarks, scale)
    
    elif augmentation_type == 'translation':
        # Random translation
        shift_x = np.random.uniform(-0.05, 0.05)
        shift_y = np.random.uniform(-0.05, 0.05)
        landmarks = translate_landmarks(landmarks, shift_x, shift_y)
    
    return landmarks
```

**Design Decision**: Apply mild spatial augmentations (±15° rotation, ±10% scaling, ±5% translation). Aggressive augmentation may distort gesture semantics. The goal is invariance to camera positioning, not gesture alteration.

#### Temporal Augmentations
```python
# Pseudocode
def temporal_augmentation(sequence, augmentation_type):
    if augmentation_type == 'speed':
        # Time warping
        speed_factor = np.random.uniform(0.8, 1.2)
        sequence = time_warp(sequence, speed_factor)
    
    elif augmentation_type == 'jitter':
        # Add temporal noise
        noise = np.random.normal(0, 0.01, sequence.shape)
        sequence = sequence + noise
    
    return sequence
```

**Design Decision**: Temporal augmentation is experimental. Time warping may disrupt gesture timing semantics. Use sparingly and evaluate impact on validation performance.

#### Dropout Augmentation
```python
# Pseudocode
def dropout_augmentation(sequence, dropout_rate=0.1):
    mask = np.random.random(sequence.shape) > dropout_rate
    sequence = sequence * mask
    return sequence
```

**Design Decision**: Randomly drop 10% of landmarks to simulate occlusion and tracking failures. This improves robustness to real-world conditions where hands may be partially occluded.

### Augmentation Strategy

**Recommended Approach**:
- Apply spatial augmentations to 50% of training data
- Apply dropout augmentation to 30% of training data
- Do not apply temporal augmentations initially
- Evaluate augmentation impact on validation set
- Disable augmentations if validation performance degrades

**Rationale**: Conservative augmentation strategy. Over-augmentation can teach the model to recognize augmented artifacts rather than genuine gesture patterns. Start mild and increase only if needed.

---

## 6. Feature Engineering

### Raw Features vs. Engineered Features

**Recommended Approach**: Use raw landmarks initially, then experiment with engineered features if performance plateaus.

### Raw Features (Baseline)
```python
# Input: (sequence_length, landmark_dim)
# Shape: (30, 156)
# Features: Normalized (x, y, z) coordinates for each landmark
```

**Design Decision**: Raw landmarks provide the most information and allow the model to learn relevant features automatically. This is the recommended starting point.

### Engineered Features (Advanced)

#### Relative Position Features
```python
# Pseudocode
def compute_relative_positions(landmarks):
    # Compute positions relative to wrist (landmark 0)
    wrist = landmarks[0]
    relative_landmarks = landmarks - wrist
    return relative_landmarks
```

**Rationale**: Relative positions provide translation invariance. The model learns gesture patterns independent of hand position in the frame.

#### Velocity Features
```python
# Pseudocode
def compute_velocities(sequence):
    velocities = np.diff(sequence, axis=0)
    # Pad to maintain sequence length
    velocities = np.vstack([velocities, velocities[-1]])
    return velocities
```

**Rationale**: Velocity features capture motion dynamics, which are crucial for distinguishing similar gestures with different motion patterns.

#### Acceleration Features
```python
# Pseudocode
def compute_accelerations(sequence):
    velocities = np.diff(sequence, axis=0)
    accelerations = np.diff(velocities, axis=0)
    # Pad to maintain sequence length
    accelerations = np.vstack([accelerations, accelerations[-1], accelerations[-1]])
    return accelerations
```

**Rationale**: Acceleration features capture motion changes, helpful for gestures with sharp movements vs smooth transitions.

#### Distance Features
```python
# Pseudocode
def compute_distances(landmarks):
    # Compute distances between key landmark pairs
    # e.g., thumb tip to index tip, wrist to middle finger tip
    distances = []
    key_pairs = [(4, 8), (0, 12), (8, 12)]  # Example pairs
    for pair in key_pairs:
        dist = np.linalg.norm(landmarks[pair[0]] - landmarks[pair[1]])
        distances.append(dist)
    return np.array(distances)
```

**Rationale**: Distance features capture hand shape configurations (e.g., open vs closed hand) independent of position and orientation.

### Feature Combination Strategy

**Recommended Progression**:
1. **Phase 1**: Raw landmarks only (baseline)
2. **Phase 2**: Raw + relative positions (if translation invariance needed)
3. **Phase 3**: Raw + relative + velocities (if motion dynamics important)
4. **Phase 4**: Full feature set (if performance plateaus)

**Design Decision**: Incremental feature addition allows isolation of each feature's contribution. Add features only if they provide measurable improvement on validation set.

---

## 7. Model Candidates

### Model Architecture Comparison

#### LSTM (Long Short-Term Memory)

**Architecture**:
```
Input: (30, 156)
↓
LSTM Layer 1: 128 units, return_sequences=True
↓
Dropout: 0.5
↓
LSTM Layer 2: 128 units, return_sequences=False
↓
Dropout: 0.5
↓
Dense: 64 units, ReLU
↓
Output: num_classes units, Softmax
```

**Pros**:
- Well-established for sequence modeling
- Handles long-term dependencies
- Computationally efficient
- Easy to implement and debug

**Cons**:
- Sequential processing limits parallelization
- May struggle with very long sequences
- Less expressive than attention mechanisms

**Recommended Use Case**: Baseline model, good starting point for quick iteration.

#### BiLSTM (Bidirectional LSTM)

**Architecture**:
```
Input: (30, 156)
↓
BiLSTM Layer 1: 128 units (64 forward + 64 backward), return_sequences=True
↓
Dropout: 0.5
↓
BiLSTM Layer 2: 128 units, return_sequences=False
↓
Dropout: 0.5
↓
Dense: 64 units, ReLU
↓
Output: num_classes units, Softmax
```

**Pros**:
- Captures context from both past and future
- Better for gestures where ending frames inform beginning interpretation
- Still relatively efficient

**Cons**:
- 2x computational cost of unidirectional LSTM
- May not be necessary for real-time inference (future context unavailable)

**Recommended Use Case**: If unidirectional LSTM plateaus and gesture interpretation benefits from future context.

#### GRU (Gated Recurrent Unit)

**Architecture**:
```
Input: (30, 156)
↓
GRU Layer 1: 128 units, return_sequences=True
↓
Dropout: 0.5
↓
GRU Layer 2: 128 units, return_sequences=False
↓
Dropout: 0.5
↓
Dense: 64 units, ReLU
↓
Output: num_classes units, Softmax
```

**Pros**:
- Simpler architecture than LSTM (fewer parameters)
- Faster training and inference
- Often performs similarly to LSTM on shorter sequences

**Cons**:
- May not capture very long-term dependencies as well as LSTM
- Less established in literature for gesture recognition

**Recommended Use Case**: If training speed is critical and sequences are short (< 50 frames).

#### Transformer

**Architecture**:
```
Input: (30, 156)
↓
Positional Encoding
↓
Transformer Encoder Block 1:
  - Multi-Head Attention: 8 heads, 64 dimensions
  - Layer Normalization
  - Feed-Forward: 256 units
  - Dropout: 0.1
↓
Transformer Encoder Block 2: (same as above)
↓
Global Average Pooling
↓
Dense: 64 units, ReLU
↓
Output: num_classes units, Softmax
```

**Pros**:
- Captures long-range dependencies effectively
- Parallelizable training (faster on GPU)
- Attention mechanism provides interpretability
- State-of-the-art performance on many sequence tasks

**Cons**:
- Higher computational cost
- Requires more data to train effectively
- More complex to tune and debug
- May be overkill for short sequences

**Recommended Use Case**: If LSTM-based models plateau and you have sufficient training data (> 10,000 samples).

#### CNN-LSTM Hybrid (Recommended for SilentVoice)

**Architecture**:
```
Input: (30, 156)
↓
Reshape: (30, 156, 1)  # Add channel dimension
↓
1D Conv Layer 1: 64 filters, kernel_size=3, padding='same'
↓
Batch Normalization
↓
ReLU Activation
↓
Max Pooling: pool_size=2
↓
1D Conv Layer 2: 128 filters, kernel_size=3, padding='same'
↓
Batch Normalization
↓
ReLU Activation
↓
Max Pooling: pool_size=2
↓
Reshape: (sequence_length, features)
↓
LSTM Layer: 128 units, return_sequences=False
↓
Dropout: 0.5
↓
Dense: 64 units, ReLU
↓
Output: num_classes units, Softmax
```

**Pros**:
- CNN extracts spatial features from landmark configurations
- LSTM models temporal dynamics
- Hybrid approach leverages strengths of both architectures
- Proven effective for gesture recognition in literature

**Cons**:
- More complex architecture
- Additional hyperparameters to tune
- Longer training time

**Recommended Use Case**: **Primary recommendation for SilentVoice**. This architecture balances performance, complexity, and real-time requirements.

### Model Selection Recommendation

**Phase 1 (Baseline)**: Start with simple LSTM
- Quick to implement and train
- Establishes performance baseline
- Identifies data quality issues

**Phase 2 (Optimization)**: Move to CNN-LSTM hybrid
- Likely to provide significant accuracy improvement
- Still maintains reasonable inference speed
- Well-suited for ISL gesture recognition

**Phase 3 (Advanced)**: Experiment with Transformer if needed
- Only if CNN-LSTM plateaus
- Requires substantial training data
- Consider for production if accuracy gains justify computational cost

---

## 8. Hyperparameter Tuning Strategy

### Tuning Framework

**Recommended Tool**: Optuna (Bayesian optimization)

**Alternative Tools**: Ray Tune, Hyperopt, Grid Search (for small parameter spaces)

### Hyperparameter Space Definition

#### Model Architecture Hyperparameters

```python
# CNN-LSTM Hybrid Hyperparameter Space
search_space = {
    # CNN layers
    'cnn_filters_1': [32, 64, 128],
    'cnn_filters_2': [64, 128, 256],
    'cnn_kernel_size': [3, 5, 7],
    'cnn_pool_size': [2, 3],
    
    # LSTM layer
    'lstm_units': [64, 128, 256],
    'lstm_num_layers': [1, 2, 3],
    
    # Dense layer
    'dense_units': [32, 64, 128],
    
    # Regularization
    'dropout_rate': [0.3, 0.5, 0.7],
    'l2_regularization': [1e-4, 1e-3, 1e-2],
}
```

#### Training Hyperparameters

```python
training_space = {
    'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
    'batch_size': [16, 32, 64],
    'optimizer': ['adam', 'adamw', 'sgd'],
    'weight_decay': [0, 1e-5, 1e-4],
    'label_smoothing': [0.0, 0.1, 0.2],
}
```

#### Data Hyperparameters

```python
data_space = {
    'sequence_length': [20, 30, 40],
    'landmark_dim': [126, 156],  # Hands only vs Hands + Face
    'normalization': ['minmax', 'standard', 'none'],
    'augmentation_probability': [0.0, 0.3, 0.5],
}
```

### Tuning Strategy

#### Stage 1: Coarse Search
- Use wide parameter ranges
- Few trials (50-100)
- Short training epochs (10-20)
- Goal: Identify promising regions of parameter space

#### Stage 2: Fine Search
- Narrow parameter ranges around best Stage 1 results
- More trials (200-300)
- Full training epochs (50-100)
- Goal: Optimize within promising regions

#### Stage 3: Final Validation
- Train top 5 configurations from Stage 2
- Full training (100-200 epochs)
- Evaluate on test set
- Select best performing model

### Early Stopping Strategy

```python
# Recommended Early Stopping Configuration
early_stopping = {
    'monitor': 'val_loss',
    'patience': 15,
    'min_delta': 0.001,
    'mode': 'min',
    'restore_best_weights': True
}
```

**Design Decision**: Use validation loss for early stopping with patience=15. This allows the model to overcome temporary plateaus while preventing overfitting. Restore best weights to ensure optimal model selection.

### Learning Rate Scheduling

**Recommended Schedule**: Cosine Annealing with Warm Restart

```python
# Pseudocode
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # Initial restart period
    T_mult=2,  # Period multiplication factor
    eta_min=1e-6  # Minimum learning rate
)
```

**Rationale**: Cosine annealing helps escape local minima. Warm restarts allow the learning rate to periodically increase, enabling exploration of new parameter regions.

### Cross-Validation Strategy

**Recommended Approach**: Subject-based 5-fold cross-validation

**Design Decision**: Use subject-based splits rather than random splits. This ensures the model generalizes to unseen individuals, which is critical for real-world deployment.

---

## 9. Evaluation Metrics

### Primary Metrics

#### Accuracy
```python
accuracy = correct_predictions / total_predictions
```

**Use Case**: Overall performance metric, easy to communicate to stakeholders.

#### Top-K Accuracy
```python
top_k_accuracy = (predicted_class in top_k_true_classes) / total_predictions
```

**Recommended**: Top-3 accuracy for ISL recognition. Gestures can be similar; top-3 provides insight into model confidence.

#### Precision, Recall, F1-Score
```python
# Per-class metrics
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

# Macro-averaged (equal weight per class)
macro_precision = mean(precision_per_class)
macro_recall = mean(recall_per_class)
macro_f1 = mean(f1_per_class)

# Weighted-averaged (weighted by class frequency)
weighted_precision = weighted_mean(precision_per_class, class_weights)
weighted_recall = weighted_mean(recall_per_class, class_weights)
weighted_f1 = weighted_mean(f1_per_class, class_weights)
```

**Design Decision**: Report both macro and weighted averages. Macro average highlights performance on rare gestures; weighted average reflects overall system performance.

### Secondary Metrics

#### Confusion Matrix
```python
# Visualize per-class errors
# Identify commonly confused gesture pairs
# Guide data collection for problematic classes
```

**Use Case**: Error analysis, identify gesture pairs that need disambiguation.

#### Class Distribution Analysis
```python
# Analyze performance vs class frequency
# Identify if model struggles with rare classes
# Guide data augmentation strategies
```

**Use Case**: Ensure balanced performance across gesture classes.

#### Inference Time Metrics
```python
avg_inference_time = mean(inference_times_per_sample)
p95_inference_time = percentile(inference_times_per_sample, 95)
p99_inference_time = percentile(inference_times_per_sample, 99)
```

**Use Case**: Real-time performance validation. Target: < 100ms per inference for smooth real-time translation.

#### Memory Usage
```python
model_memory_size = model_parameters * 4 bytes  # FP32
inference_memory = peak_memory_during_inference
```

**Use Case**: Deployment constraints, especially for edge devices.

### Evaluation Protocol

#### Training Set Metrics
- Monitor for overfitting
- Should achieve > 95% accuracy (model capacity check)

#### Validation Set Metrics
- Primary metric for hyperparameter tuning
- Target: > 85% accuracy for production readiness

#### Test Set Metrics
- Final evaluation only (no tuning based on test set)
- Target: > 80% accuracy for production deployment

#### Real-World Validation
- Collect additional test set from different subjects
- Test in varied lighting conditions
- Test with different camera angles
- Target: > 75% accuracy in real-world conditions

### Metric Thresholds for Production

**Minimum Viable**:
- Accuracy: > 70%
- Top-3 Accuracy: > 85%
- Inference Time: < 200ms
- Macro F1: > 0.65

**Production Ready**:
- Accuracy: > 80%
- Top-3 Accuracy: > 90%
- Inference Time: < 100ms
- Macro F1: > 0.75

**Excellent**:
- Accuracy: > 90%
- Top-3 Accuracy: > 95%
- Inference Time: < 50ms
- Macro F1: > 0.85

---

## 10. Real-Time Inference Pipeline

### Pipeline Architecture

```
Video Stream (Camera)
    ↓
Frame Buffer (Circular Buffer, 30 frames)
    ↓
Frame Preprocessing
    ↓
Landmark Extraction (MediaPipe)
    ↓
Sequence Construction (Sliding Window)
    ↓
Model Inference (CNN-LSTM)
    ↓
Post-Processing
    ↓
Output (Gesture Label + Confidence)
```

### Component Design

#### Frame Buffer
```python
# Pseudocode
class FrameBuffer:
    def __init__(self, buffer_size=30):
        self.buffer = collections.deque(maxlen=buffer_size)
    
    def add_frame(self, frame):
        self.buffer.append(frame)
    
    def get_sequence(self):
        return list(self.buffer)
    
    def is_ready(self):
        return len(self.buffer) == self.buffer.maxlen
```

**Design Decision**: Use circular buffer of 30 frames. This maintains the most recent frames for inference while managing memory efficiently. Buffer is ready when full (30 frames accumulated).

#### Sliding Window Inference
```python
# Pseudocode
def sliding_window_inference(buffer, model, stride=5):
    if not buffer.is_ready():
        return None, 0.0
    
    sequence = buffer.get_sequence()
    
    # Extract landmarks for entire sequence
    landmarks = [extract_landmarks(frame) for frame in sequence]
    
    # Run inference
    prediction = model.predict(landmarks)
    confidence = max(prediction)
    gesture = gesture_classes[np.argmax(prediction)]
    
    return gesture, confidence
```

**Design Decision**: Run inference every 5 frames (stride=5) rather than every frame. This reduces computational load while maintaining reasonable responsiveness. 5-frame stride = 6 inferences per second at 30 FPS.

#### Confidence-Based Filtering
```python
# Pseudocode
def filter_prediction(gesture, confidence, threshold=0.7):
    if confidence < threshold:
        return None, 0.0  # Reject low-confidence predictions
    return gesture, confidence
```

**Design Decision**: Reject predictions below confidence threshold of 0.7. This prevents false positives and improves user experience. Threshold can be adjusted based on user feedback.

#### Temporal Smoothing
```python
# Pseudocode
class TemporalSmoother:
    def __init__(self, window_size=5):
        self.window = collections.deque(maxlen=window_size)
    
    def smooth(self, gesture, confidence):
        self.window.append((gesture, confidence))
        
        # Majority voting with confidence weighting
        gesture_votes = {}
        for g, c in self.window:
            if g not in gesture_votes:
                gesture_votes[g] = 0
            gesture_votes[g] += c
        
        best_gesture = max(gesture_votes, key=gesture_votes.get)
        avg_confidence = gesture_votes[best_gesture] / len(self.window)
        
        return best_gesture, avg_confidence
```

**Design Decision**: Apply temporal smoothing over 5 recent predictions. This reduces jitter and provides more stable output. Confidence-weighted voting gives more weight to high-confidence predictions.

### Performance Optimization

#### Model Optimization
```python
# Convert model to TorchScript for faster inference
model = torch.jit.script(model)

# Or use ONNX for cross-platform deployment
torch.onnx.export(model, input_sample, "model.onnx")
```

**Design Decision**: Use TorchScript or ONNX for production inference. Both provide significant speedup over PyTorch eager execution.

#### Batch Processing
```python
# Process multiple sequences in batch
batch_sequences = [seq1, seq2, seq3, seq4]
predictions = model.predict_batch(batch_sequences)
```

**Design Decision**: If processing multiple camera streams or users, batch inference for GPU utilization. For single-user real-time, batch processing not beneficial.

#### Landmark Caching
```python
# Cache landmarks to avoid recomputation
landmark_cache = {}
def get_cached_landmarks(frame_hash):
    if frame_hash in landmark_cache:
        return landmark_cache[frame_hash]
    landmarks = extract_landmarks(frame)
    landmark_cache[frame_hash] = landmarks
    return landmarks
```

**Design Decision**: Cache landmarks if frames are reused (e.g., multiple sliding windows). For real-time streaming with stride=5, caching may not provide significant benefit.

### Latency Budget

**Target Total Latency**: < 100ms

**Budget Allocation**:
- Frame Capture: 5ms
- Frame Preprocessing: 10ms
- Landmark Extraction: 30ms
- Sequence Construction: 5ms
- Model Inference: 40ms
- Post-Processing: 10ms

**Total**: 100ms (10 FPS effective inference rate)

### Fallback Strategies

#### Landmark Extraction Failure
```python
if landmarks is None:
    # Use previous frame's landmarks
    landmarks = previous_landmarks
    
    # Or interpolate from neighboring frames
    landmarks = interpolate_landmarks(buffer)
```

**Design Decision**: If MediaPipe fails to detect landmarks, use previous frame's landmarks or interpolate. This maintains continuity rather than dropping frames.

#### Model Inference Failure
```python
try:
    prediction = model.predict(landmarks)
except Exception as e:
    # Use cached prediction
    prediction = cached_prediction
    
    # Or return neutral gesture
    prediction = neutral_class
```

**Design Decision**: Graceful degradation on model failure. Use cached prediction or neutral gesture rather than crashing.

---

## Summary and Recommendations

### Recommended Architecture for SilentVoice

**Model**: CNN-LSTM Hybrid
- CNN: 2 layers (64, 128 filters), kernel_size=3
- LSTM: 1 layer, 128 units
- Dropout: 0.5
- Output: Dense layer with softmax

**Input**: 30-frame sequences, 156-dimensional landmarks (hands + face)

**Training**: Adam optimizer, learning rate=1e-3, batch_size=32, cosine annealing scheduler

**Data**: Subject-based train/val/test split, spatial augmentation (50%), dropout augmentation (30%)

**Evaluation**: Target > 80% test accuracy, > 90% top-3 accuracy, < 100ms inference time

**Inference**: Sliding window with stride=5, confidence threshold=0.7, temporal smoothing over 5 predictions

### Implementation Phases

**Phase 1 (Weeks 1-2)**: Data Collection and Preprocessing
- Collect ISL gesture videos
- Implement landmark extraction pipeline
- Generate training/validation/test splits

**Phase 2 (Weeks 3-4)**: Baseline Model
- Implement LSTM baseline
- Establish performance baseline
- Identify data quality issues

**Phase 3 (Weeks 5-6)**: CNN-LSTM Optimization
- Implement CNN-LSTM hybrid
- Hyperparameter tuning with Optuna
- Achieve target accuracy

**Phase 4 (Weeks 7-8)**: Real-Time Inference
- Implement inference pipeline
- Optimize for latency
- Integrate with frontend

**Phase 5 (Weeks 9-10)**: Testing and Refinement
- Real-world testing
- User feedback integration
- Performance optimization

### Success Criteria

**Minimum Viable Product**:
- Recognize 10 basic ISL gestures
- > 70% accuracy
- < 200ms inference time
- Stable real-time performance

**Production Ready**:
- Recognize 26 ISL gestures (A-Z)
- > 80% accuracy
- < 100ms inference time
- Robust to lighting and background variations

**Excellent**:
- Recognize 50+ ISL gestures
- > 90% accuracy
- < 50ms inference time
- Generalizes to unseen subjects
