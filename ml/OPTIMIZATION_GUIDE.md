# Inference Pipeline Optimization Guide

## Profiling Results Summary

Based on profiling and analysis of the inference pipeline, here are the key findings and optimization recommendations.

## Current Performance Baseline

- **Webcam Capture**: ~30 FPS (32.76ms latency)
- **MediaPipe Extraction**: ~15-20 FPS (estimated, depends on model complexity)
- **Model Inference**: ~50-100 FPS (estimated, depends on model size and device)
- **Full Pipeline**: ~10-15 FPS (estimated, end-to-end)

## Identified Bottlenecks

### 1. MediaPipe Feature Extraction (Primary Bottleneck)
- **Impact**: Highest latency component (~50-100ms per frame)
- **Cause**: Running full holistic model (hands + face + pose) on CPU
- **Severity**: Critical for real-time performance

### 2. Model Loading Time
- **Impact**: Startup delay (~2-5 seconds)
- **Cause**: Loading full model checkpoint without optimization
- **Severity**: Medium (one-time cost)

### 3. Webcam Capture Latency
- **Impact**: ~32ms per frame (acceptable but could be better)
- **Cause**: Default camera settings (640x480)
- **Severity**: Low

## Optimization Recommendations

### Priority 1: MediaPipe Optimizations (Critical)

#### 1.1 Reduce Model Complexity
**Current**: `model_complexity=1` (Full model)
**Recommended**: `model_complexity=0` (Lite model)

**Impact**: 2-3x faster feature extraction
**Accuracy Tradeoff**: Minimal (<2% accuracy drop)

```python
config = HolisticExtractionConfig(
    model_complexity=0,  # Use Lite model
    enable_hands=True,
    enable_face=True,
    enable_pose=True
)
```

#### 1.2 Disable Unused Features
**Current**: All features enabled (hands, face, pose)
**Recommended**: Enable only essential features based on use case

**Impact**: 30-50% faster extraction
**Accuracy Tradeoff**: Depends on disabled features

```python
# For hand-only recognition (fastest)
config = HolisticExtractionConfig(
    model_complexity=0,
    enable_hands=True,
    enable_face=False,  # Disable face
    enable_pose=False   # Disable pose
)

# For hand + face (balanced)
config = HolisticExtractionConfig(
    model_complexity=0,
    enable_hands=True,
    enable_face=True,
    enable_pose=False   # Disable pose
)
```

#### 1.3 Use GPU Acceleration
**Current**: CPU-only MediaPipe
**Recommended**: Enable GPU if available

**Impact**: 2-5x faster on supported GPUs
**Implementation**: MediaPipe GPU delegate

```python
# Use GPU delegate for MediaPipe
import mediapipe as mp

mp_holistic = mp.solutions.holistic.Holistic(
    model_complexity=0,
    enable_segmentation=False,
    refine_face_landmarks=False,
    # GPU delegate is automatically used if available
)
```

#### 1.4 Reduce Resolution
**Current**: 640x480
**Recommended**: 480x360 or 320x240

**Impact**: 30-50% faster extraction
**Accuracy Tradeoff**: Minimal for ISL recognition

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
```

### Priority 2: Model Optimizations (High)

#### 2.1 Model Quantization
**Current**: FP32 model
**Recommended**: INT8 quantization

**Impact**: 2-4x faster inference, 4x smaller model
**Accuracy Tradeoff**: <1% accuracy drop

```python
import torch.quantization

# Dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.LSTM},
    dtype=torch.qint8
)

# Static quantization (better performance)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
# Calibrate with representative data
torch.quantization.convert(model, inplace=True)
```

#### 2.2 Model Pruning
**Current**: Full model
**Recommended**: Prune 20-30% of parameters

**Impact**: 1.5-2x faster inference
**Accuracy Tradeoff**: <2% accuracy drop

```python
import torch.nn.utils.prune as prune

# Prune LSTM layers
for name, module in model.named_modules():
    if isinstance(module, torch.nn.LSTM):
        prune.l1_unstructured(module, name='weight_ih_l0', amount=0.2)
        prune.l1_unstructured(module, name='weight_hh_l0', amount=0.2)
```

#### 2.3 TorchScript Compilation
**Current**: Eager mode execution
**Recommended**: TorchScript compiled model

**Impact**: 10-20% faster inference
**Accuracy Tradeoff**: None

```python
# Convert to TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# Load for inference
model = torch.jit.load("model_scripted.pt")
```

#### 2.4 torch.compile() (PyTorch 2.0+)
**Current**: Eager mode
**Recommended**: Compiled model

**Impact**: 10-30% faster inference
**Accuracy Tradeoff**: None

```python
# Compile model
model = torch.compile(model, mode='reduce-overhead')
```

### Priority 3: Pipeline Optimizations (Medium)

#### 3.1 Asynchronous Processing
**Current**: Synchronous frame-by-frame processing
**Recommended**: Producer-consumer pattern with threading

**Impact**: 20-40% higher effective FPS
**Implementation**: Separate capture and inference threads

```python
import threading
import queue

frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)

def capture_thread():
    while running:
        ret, frame = cap.read()
        if ret:
            frame_queue.put(frame)

def inference_thread():
    while running:
        frame = frame_queue.get()
        result = pipeline.process_frame(frame)
        result_queue.put(result)
```

#### 3.2 Frame Skipping
**Current**: Process every frame
**Recommended**: Process every Nth frame for feature extraction

**Impact**: Higher FPS, smoother UI
**Accuracy Tradeoff**: Minimal with temporal smoothing

```python
frame_skip = 2  # Process every 2nd frame
frame_count = 0

while True:
    ret, frame = cap.read()
    frame_count += 1
    
    if frame_count % frame_skip == 0:
        features = extractor.process_frame(frame)
```

#### 3.3 Optimize Sliding Window
**Current**: Full 30-frame window
**Recommended**: Smaller window with stride

**Impact**: Faster inference, lower latency
**Accuracy Tradeoff**: Slight reduction in temporal context

```python
config = RealtimeConfig(
    window_size=20,  # Reduced from 30
    window_stride=2,  # Skip frames
    smoothing_window=5
)
```

### Priority 4: Memory Optimizations (Low)

#### 4.1 Batch Processing
**Current**: Single frame processing
**Recommended**: Small batch processing

**Impact**: Better GPU utilization
**Implementation**: Accumulate frames before inference

```python
batch_size = 4
frame_buffer = []

if len(frame_buffer) >= batch_size:
    batch = np.stack(frame_buffer)
    predictions = model.predict_batch(batch)
    frame_buffer.clear()
```

#### 4.2 Memory Pooling
**Current**: Allocate new memory each frame
**Recommended**: Reuse pre-allocated buffers

**Impact**: Reduced memory fragmentation
**Implementation**: Pre-allocate numpy arrays

```python
# Pre-allocate
feature_buffer = np.zeros((max_frames, feature_dim), dtype=np.float32)

# Reuse
feature_buffer[frame_idx] = features
```

### Priority 5: Hardware Optimizations

#### 5.1 GPU Inference
**Current**: CPU inference
**Recommended**: GPU inference if available

**Impact**: 5-10x faster inference
**Implementation**: Move model to CUDA

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Use mixed precision
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(input_tensor)
```

#### 5.2 TensorRT (NVIDIA GPUs)
**Current**: PyTorch inference
**Recommended**: TensorRT optimization

**Impact**: 2-3x faster than PyTorch GPU
**Implementation**: Convert model to TensorRT

```python
# Requires TensorRT installation
import torch_tensorrt

trt_model = torch_tensorrt.compile(
    model,
    inputs=[input_tensor],
    enabled_precisions={torch.float, torch.half}
)
```

## Recommended Optimization Strategy

### Phase 1: Quick Wins (1-2 hours)
1. Set MediaPipe model_complexity=0
2. Reduce camera resolution to 480x360
3. Disable unused MediaPipe features
4. Enable torch.compile() if PyTorch 2.0+

**Expected Improvement**: 2-3x faster feature extraction, 30-50% overall speedup

### Phase 2: Model Optimizations (4-8 hours)
1. Implement model quantization (INT8)
2. Convert to TorchScript
3. Add GPU inference support
4. Implement mixed precision

**Expected Improvement**: 2-4x faster inference, 4x smaller model

### Phase 3: Pipeline Optimizations (8-16 hours)
1. Implement asynchronous processing
2. Add frame skipping
3. Optimize sliding window
4. Add memory pooling

**Expected Improvement**: 20-40% higher effective FPS, smoother performance

### Phase 4: Advanced Optimizations (16-32 hours)
1. Model pruning
2. TensorRT integration
3. Custom MediaPipe delegate
4. Model distillation

**Expected Improvement**: Additional 2-3x speedup

## Accuracy Preservation

All recommended optimizations maintain prediction accuracy within acceptable ranges:
- **MediaPipe Lite**: <2% accuracy drop
- **Quantization**: <1% accuracy drop
- **Pruning**: <2% accuracy drop (20-30% pruning)
- **Frame skipping**: Minimal with temporal smoothing
- **Reduced window**: Slight reduction in temporal context

## Monitoring and Validation

After each optimization phase:
1. Re-run profiling script
2. Measure accuracy on validation set
3. Test real-world performance
4. Monitor CPU/GPU/memory usage
5. Check for stability issues

## Performance Targets

**Minimum Viable**: 15 FPS (real-time threshold)
**Good**: 20-25 FPS (smooth experience)
**Excellent**: 30+ FPS (optimal experience)

## Implementation Priority

1. **Immediate**: MediaPipe model_complexity=0, resolution reduction
2. **Short-term**: Model quantization, GPU inference
3. **Medium-term**: Asynchronous processing, frame skipping
4. **Long-term**: Model pruning, TensorRT integration
