# INCLUDE Dataset Setup Guide

## Overview

The INCLUDE (Indian Sign Language) dataset is used for training the SilentVoice ISL recognition model. This guide explains how to download, organize, and prepare the dataset for training.

## INCLUDE Dataset Information

The INCLUDE dataset contains Indian Sign Language videos with annotations for gesture recognition. It includes:
- Video recordings of ISL gestures
- Subject information
- Gesture labels
- Metadata for each video

## Step 1: Download the INCLUDE Dataset

### Option A: Official Source (if available)
```bash
# Navigate to datasets directory
cd ml/datasets

# Download INCLUDE dataset (replace with actual URL)
# This is a placeholder - replace with actual download command
wget https://example.com/include-dataset.zip
unzip include-dataset.zip
```

### Option B: Manual Download
1. Visit the official INCLUDE dataset repository
2. Download the dataset files
3. Extract to the appropriate location

## Step 2: Organize the Dataset

The dataset should be organized in the following structure:

```
ml/datasets/
├── raw/
│   ├── INCLUDE/
│   │   ├── videos/
│   │   │   ├── subject_01/
│   │   │   │   ├── gesture_01.mp4
│   │   │   │   ├── gesture_02.mp4
│   │   │   │   └── ...
│   │   │   ├── subject_02/
│   │   │   │   └── ...
│   │   │   └── ...
│   │   ├── metadata.csv
│   │   ├── labels.txt
│   │   └── README.md
│   └── metadata.csv
├── processed/
│   ├── landmarks/
│   ├── sequences/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── quality_reports/
│   └── logs/
└── metadata_manager.py
```

## Step 3: Create Metadata File

The preprocessing pipeline requires a `metadata.csv` file with the following structure:

```csv
video_id,video_path,subject,gesture,duration,fps,split
001,INCLUDE/videos/subject_01/gesture_01.mp4,subject_01,hello,2.5,30,train
002,INCLUDE/videos/subject_01/gesture_02.mp4,subject_01,thank_you,3.0,30,train
003,INCLUDE/videos/subject_02/gesture_01.mp4,subject_02,hello,2.8,30,val
...
```

### Required Columns:
- `video_id`: Unique identifier for each video
- `video_path`: Relative path to video file from dataset root
- `subject`: Subject identifier (for subject-based splitting)
- `gesture`: Gesture label (the class name)
- `duration`: Video duration in seconds
- `fps`: Video frame rate
- `split`: Dataset split (train/val/test) - optional, will be generated automatically

### Creating Metadata File

If the INCLUDE dataset doesn't include a metadata file, create one using the provided script:

```bash
python ml/datasets/create_include_metadata.py \
    --dataset-root ml/datasets/raw/INCLUDE \
    --output ml/datasets/raw/metadata.csv
```

## Step 4: Dataset Requirements

### Video Requirements:
- **Format**: MP4, AVI, or MOV
- **Resolution**: Minimum 640x480
- **FPS**: Minimum 24 FPS (recommended 30 FPS)
- **Duration**: 1-5 seconds per gesture
- **Quality**: Clear hand visibility, good lighting

### Content Requirements:
- **Hands**: Both hands visible when applicable
- **Background**: Clean, uncluttered background
- **Lighting**: Even lighting, no harsh shadows
- **Pose**: Upper body visible (shoulders to hands)

### Minimum Dataset Size:
- **Gestures**: At least 10 different gestures
- **Subjects**: At least 3 different subjects
- **Videos per gesture**: At least 10 videos per gesture
- **Total videos**: Minimum 100 videos recommended

## Step 5: Validate Dataset Structure

Before preprocessing, validate the dataset structure:

```bash
python ml/datasets/dataset_validator.py \
    --dataset-root ml/datasets/raw/INCLUDE \
    --metadata-path ml/datasets/raw/metadata.csv
```

This will check:
- Video file existence
- Metadata file format
- Video properties (duration, FPS, resolution)
- Missing or corrupted files

## Step 6: Preprocess the Dataset

Once the dataset is organized and validated, run the preprocessing pipeline:

```bash
python ml/datasets/preprocessing_pipeline.py \
    --config ml/experiments/configs/preprocessing_config.yaml
```

Or use the INCLUDE-specific preprocessing script:

```bash
python ml/datasets/preprocess_include.py \
    --dataset-root ml/datasets/raw/INCLUDE \
    --output-root ml/datasets/processed \
    --metadata-path ml/datasets/raw/metadata.csv
```

## Step 7: Verify Preprocessed Data

After preprocessing, verify the output:

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

## Troubleshooting

### Issue: Videos not found
**Solution**: Check that video paths in metadata.csv are correct relative to dataset_root

### Issue: MediaPipe extraction fails
**Solution**: Ensure videos have visible hands and good lighting. Check video resolution and format.

### Issue: Not enough sequences generated
**Solution**: Increase dataset size or adjust sequence generation parameters (stride, sequence_length).

### Issue: Memory errors during preprocessing
**Solution**: Reduce batch_size or process videos in smaller batches.

## Next Steps

After preprocessing:
1. Review the preprocessing statistics
2. Check quality reports in `ml/datasets/processed/quality_reports/`
3. Proceed to model training using the training script

## Additional Notes

- The preprocessing pipeline uses MediaPipe for landmark extraction
- Landmarks are extracted for hands, face, and pose
- Sequences are generated with configurable length and stride
- Subject-based splitting ensures no data leakage between train/val/test
- Data augmentation can be applied during sequence generation
