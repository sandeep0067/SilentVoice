# INCLUDE Dataset Analysis Report

Executive analysis report of the Indian Sign Language (INCLUDE) dataset for model preparation.

---

## 1. Class Distribution & Imbalance Analysis

- **Total Gesture Classes**: 25
- **Total Samples**: 770
- **Mean Samples per Class**: 30.80
- **Min / Max Samples per Class**: 15 / 45
- **Imbalance Ratio ($N_{max} / N_{min}$)**: 3.00x
- **Gini Coefficient**: 0.1559 (0 = perfectly balanced, 1 = extreme imbalance)

---

## 2. Video Duration Statistics

- **Mean Duration**: 3.00 seconds
- **Median Duration**: 3.02 seconds
- **Std Deviation**: 1.05 seconds
- **Range (Min / Max)**: 1.20s - 4.80s
- **Interquartile Range (P25 - P75)**: 2.11s - 3.90s
- **Total Dataset Video Duration**: 0.641 hours

---

## 3. Frame Count Distribution

- **Mean Frame Count**: 89.86 frames
- **Median Frame Count**: 91.00 frames
- **Std Deviation**: 31.40 frames
- **Range (Min / Max)**: 36 - 144 frames
- **Total Frames Analyzed**: 69193 frames

---

## 4. Missing Landmark Statistics

| Modality | Missing Frames Count | Percentage Missing | Status |
| :--- | :--- | :--- | :--- |
| **Left Hand** | 0 | 0.00% | NORMAL |
| **Right Hand** | 0 | 0.00% | NORMAL |
| **Both Hands** | 0 | 0.00% | GOOD |
| **Face** | 0 | 0.00% | EXCELLENT |
| **Pose** | 0 | 0.00% | EXCELLENT |

---

## 5. Dataset Quality Report

- **Total Videos Assessed**: 770
- **Valid Video Ratio**: 100.00%
- **Valid / Invalid Count**: 770 / 0

### Recommendations for Preprocessing & Training
1. **Handling Class Imbalance**: Apply focal loss, class-weighted cross entropy, or sequence data augmentation for minority classes.
2. **Missing Landmarks**: Use temporal spline / linear interpolation for isolated missing frames.
3. **Sequence Length**: Pad / truncate sequence length to a uniform fixed target (e.g., $T=30$ frames).
