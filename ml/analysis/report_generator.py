"""
Report Generator Module for INCLUDE Dataset Analysis.

Exports dataset analysis metrics into CSV spreadsheets and Markdown reports.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, Optional, Union

from ml.analysis.dataset_analyzer import (
    ClassDistributionMetrics,
    DurationMetrics,
    FrameCountMetrics,
    MissingLandmarkMetrics,
    QualityReportMetrics,
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Exports CSV files and Markdown reports for dataset analysis.
    """

    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize ReportGenerator.

        Args:
            output_dir: Output directory path where CSVs and markdown reports will be written.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_class_distribution_csv(
        self,
        metrics: ClassDistributionMetrics,
        filename: str = "class_distribution.csv"
    ) -> Path:
        """
        Export class sample counts to CSV file.

        Args:
            metrics: ClassDistributionMetrics object
            filename: CSV output filename

        Returns:
            Path to exported CSV file
        """
        output_path = self.output_dir / filename
        total = max(1, metrics.total_samples)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Class_Name', 'Sample_Count', 'Percentage_Share', 'Imbalance_Weight'])
            
            for class_name, count in metrics.class_counts.items():
                pct = (count / total) * 100.0
                # Imbalance weight relative to uniform mean count
                weight = metrics.mean_samples_per_class / max(1, count)
                writer.writerow([class_name, count, f"{pct:.2f}%", f"{weight:.3f}"])

        logger.info(f"Exported class distribution CSV to {output_path}")
        return output_path

    def export_class_imbalance_summary_csv(
        self,
        metrics: ClassDistributionMetrics,
        filename: str = "class_imbalance_summary.csv"
    ) -> Path:
        """
        Export overall class imbalance summary statistics to CSV file.

        Args:
            metrics: ClassDistributionMetrics object
            filename: CSV output filename

        Returns:
            Path to exported CSV file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Classes', metrics.total_classes])
            writer.writerow(['Total Samples', metrics.total_samples])
            writer.writerow(['Min Samples per Class', metrics.min_samples_per_class])
            writer.writerow(['Max Samples per Class', metrics.max_samples_per_class])
            writer.writerow(['Mean Samples per Class', f"{metrics.mean_samples_per_class:.2f}"])
            writer.writerow(['Median Samples per Class', f"{metrics.median_samples_per_class:.2f}"])
            writer.writerow(['Std Dev Samples per Class', f"{metrics.std_samples_per_class:.2f}"])
            writer.writerow(['Imbalance Ratio (Max/Min)', f"{metrics.imbalance_ratio:.2f}"])
            writer.writerow(['Gini Inequality Coefficient', f"{metrics.gini_coefficient:.4f}"])

        logger.info(f"Exported class imbalance summary CSV to {output_path}")
        return output_path

    def export_duration_stats_csv(
        self,
        metrics: DurationMetrics,
        filename: str = "video_duration_stats.csv"
    ) -> Path:
        """
        Export video duration statistics to CSV file.

        Args:
            metrics: DurationMetrics object
            filename: CSV output filename

        Returns:
            Path to exported CSV file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Duration_Seconds'])
            writer.writerow(['Mean Duration', f"{metrics.mean_duration:.2f}"])
            writer.writerow(['Std Duration', f"{metrics.std_duration:.2f}"])
            writer.writerow(['Median Duration', f"{metrics.median_duration:.2f}"])
            writer.writerow(['Min Duration', f"{metrics.min_duration:.2f}"])
            writer.writerow(['Max Duration', f"{metrics.max_duration:.2f}"])
            writer.writerow(['25th Percentile', f"{metrics.p25_duration:.2f}"])
            writer.writerow(['75th Percentile', f"{metrics.p75_duration:.2f}"])
            writer.writerow(['Total Hours', f"{metrics.total_duration_hours:.3f}"])

        logger.info(f"Exported video duration stats CSV to {output_path}")
        return output_path

    def export_frame_count_stats_csv(
        self,
        metrics: FrameCountMetrics,
        filename: str = "frame_count_stats.csv"
    ) -> Path:
        """
        Export frame count statistics to CSV file.

        Args:
            metrics: FrameCountMetrics object
            filename: CSV output filename

        Returns:
            Path to exported CSV file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Frame_Count'])
            writer.writerow(['Mean Frames', f"{metrics.mean_frames:.2f}"])
            writer.writerow(['Std Frames', f"{metrics.std_frames:.2f}"])
            writer.writerow(['Median Frames', f"{metrics.median_frames:.2f}"])
            writer.writerow(['Min Frames', metrics.min_frames])
            writer.writerow(['Max Frames', metrics.max_frames])
            writer.writerow(['25th Percentile', f"{metrics.p25_frames:.2f}"])
            writer.writerow(['75th Percentile', f"{metrics.p75_frames:.2f}"])
            writer.writerow(['Total Frames', metrics.total_frames])

        logger.info(f"Exported frame count stats CSV to {output_path}")
        return output_path

    def export_missing_landmarks_summary_csv(
        self,
        metrics: MissingLandmarkMetrics,
        filename: str = "missing_landmarks_summary.csv"
    ) -> Path:
        """
        Export missing landmark statistics to CSV file.

        Args:
            metrics: MissingLandmarkMetrics object
            filename: CSV output filename

        Returns:
            Path to exported CSV file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Modality', 'Missing_Frame_Count', 'Missing_Percentage', 'Total_Frames'])
            writer.writerow(['Left Hand', metrics.missing_left_hand_frames, f"{metrics.pct_missing_left_hand:.2f}%", metrics.total_frames_analyzed])
            writer.writerow(['Right Hand', metrics.missing_right_hand_frames, f"{metrics.pct_missing_right_hand:.2f}%", metrics.total_frames_analyzed])
            writer.writerow(['Both Hands', metrics.missing_both_hands_frames, f"{metrics.pct_missing_both_hands:.2f}%", metrics.total_frames_analyzed])
            writer.writerow(['Face', metrics.missing_face_frames, f"{metrics.pct_missing_face:.2f}%", metrics.total_frames_analyzed])
            writer.writerow(['Pose', metrics.missing_pose_frames, f"{metrics.pct_missing_pose:.2f}%", metrics.total_frames_analyzed])

        logger.info(f"Exported missing landmarks summary CSV to {output_path}")
        return output_path

    def export_dataset_quality_report_csv(
        self,
        quality: QualityReportMetrics,
        filename: str = "dataset_quality_report.csv"
    ) -> Path:
        """
        Export dataset quality audit report to CSV file.

        Args:
            quality: QualityReportMetrics object
            filename: CSV output filename

        Returns:
            Path to exported CSV file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Category', 'Metric', 'Value', 'Status'])
            writer.writerow(['Sample Integrity', 'Total Videos', quality.total_videos, 'INFO'])
            writer.writerow(['Sample Integrity', 'Valid Videos', quality.valid_videos, 'PASS'])
            writer.writerow(['Sample Integrity', 'Invalid Videos', quality.invalid_videos, 'WARN' if quality.invalid_videos > 0 else 'PASS'])
            writer.writerow(['Sample Integrity', 'Valid Ratio', f"{quality.valid_ratio:.2f}%", 'PASS' if quality.valid_ratio >= 90 else 'WARN'])

            for res, cnt in quality.resolution_counts.items():
                writer.writerow(['Resolution', f'Resolution_{res}', cnt, 'INFO'])

            for fps, cnt in quality.fps_counts.items():
                writer.writerow(['FPS', f'FPS_{fps}', cnt, 'INFO'])

            for q_grade, cnt in quality.quality_grade_counts.items():
                writer.writerow(['Quality Grade', f'Grade_{q_grade}', cnt, 'INFO'])

            for issue in quality.issues_found:
                writer.writerow(['Issue Log', 'Defect Found', issue, 'WARNING'])

        logger.info(f"Exported dataset quality report CSV to {output_path}")
        return output_path

    def generate_markdown_report(
        self,
        cls_metrics: ClassDistributionMetrics,
        dur_metrics: DurationMetrics,
        frame_metrics: FrameCountMetrics,
        missing_metrics: MissingLandmarkMetrics,
        quality_metrics: QualityReportMetrics,
        filename: str = "dataset_analysis_report.md"
    ) -> Path:
        """
        Generate executive markdown analysis report.

        Args:
            cls_metrics: Class distribution metrics
            dur_metrics: Duration metrics
            frame_metrics: Frame count metrics
            missing_metrics: Missing landmark metrics
            quality_metrics: Quality report metrics
            filename: Markdown filename

        Returns:
            Path to generated Markdown report
        """
        output_path = self.output_dir / filename

        content = f"""# INCLUDE Dataset Analysis Report

Executive analysis report of the Indian Sign Language (INCLUDE) dataset for model preparation.

---

## 1. Class Distribution & Imbalance Analysis

- **Total Gesture Classes**: {cls_metrics.total_classes}
- **Total Samples**: {cls_metrics.total_samples}
- **Mean Samples per Class**: {cls_metrics.mean_samples_per_class:.2f}
- **Min / Max Samples per Class**: {cls_metrics.min_samples_per_class} / {cls_metrics.max_samples_per_class}
- **Imbalance Ratio ($N_{{max}} / N_{{min}}$)**: {cls_metrics.imbalance_ratio:.2f}x
- **Gini Coefficient**: {cls_metrics.gini_coefficient:.4f} (0 = perfectly balanced, 1 = extreme imbalance)

---

## 2. Video Duration Statistics

- **Mean Duration**: {dur_metrics.mean_duration:.2f} seconds
- **Median Duration**: {dur_metrics.median_duration:.2f} seconds
- **Std Deviation**: {dur_metrics.std_duration:.2f} seconds
- **Range (Min / Max)**: {dur_metrics.min_duration:.2f}s - {dur_metrics.max_duration:.2f}s
- **Interquartile Range (P25 - P75)**: {dur_metrics.p25_duration:.2f}s - {dur_metrics.p75_duration:.2f}s
- **Total Dataset Video Duration**: {dur_metrics.total_duration_hours:.3f} hours

---

## 3. Frame Count Distribution

- **Mean Frame Count**: {frame_metrics.mean_frames:.2f} frames
- **Median Frame Count**: {frame_metrics.median_frames:.2f} frames
- **Std Deviation**: {frame_metrics.std_frames:.2f} frames
- **Range (Min / Max)**: {frame_metrics.min_frames} - {frame_metrics.max_frames} frames
- **Total Frames Analyzed**: {frame_metrics.total_frames} frames

---

## 4. Missing Landmark Statistics

| Modality | Missing Frames Count | Percentage Missing | Status |
| :--- | :--- | :--- | :--- |
| **Left Hand** | {missing_metrics.missing_left_hand_frames} | {missing_metrics.pct_missing_left_hand:.2f}% | {'NORMAL' if missing_metrics.pct_missing_left_hand < 30 else 'HIGH'} |
| **Right Hand** | {missing_metrics.missing_right_hand_frames} | {missing_metrics.pct_missing_right_hand:.2f}% | {'NORMAL' if missing_metrics.pct_missing_right_hand < 30 else 'HIGH'} |
| **Both Hands** | {missing_metrics.missing_both_hands_frames} | {missing_metrics.pct_missing_both_hands:.2f}% | {'GOOD' if missing_metrics.pct_missing_both_hands < 10 else 'ALERT'} |
| **Face** | {missing_metrics.missing_face_frames} | {missing_metrics.pct_missing_face:.2f}% | {'EXCELLENT' if missing_metrics.pct_missing_face < 5 else 'CHECK'} |
| **Pose** | {missing_metrics.missing_pose_frames} | {missing_metrics.pct_missing_pose:.2f}% | {'EXCELLENT' if missing_metrics.pct_missing_pose < 5 else 'CHECK'} |

---

## 5. Dataset Quality Report

- **Total Videos Assessed**: {quality_metrics.total_videos}
- **Valid Video Ratio**: {quality_metrics.valid_ratio:.2f}%
- **Valid / Invalid Count**: {quality_metrics.valid_videos} / {quality_metrics.invalid_videos}

### Recommendations for Preprocessing & Training
1. **Handling Class Imbalance**: Apply focal loss, class-weighted cross entropy, or sequence data augmentation for minority classes.
2. **Missing Landmarks**: Use temporal spline / linear interpolation for isolated missing frames.
3. **Sequence Length**: Pad / truncate sequence length to a uniform fixed target (e.g., $T=30$ frames).
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Generated Markdown report at {output_path}")
        return output_path
