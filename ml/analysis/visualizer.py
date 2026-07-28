"""
Visualization Module for INCLUDE Dataset Analysis.

Generates figures and plots for class distribution, imbalance, video durations,
frame counts, missing landmarks, and overall dataset quality summary.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CLI compatibility
import matplotlib.pyplot as plt
import numpy as np

from ml.analysis.dataset_analyzer import (
    ClassDistributionMetrics,
    DurationMetrics,
    FrameCountMetrics,
    MissingLandmarkMetrics,
    QualityReportMetrics,
)

logger = logging.getLogger(__name__)

# Style setup for clean visuals
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')


class DatasetVisualizer:
    """
    Generates dataset analysis charts and visual dashboards.
    """

    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize DatasetVisualizer.

        Args:
            output_dir: Path to directory where plots will be saved.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_class_distribution(
        self,
        metrics: ClassDistributionMetrics,
        top_k: int = 25,
        filename: str = "class_distribution.png"
    ) -> Path:
        """
        Plot class sample counts for top-K classes.

        Args:
            metrics: ClassDistributionMetrics object
            top_k: Top K classes to display
            filename: Output filename

        Returns:
            Path to saved plot file
        """
        class_counts = metrics.class_counts
        items = list(class_counts.items())[:top_k]
        labels = [item[0] for item in items]
        counts = [item[1] for item in items]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(labels, counts, color='#3498db', edgecolor='#2980b9', alpha=0.85)

        ax.set_title(f'Sample Count per Class (Top {top_k} Classes)', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Gesture Class', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=9)

        # Add data labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        ax.axhline(y=metrics.mean_samples_per_class, color='#e74c3c', linestyle='--', linewidth=1.5,
                   label=f'Mean ({metrics.mean_samples_per_class:.1f})')
        ax.legend(loc='upper right')

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300)
        plt.close()

        logger.info(f"Saved class distribution plot to {save_path}")
        return save_path

    def plot_class_imbalance(
        self,
        metrics: ClassDistributionMetrics,
        filename: str = "class_imbalance.png"
    ) -> Path:
        """
        Plot Pareto / cumulative class imbalance curve.

        Args:
            metrics: ClassDistributionMetrics object
            filename: Output filename

        Returns:
            Path to saved plot file
        """
        counts = sorted(list(metrics.class_counts.values()), reverse=True)
        total_samples = sum(counts)
        cum_pct = (np.cumsum(counts) / max(1, total_samples)) * 100.0
        ranks = np.arange(1, len(counts) + 1)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = '#2ecc71'
        ax1.set_xlabel('Class Rank (Sorted by Frequency)', fontsize=12)
        ax1.set_ylabel('Samples per Class', color='#2980b9', fontsize=12)
        ax1.bar(ranks, counts, color='#3498db', alpha=0.6, label='Class Sample Count')
        ax1.tick_params(axis='y', labelcolor='#2980b9')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Cumulative Percentage (%)', color='#27ae60', fontsize=12)
        ax2.plot(ranks, cum_pct, color=color, linewidth=2.5, label='Cumulative %')
        ax2.tick_params(axis='y', labelcolor='#27ae60')

        plt.title(f'Class Imbalance Pareto Curve (Gini: {metrics.gini_coefficient:.2f}, Imbalance Ratio: {metrics.imbalance_ratio:.1f}x)',
                  fontsize=13, fontweight='bold', pad=15)

        fig.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300)
        plt.close()

        logger.info(f"Saved class imbalance plot to {save_path}")
        return save_path

    def plot_duration_distribution(
        self,
        durations: list,
        metrics: DurationMetrics,
        filename: str = "duration_distribution.png"
    ) -> Path:
        """
        Plot video duration distribution histogram and boxplot.

        Args:
            durations: List of video duration values in seconds
            metrics: DurationMetrics object
            filename: Output filename

        Returns:
            Path to saved plot file
        """
        fig, (ax_box, ax_hist) = plt.subplots(
            2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [0.2, 0.8]}
        )

        ax_box.boxplot(durations, vert=False, patch_artist=True,
                       boxprops=dict(facecolor='#9b59b6', color='#8e44ad', alpha=0.7),
                       medianprops=dict(color='#e74c3c', linewidth=2))
        ax_box.set_yticks([])
        ax_box.set_title('Video Duration Distribution (seconds)', fontsize=14, fontweight='bold', pad=10)

        n, bins, patches = ax_hist.hist(durations, bins=25, color='#9b59b6', edgecolor='#8e44ad', alpha=0.75)
        ax_hist.set_xlabel('Duration (seconds)', fontsize=12)
        ax_hist.set_ylabel('Number of Videos', fontsize=12)

        ax_hist.axvline(metrics.mean_duration, color='#e74c3c', linestyle='--', linewidth=1.5,
                        label=f'Mean ({metrics.mean_duration:.2f}s)')
        ax_hist.axvline(metrics.median_duration, color='#2ecc71', linestyle='-', linewidth=1.5,
                        label=f'Median ({metrics.median_duration:.2f}s)')
        ax_hist.legend(loc='upper right')

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300)
        plt.close()

        logger.info(f"Saved duration distribution plot to {save_path}")
        return save_path

    def plot_frame_count_distribution(
        self,
        frame_counts: list,
        metrics: FrameCountMetrics,
        filename: str = "frame_count_distribution.png"
    ) -> Path:
        """
        Plot frame count distribution histogram.

        Args:
            frame_counts: List of frame count values
            metrics: FrameCountMetrics object
            filename: Output filename

        Returns:
            Path to saved plot file
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        n, bins, patches = ax.hist(frame_counts, bins=25, color='#1abc9c', edgecolor='#16a085', alpha=0.75)
        ax.set_title('Sequence Frame Count Distribution', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Frame Count per Sequence', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)

        ax.axvline(metrics.mean_frames, color='#e74c3c', linestyle='--', linewidth=1.5,
                   label=f'Mean ({metrics.mean_frames:.1f} frames)')
        ax.axvline(metrics.median_frames, color='#2ecc71', linestyle='-', linewidth=1.5,
                   label=f'Median ({metrics.median_frames:.1f} frames)')
        ax.legend(loc='upper right')

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300)
        plt.close()

        logger.info(f"Saved frame count distribution plot to {save_path}")
        return save_path

    def plot_missing_landmarks(
        self,
        metrics: MissingLandmarkMetrics,
        filename: str = "missing_landmarks.png"
    ) -> Path:
        """
        Plot missing landmark percentages across modalities.

        Args:
            metrics: MissingLandmarkMetrics object
            filename: Output filename

        Returns:
            Path to saved plot file
        """
        categories = ['Left Hand', 'Right Hand', 'Both Hands', 'Face', 'Pose']
        percentages = [
            metrics.pct_missing_left_hand,
            metrics.pct_missing_right_hand,
            metrics.pct_missing_both_hands,
            metrics.pct_missing_face,
            metrics.pct_missing_pose
        ]

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ['#e74c3c', '#e67e22', '#c0392b', '#f1c40f', '#3498db']
        bars = ax.bar(categories, percentages, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)

        ax.set_title('Missing Landmark Frame Percentages by Modality', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('Missing Frames (%)', fontsize=12)
        ax.set_ylim(0, max(100, max(percentages) + 10 if percentages else 100))

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300)
        plt.close()

        logger.info(f"Saved missing landmarks plot to {save_path}")
        return save_path

    def plot_dataset_quality_summary(
        self,
        quality: QualityReportMetrics,
        cls_metrics: ClassDistributionMetrics,
        filename: str = "dataset_quality_summary.png"
    ) -> Path:
        """
        Generate a multi-panel dashboard summarizing overall dataset quality.

        Args:
            quality: QualityReportMetrics object
            cls_metrics: ClassDistributionMetrics object
            filename: Output filename

        Returns:
            Path to saved summary plot file
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Panel 1: Valid vs Invalid Samples
        labels_v = ['Valid Samples', 'Invalid / Defective']
        sizes_v = [quality.valid_videos, max(0, quality.total_videos - quality.valid_videos)]
        colors_v = ['#2ecc71', '#e74c3c']
        axes[0, 0].pie(sizes_v, labels=labels_v, autopct='%1.1f%%', colors=colors_v, startangle=140,
                       explode=(0.05, 0))
        axes[0, 0].set_title('Sample Validity Ratio', fontsize=12, fontweight='bold')

        # Panel 2: Resolutions breakdown
        if quality.resolution_counts:
            res_labels = list(quality.resolution_counts.keys())
            res_vals = list(quality.resolution_counts.values())
            axes[0, 1].bar(res_labels, res_vals, color='#34495e', alpha=0.8)
            axes[0, 1].set_title('Video Resolutions Distribution', fontsize=12, fontweight='bold')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].tick_params(axis='x', rotation=30)
        else:
            axes[0, 1].text(0.5, 0.5, 'N/A', ha='center', va='center')
            axes[0, 1].set_title('Video Resolutions Distribution', fontsize=12, fontweight='bold')

        # Panel 3: Key metrics summary text
        axes[1, 0].axis('off')
        summary_text = (
            f"--- DATASET HEALTH DASHBOARD ---\n\n"
            f"• Total Samples: {cls_metrics.total_samples}\n"
            f"• Total Classes: {cls_metrics.total_classes}\n"
            f"• Mean Samples / Class: {cls_metrics.mean_samples_per_class:.1f}\n"
            f"• Imbalance Ratio (Max/Min): {cls_metrics.imbalance_ratio:.1f}x\n"
            f"• Gini Inequality Index: {cls_metrics.gini_coefficient:.2f}\n"
            f"• Valid Video Ratio: {quality.valid_ratio:.1f}%\n"
        )
        axes[1, 0].text(0.1, 0.2, summary_text, fontsize=11, family='monospace',
                        bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

        # Panel 4: Quality grades
        if quality.quality_grade_counts:
            q_labels = list(quality.quality_grade_counts.keys())
            q_vals = list(quality.quality_grade_counts.values())
            axes[1, 1].bar(q_labels, q_vals, color='#f39c12', alpha=0.8)
            axes[1, 1].set_title('Quality Grade Distribution', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Count')
        else:
            axes[1, 1].text(0.5, 0.5, 'N/A', ha='center', va='center')
            axes[1, 1].set_title('Quality Grade Distribution', fontsize=12, fontweight='bold')

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300)
        plt.close()

        logger.info(f"Saved dataset quality summary dashboard to {save_path}")
        return save_path
