"""
Dataset Analyzer Module for INCLUDE / ISL Recognition Dataset.

Analyzes raw dataset metadata, video files, and extracted landmark sequence files
to report sample distributions, class imbalance, duration stats, frame counts,
missing landmark occurrences, and quality metrics.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClassDistributionMetrics:
    """Class distribution statistics."""
    total_classes: int
    total_samples: int
    min_samples_per_class: int
    max_samples_per_class: int
    mean_samples_per_class: float
    median_samples_per_class: float
    std_samples_per_class: float
    imbalance_ratio: float  # Max / Min ratio
    gini_coefficient: float
    class_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class DurationMetrics:
    """Video duration distribution statistics in seconds."""
    mean_duration: float
    std_duration: float
    median_duration: float
    min_duration: float
    max_duration: float
    p25_duration: float
    p75_duration: float
    total_duration_hours: float


@dataclass
class FrameCountMetrics:
    """Frame count distribution statistics."""
    mean_frames: float
    std_frames: float
    median_frames: float
    min_frames: int
    max_frames: int
    p25_frames: float
    p75_frames: float
    total_frames: int


@dataclass
class MissingLandmarkMetrics:
    """Missing landmark statistics across extracted sequences."""
    total_frames_analyzed: int
    missing_left_hand_frames: int
    missing_right_hand_frames: int
    missing_both_hands_frames: int
    missing_face_frames: int
    missing_pose_frames: int
    pct_missing_left_hand: float
    pct_missing_right_hand: float
    pct_missing_both_hands: float
    pct_missing_face: float
    pct_missing_pose: float


@dataclass
class QualityReportMetrics:
    """Overall dataset quality assessment metrics."""
    total_videos: int
    valid_videos: int
    invalid_videos: int
    valid_ratio: float
    resolution_counts: Dict[str, int]
    fps_counts: Dict[str, float]
    quality_grade_counts: Dict[str, int]
    issues_found: List[str]


class DatasetAnalyzer:
    """
    Comprehensive dataset analyzer for ISL / INCLUDE datasets.
    """

    def __init__(
        self,
        metadata_csv: Optional[Union[str, Path]] = None,
        processed_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize DatasetAnalyzer.

        Args:
            metadata_csv: Path to dataset metadata CSV file
            processed_dir: Path to processed landmark directory (.npy / .npz)
        """
        self.metadata_csv = Path(metadata_csv) if metadata_csv else None
        self.processed_dir = Path(processed_dir) if processed_dir else None
        
        self.samples: List[Dict] = []
        self.landmark_samples: List[Dict] = []
        
        if self.metadata_csv and self.metadata_csv.exists():
            self.load_metadata_csv(self.metadata_csv)
            
        if self.processed_dir and self.processed_dir.exists():
            self.load_processed_landmarks(self.processed_dir)

    def load_metadata_csv(self, csv_path: Union[str, Path]) -> int:
        """
        Load video metadata from CSV file.

        Args:
            csv_path: Path to CSV metadata file

        Returns:
            Number of loaded samples
        """
        csv_path = Path(csv_path)
        self.samples = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample = {
                    'video_id': row.get('video_id', ''),
                    'gesture': row.get('gesture', row.get('label', 'unknown')),
                    'subject': row.get('subject', 'unknown'),
                    'video_path': row.get('video_path', ''),
                    'duration': float(row['duration']) if 'duration' in row and row['duration'] else 0.0,
                    'frame_count': int(row['frame_count']) if 'frame_count' in row and row['frame_count'] else 0,
                    'fps': float(row['fps']) if 'fps' in row and row['fps'] else 30.0,
                    'resolution': row.get('resolution', '1920x1080'),
                    'quality': row.get('quality', 'medium'),
                    'lighting_condition': row.get('lighting_condition', 'normal'),
                    'background': row.get('background', 'indoor'),
                }
                self.samples.append(sample)

        logger.info(f"Loaded {len(self.samples)} sample metadata entries from {csv_path}")
        return len(self.samples)

    def load_processed_landmarks(self, processed_dir: Union[str, Path]) -> int:
        """
        Load processed landmark files (.npy / .npz) from processed directory.

        Args:
            processed_dir: Path to directory containing processed landmark sequences

        Returns:
            Number of landmark files loaded
        """
        processed_dir = Path(processed_dir)
        self.landmark_samples = []

        npz_files = list(processed_dir.rglob('*.npz'))
        npy_files = list(processed_dir.rglob('*.npy'))

        for file_path in npz_files:
            try:
                data = np.load(file_path, allow_pickle=True)
                features = data['features'] if 'features' in data else None
                label = str(data['label']) if 'label' in data else file_path.parent.name
                
                if features is not None:
                    self.landmark_samples.append({
                        'file_path': str(file_path),
                        'label': label,
                        'shape': features.shape,
                        'num_frames': features.shape[0],
                        'feature_dim': features.shape[1] if features.ndim > 1 else 0,
                        'features': features
                    })
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")

        for file_path in npy_files:
            if file_path.name == 'sequences.npy' or file_path.name == 'labels.npy':
                continue
            try:
                features = np.load(file_path, allow_pickle=True)
                label = file_path.parent.name if file_path.parent.name not in ['train', 'val', 'test', 'landmarks'] else file_path.stem.split('_')[0]
                if features.ndim >= 2:
                    self.landmark_samples.append({
                        'file_path': str(file_path),
                        'label': label,
                        'shape': features.shape,
                        'num_frames': features.shape[0],
                        'feature_dim': features.shape[1] if features.ndim > 1 else 0,
                        'features': features
                    })
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")

        logger.info(f"Loaded {len(self.landmark_samples)} landmark sample files from {processed_dir}")
        return len(self.landmark_samples)

    def analyze_class_distribution(self) -> ClassDistributionMetrics:
        """
        Analyze class distribution and compute imbalance metrics.

        Returns:
            ClassDistributionMetrics object
        """
        if not self.samples and not self.landmark_samples:
            raise ValueError("No metadata or landmark samples loaded for analysis.")

        # Aggregate class counts from metadata or landmark files
        class_counts: Dict[str, int] = {}
        source = self.samples if self.samples else self.landmark_samples
        
        for sample in source:
            label = sample.get('gesture', sample.get('label', 'unknown'))
            class_counts[label] = class_counts.get(label, 0) + 1

        counts = np.array(list(class_counts.values()), dtype=np.float64)
        total_samples = int(np.sum(counts))
        total_classes = len(counts)

        min_cnt = int(np.min(counts)) if total_classes > 0 else 0
        max_cnt = int(np.max(counts)) if total_classes > 0 else 0
        mean_cnt = float(np.mean(counts)) if total_classes > 0 else 0.0
        median_cnt = float(np.median(counts)) if total_classes > 0 else 0.0
        std_cnt = float(np.std(counts)) if total_classes > 0 else 0.0

        imbalance_ratio = (max_cnt / min_cnt) if min_cnt > 0 else float('inf')
        gini = self._compute_gini_coefficient(counts)

        return ClassDistributionMetrics(
            total_classes=total_classes,
            total_samples=total_samples,
            min_samples_per_class=min_cnt,
            max_samples_per_class=max_cnt,
            mean_samples_per_class=mean_cnt,
            median_samples_per_class=median_cnt,
            std_samples_per_class=std_cnt,
            imbalance_ratio=imbalance_ratio,
            gini_coefficient=gini,
            class_counts=dict(sorted(class_counts.items(), key=lambda x: x[1], reverse=True))
        )

    def analyze_video_durations(self) -> DurationMetrics:
        """
        Analyze video duration statistics.

        Returns:
            DurationMetrics object
        """
        durations = [s['duration'] for s in self.samples if s.get('duration', 0) > 0]

        if not durations and self.landmark_samples:
            # Estimate duration assuming 30 fps
            durations = [s['num_frames'] / 30.0 for s in self.landmark_samples]

        if not durations:
            return DurationMetrics(0, 0, 0, 0, 0, 0, 0, 0)

        arr = np.array(durations, dtype=np.float64)
        return DurationMetrics(
            mean_duration=float(np.mean(arr)),
            std_duration=float(np.std(arr)),
            median_duration=float(np.median(arr)),
            min_duration=float(np.min(arr)),
            max_duration=float(np.max(arr)),
            p25_duration=float(np.percentile(arr, 25)),
            p75_duration=float(np.percentile(arr, 75)),
            total_duration_hours=float(np.sum(arr) / 3600.0)
        )

    def analyze_frame_counts(self) -> FrameCountMetrics:
        """
        Analyze frame count distribution.

        Returns:
            FrameCountMetrics object
        """
        counts = [s['frame_count'] for s in self.samples if s.get('frame_count', 0) > 0]

        if not counts and self.landmark_samples:
            counts = [s['num_frames'] for s in self.landmark_samples]

        if not counts:
            return FrameCountMetrics(0, 0, 0, 0, 0, 0, 0, 0)

        arr = np.array(counts, dtype=np.float64)
        return FrameCountMetrics(
            mean_frames=float(np.mean(arr)),
            std_frames=float(np.std(arr)),
            median_frames=float(np.median(arr)),
            min_frames=int(np.min(arr)),
            max_frames=int(np.max(arr)),
            p25_frames=float(np.percentile(arr, 25)),
            p75_frames=float(np.percentile(arr, 75)),
            total_frames=int(np.sum(arr))
        )

    def analyze_missing_landmarks(self) -> MissingLandmarkMetrics:
        """
        Analyze missing landmark statistics from loaded landmark feature samples.

        Returns:
            MissingLandmarkMetrics object
        """
        if not self.landmark_samples:
            return MissingLandmarkMetrics(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

        total_frames = 0
        missing_lh = 0
        missing_rh = 0
        missing_both_h = 0
        missing_face = 0
        missing_pose = 0

        for sample in self.landmark_samples:
            feats = sample['features']  # Shape: (T, D)
            num_frames = feats.shape[0]
            feature_dim = feats.shape[1] if feats.ndim > 1 else 0
            total_frames += num_frames

            # Feature layout for default 279-dim setup:
            # [0:63] Left Hand, [63:126] Right Hand, [126:246] Face, [246:279] Pose
            if feature_dim >= 126:
                lh = feats[:, 0:63]
                rh = feats[:, 63:126]

                lh_missing_mask = np.all(lh == 0, axis=1)
                rh_missing_mask = np.all(rh == 0, axis=1)

                missing_lh += int(np.sum(lh_missing_mask))
                missing_rh += int(np.sum(rh_missing_mask))
                missing_both_h += int(np.sum(lh_missing_mask & rh_missing_mask))

            if feature_dim >= 246:
                face = feats[:, 126:246]
                missing_face += int(np.sum(np.all(face == 0, axis=1)))

            if feature_dim >= 279:
                pose = feats[:, 246:279]
                missing_pose += int(np.sum(np.all(pose == 0, axis=1)))

        denom = max(1, total_frames)
        return MissingLandmarkMetrics(
            total_frames_analyzed=total_frames,
            missing_left_hand_frames=missing_lh,
            missing_right_hand_frames=missing_rh,
            missing_both_hands_frames=missing_both_h,
            missing_face_frames=missing_face,
            missing_pose_frames=missing_pose,
            pct_missing_left_hand=(missing_lh / denom) * 100.0,
            pct_missing_right_hand=(missing_rh / denom) * 100.0,
            pct_missing_both_hands=(missing_both_h / denom) * 100.0,
            pct_missing_face=(missing_face / denom) * 100.0,
            pct_missing_pose=(missing_pose / denom) * 100.0
        )

    def generate_quality_report(self) -> QualityReportMetrics:
        """
        Generate overall dataset quality assessment report.

        Returns:
            QualityReportMetrics object
        """
        total_videos = len(self.samples) if self.samples else len(self.landmark_samples)
        valid_videos = 0
        invalid_videos = 0
        resolutions: Dict[str, int] = {}
        fps_dist: Dict[str, float] = {}
        qualities: Dict[str, int] = {}
        issues = []

        for sample in self.samples:
            res = sample.get('resolution', 'unknown')
            resolutions[res] = resolutions.get(res, 0) + 1

            fps_val = str(sample.get('fps', 30.0))
            fps_dist[fps_val] = fps_dist.get(fps_val, 0) + 1

            q = sample.get('quality', 'medium')
            qualities[q] = qualities.get(q, 0) + 1

            dur = sample.get('duration', 0.0)
            fc = sample.get('frame_count', 0)

            if dur > 0.5 and fc >= 10:
                valid_videos += 1
            else:
                invalid_videos += 1
                issues.append(f"Sample {sample.get('video_id', 'unknown')} too short (duration={dur}s, frames={fc})")

        if not self.samples and self.landmark_samples:
            valid_videos = len(self.landmark_samples)

        valid_ratio = (valid_videos / max(1, total_videos)) * 100.0

        if invalid_videos > 0:
            issues.append(f"{invalid_videos} samples failed minimum duration/frame count criteria")

        return QualityReportMetrics(
            total_videos=total_videos,
            valid_videos=valid_videos,
            invalid_videos=invalid_videos,
            valid_ratio=valid_ratio,
            resolution_counts=resolutions,
            fps_counts=fps_dist,
            quality_grade_counts=qualities,
            issues_found=issues
        )

    @staticmethod
    def _compute_gini_coefficient(array: np.ndarray) -> float:
        """Compute Gini coefficient of array values to quantify class inequality."""
        if len(array) == 0 or np.all(array == 0):
            return 0.0
        sorted_arr = np.sort(array)
        n = len(array)
        index = np.arange(1, n + 1)
        return float((np.sum((2 * index - n - 1) * sorted_arr)) / (n * np.sum(sorted_arr)))
