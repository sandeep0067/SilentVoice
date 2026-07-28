"""
Dataset Validator for INCLUDE dataset.

Validates dataset integrity, checks for missing files, and ensures data quality.
"""

import json
from pathlib import Path
from typing import Dict, List, Set
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of dataset validation."""
    is_valid: bool
    missing_videos: List[str]
    corrupted_videos: List[str]
    missing_metadata: List[str]
    duplicate_entries: List[str]
    quality_issues: Dict[str, List[str]]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'is_valid': self.is_valid,
            'missing_videos': self.missing_videos,
            'corrupted_videos': self.corrupted_videos,
            'missing_metadata': self.missing_metadata,
            'duplicate_entries': self.duplicate_entries,
            'quality_issues': self.quality_issues
        }


class DatasetValidator:
    """Validates dataset integrity and quality."""
    
    def __init__(self, dataset_root: str, metadata_manager):
        """
        Initialize dataset validator.
        
        Args:
            dataset_root: Root directory of dataset
            metadata_manager: MetadataManager instance
        """
        self.dataset_root = Path(dataset_root)
        self.metadata_manager = metadata_manager
    
    def validate_all(self) -> ValidationResult:
        """
        Run all validation checks.
        
        Returns:
            ValidationResult with all issues
        """
        missing_videos = self._check_missing_videos()
        corrupted_videos = self._check_corrupted_videos()
        missing_metadata = self._check_missing_metadata()
        duplicate_entries = self._check_duplicates()
        quality_issues = self._check_quality()
        
        is_valid = (
            not missing_videos and
            not corrupted_videos and
            not missing_metadata and
            not duplicate_entries and
            not quality_issues
        )
        
        return ValidationResult(
            is_valid=is_valid,
            missing_videos=missing_videos,
            corrupted_videos=corrupted_videos,
            missing_metadata=missing_metadata,
            duplicate_entries=duplicate_entries,
            quality_issues=quality_issues
        )
    
    def _check_missing_videos(self) -> List[str]:
        """Check for videos listed in metadata but missing from disk."""
        missing = []
        
        for video_id, metadata in self.metadata_manager.metadata.items():
            video_path = Path(metadata.video_path)
            if not video_path.exists():
                missing.append(video_id)
        
        return missing
    
    def _check_corrupted_videos(self) -> List[str]:
        """Check for corrupted video files."""
        import cv2
        
        corrupted = []
        
        for video_id, metadata in self.metadata_manager.metadata.items():
            video_path = Path(metadata.video_path)
            if not video_path.exists():
                continue
            
            try:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    corrupted.append(video_id)
                cap.release()
            except Exception:
                corrupted.append(video_id)
        
        return corrupted
    
    def _check_missing_metadata(self) -> List[str]:
        """Check for videos on disk without metadata entries."""
        video_files = set(self.dataset_root.rglob("*.mp4"))
        video_files.update(self.dataset_root.rglob("*.avi"))
        video_files.update(self.dataset_root.rglob("*.mov"))
        
        metadata_paths = {
            Path(m.video_path) for m in self.metadata_manager.metadata.values()
        }
        
        missing_metadata = [
            str(f.relative_to(self.dataset_root))
            for f in video_files
            if f not in metadata_paths
        ]
        
        return missing_metadata
    
    def _check_duplicates(self) -> List[str]:
        """Check for duplicate video entries in metadata."""
        seen_paths: Set[str] = set()
        duplicates = []
        
        for video_id, metadata in self.metadata_manager.metadata.items():
            if metadata.video_path in seen_paths:
                duplicates.append(video_id)
            seen_paths.add(metadata.video_path)
        
        return duplicates
    
    def _check_quality(self) -> Dict[str, List[str]]:
        """Check for quality issues in dataset."""
        issues = {
            'short_duration': [],
            'long_duration': [],
            'low_fps': [],
            'low_resolution': []
        }
        
        for video_id, metadata in self.metadata_manager.metadata.items():
            if metadata.duration < 1.0:
                issues['short_duration'].append(video_id)
            elif metadata.duration > 5.0:
                issues['long_duration'].append(video_id)
            
            if metadata.fps < 24:
                issues['low_fps'].append(video_id)
            
            if metadata.resolution[0] < 640 or metadata.resolution[1] < 480:
                issues['low_resolution'].append(video_id)
        
        return issues
    
    def generate_report(self, output_path: str) -> None:
        """
        Generate validation report.
        
        Args:
            output_path: Path to output JSON report
        """
        result = self.validate_all()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2)
