"""
Dataset Metadata Manager for INCLUDE dataset.

Manages metadata tracking, validation, and organization for ISL gesture videos.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class VideoMetadata:
    """Metadata for a single video file."""
    video_id: str
    gesture: str
    subject: str
    video_path: str
    duration: float
    frame_count: int
    fps: float
    resolution: tuple
    quality: str
    lighting_condition: str
    background: str
    created_at: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        data['resolution'] = f"{self.resolution[0]}x{self.resolution[1]}"
        return data


class MetadataManager:
    """Manages dataset metadata operations."""
    
    def __init__(self, metadata_path: str):
        """
        Initialize metadata manager.
        
        Args:
            metadata_path: Path to metadata CSV file
        """
        self.metadata_path = Path(metadata_path)
        self.metadata: Dict[str, VideoMetadata] = {}
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load metadata from CSV file."""
        if not self.metadata_path.exists():
            return
        
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                resolution = tuple(map(int, row['resolution'].split('x')))
                metadata = VideoMetadata(
                    video_id=row['video_id'],
                    gesture=row['gesture'],
                    subject=row['subject'],
                    video_path=row['video_path'],
                    duration=float(row['duration']),
                    frame_count=int(row['frame_count']),
                    fps=float(row['fps']),
                    resolution=resolution,
                    quality=row['quality'],
                    lighting_condition=row['lighting_condition'],
                    background=row['background'],
                    created_at=row['created_at']
                )
                self.metadata[metadata.video_id] = metadata
    
    def save_metadata(self) -> None:
        """Save metadata to CSV file."""
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = [
            'video_id', 'gesture', 'subject', 'video_path', 'duration',
            'frame_count', 'fps', 'resolution', 'quality',
            'lighting_condition', 'background', 'created_at'
        ]
        
        with open(self.metadata_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for metadata in self.metadata.values():
                writer.writerow(metadata.to_dict())
    
    def add_video(self, metadata: VideoMetadata) -> None:
        """
        Add video metadata.
        
        Args:
            metadata: Video metadata object
        """
        self.metadata[metadata.video_id] = metadata
        self.save_metadata()
    
    def get_video(self, video_id: str) -> Optional[VideoMetadata]:
        """
        Get video metadata by ID.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Video metadata or None if not found
        """
        return self.metadata.get(video_id)
    
    def get_videos_by_gesture(self, gesture: str) -> List[VideoMetadata]:
        """
        Get all videos for a specific gesture.
        
        Args:
            gesture: Gesture name
            
        Returns:
            List of video metadata
        """
        return [m for m in self.metadata.values() if m.gesture == gesture]
    
    def get_videos_by_subject(self, subject: str) -> List[VideoMetadata]:
        """
        Get all videos for a specific subject.
        
        Args:
            subject: Subject identifier
            
        Returns:
            List of video metadata
        """
        return [m for m in self.metadata.values() if m.subject == subject]
    
    def get_all_gestures(self) -> List[str]:
        """
        Get list of all unique gestures.
        
        Returns:
            List of gesture names
        """
        return sorted(set(m.gesture for m in self.metadata.values()))
    
    def get_all_subjects(self) -> List[str]:
        """
        Get list of all unique subjects.
        
        Returns:
            List of subject identifiers
        """
        return sorted(set(m.subject for m in self.metadata.values()))
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        gestures = self.get_all_gestures()
        subjects = self.get_all_subjects()
        
        gesture_counts = {g: len(self.get_videos_by_gesture(g)) for g in gestures}
        subject_counts = {s: len(self.get_videos_by_subject(s)) for s in subjects}
        
        return {
            'total_videos': len(self.metadata),
            'total_gestures': len(gestures),
            'total_subjects': len(subjects),
            'gesture_counts': gesture_counts,
            'subject_counts': subject_counts,
            'avg_videos_per_gesture': len(self.metadata) / len(gestures) if gestures else 0,
            'avg_videos_per_subject': len(self.metadata) / len(subjects) if subjects else 0
        }
    
    def export_statistics(self, output_path: str) -> None:
        """
        Export statistics to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        stats = self.get_statistics()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
