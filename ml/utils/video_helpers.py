"""
Video processing utilities.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List


def get_video_duration(video_path: str) -> float:
    """
    Get video duration in seconds.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Duration in seconds
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return frame_count / fps if fps > 0 else 0


def get_video_fps(video_path: str) -> float:
    """
    Get video FPS.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Frames per second
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    return fps


def get_video_resolution(video_path: str) -> Tuple[int, int]:
    """
    Get video resolution.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (width, height)
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return (width, height)


def get_video_codec(video_path: str) -> str:
    """
    Get video codec.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Codec string
    """
    cap = cv2.VideoCapture(video_path)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    
    return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])


def is_valid_video(video_path: str) -> bool:
    """
    Check if video file is valid and can be opened.
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if video is valid
    """
    cap = cv2.VideoCapture(video_path)
    is_valid = cap.isOpened()
    cap.release()
    
    return is_valid


def count_frames(video_path: str) -> int:
    """
    Count total frames in video.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Total frame count
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return frame_count


def resize_video(
    video_path: str,
    output_path: str,
    target_size: Tuple[int, int],
    target_fps: Optional[float] = None
) -> None:
    """
    Resize video to target size and FPS.
    
    Args:
        video_path: Input video path
        output_path: Output video path
        target_size: Target size (width, height)
        target_fps: Target FPS (None to keep original)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = target_fps if target_fps is not None else original_fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        resized = cv2.resize(frame, target_size)
        out.write(resized)
    
    cap.release()
    out.release()


def extract_audio(video_path: str, output_path: str) -> None:
    """
    Extract audio from video (requires ffmpeg).
    
    Args:
        video_path: Input video path
        output_path: Output audio path
    """
    import subprocess
    
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vn',
        '-acodec', 'libmp3lame',
        '-q:a', '2',
        output_path
    ]
    
    subprocess.run(cmd, check=True)


def create_video_from_frames(
    frames_dir: str,
    output_path: str,
    fps: float = 30.0,
    codec: str = 'mp4v'
) -> None:
    """
    Create video from frame images.
    
    Args:
        frames_dir: Directory containing frame images
        output_path: Output video path
        fps: Frames per second
        codec: Video codec
    """
    frames_path = Path(frames_dir)
    frame_files = sorted(frames_path.glob('*.png'))
    frame_files.extend(sorted(frames_path.glob('*.jpg')))
    frame_files = sorted(frame_files)
    
    if not frame_files:
        raise ValueError("No frame images found")
    
    first_frame = cv2.imread(str(frame_files[0]))
    height, width = first_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        out.write(frame)
    
    out.release()


def split_video(
    video_path: str,
    output_dir: str,
    segment_duration: float = 5.0
) -> List[str]:
    """
    Split video into segments.
    
    Args:
        video_path: Input video path
        output_dir: Output directory
        segment_duration: Duration of each segment in seconds
        
    Returns:
        List of output segment paths
    """
    import subprocess
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-c', 'copy',
        '-f', 'segment',
        '-segment_time', str(segment_duration),
        str(output_path / 'segment_%03d.mp4')
    ]
    
    subprocess.run(cmd, check=True)
    
    segment_files = sorted(output_path.glob('segment_*.mp4'))
    return [str(f) for f in segment_files]
