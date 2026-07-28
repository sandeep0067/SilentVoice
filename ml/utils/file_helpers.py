"""
File system utilities for dataset management.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


def ensure_directory(path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_size(path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        path: File path
        
    Returns:
        File size in bytes
    """
    return Path(path).stat().st_size


def get_directory_size(path: str) -> int:
    """
    Get total directory size in bytes.
    
    Args:
        path: Directory path
        
    Returns:
        Total size in bytes
    """
    total = 0
    for file_path in Path(path).rglob('*'):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


def find_video_files(directory: str, extensions: Optional[List[str]] = None) -> List[Path]:
    """
    Find all video files in directory recursively.
    
    Args:
        directory: Directory to search
        extensions: List of video extensions to include
        
    Returns:
        List of video file paths
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    video_files = []
    directory_path = Path(directory)
    
    for ext in extensions:
        video_files.extend(directory_path.rglob(f'*{ext}'))
    
    return sorted(video_files)


def copy_file_with_progress(src: str, dst: str, chunk_size: int = 8192) -> None:
    """
    Copy file with progress tracking.
    
    Args:
        src: Source file path
        dst: Destination file path
        chunk_size: Chunk size for copying
    """
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(src_path, 'rb') as f_src, open(dst_path, 'wb') as f_dst:
        while True:
            chunk = f_src.read(chunk_size)
            if not chunk:
                break
            f_dst.write(chunk)


def parallel_copy_files(src_dst_pairs: List[tuple], max_workers: int = 4) -> None:
    """
    Copy multiple files in parallel.
    
    Args:
        src_dst_pairs: List of (source, destination) tuples
        max_workers: Maximum number of parallel workers
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(copy_file_with_progress, src, dst): (src, dst)
            for src, dst in src_dst_pairs
        }
        
        for future in as_completed(futures):
            src, dst = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Failed to copy {src} to {dst}: {e}")


def clean_directory(directory: str, keep_extensions: Optional[List[str]] = None) -> None:
    """
    Clean directory by removing files not matching extensions.
    
    Args:
        directory: Directory to clean
        keep_extensions: Extensions to keep (None to remove all)
    """
    directory_path = Path(directory)
    
    for file_path in directory_path.iterdir():
        if file_path.is_file():
            if keep_extensions is None:
                file_path.unlink()
            elif file_path.suffix not in keep_extensions:
                file_path.unlink()


def create_symlink(src: str, dst: str) -> None:
    """
    Create symbolic link.
    
    Args:
        src: Source path
        dst: Destination path
    """
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dst_path.exists():
        dst_path.unlink()
    
    dst_path.symlink_to(src_path.resolve())


def get_relative_path(path: str, base: str) -> str:
    """
    Get relative path from base.
    
    Args:
        path: Full path
        base: Base directory
        
    Returns:
        Relative path
    """
    return str(Path(path).relative_to(Path(base)))
