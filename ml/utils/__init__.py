"""
Utilities module for SilentVoice.
"""

from ml.utils.file_helpers import (
    ensure_directory,
    get_file_size,
    get_directory_size,
    find_video_files,
    copy_file_with_progress,
    parallel_copy_files,
    clean_directory,
    create_symlink,
    get_relative_path,
)
from ml.utils.video_helpers import (
    get_video_duration,
    get_video_fps,
    get_video_resolution,
    get_video_codec,
    is_valid_video,
    count_frames,
    resize_video,
    extract_audio,
    create_video_from_frames,
    split_video,
)
from ml.utils.landmark_helpers import (
    normalize_landmarks,
    make_relative_to_wrist,
    smooth_landmarks,
    interpolate_missing_landmarks,
    calculate_landmark_velocity,
    calculate_landmark_acceleration,
    calculate_landmark_distances,
    filter_landmarks_by_confidence,
    pad_sequence_to_length,
)
from ml.utils.augmentation_helpers import (
    rotate_landmarks_2d,
    scale_landmarks,
    translate_landmarks,
    add_gaussian_noise,
    dropout_landmarks,
    time_warp_sequence,
    mixup_sequences,
    cutmix_sequences,
    normalize_sequence_range,
    standardize_sequence,
)
from ml.utils.seed import set_seed, get_seed

__all__ = [
    # File helpers
    'ensure_directory',
    'get_file_size',
    'get_directory_size',
    'find_video_files',
    'copy_file_with_progress',
    'parallel_copy_files',
    'clean_directory',
    'create_symlink',
    'get_relative_path',
    # Video helpers
    'get_video_duration',
    'get_video_fps',
    'get_video_resolution',
    'get_video_codec',
    'is_valid_video',
    'count_frames',
    'resize_video',
    'extract_audio',
    'create_video_from_frames',
    'split_video',
    # Landmark helpers
    'normalize_landmarks',
    'make_relative_to_wrist',
    'smooth_landmarks',
    'interpolate_missing_landmarks',
    'calculate_landmark_velocity',
    'calculate_landmark_acceleration',
    'calculate_landmark_distances',
    'filter_landmarks_by_confidence',
    'pad_sequence_to_length',
    # Augmentation helpers
    'rotate_landmarks_2d',
    'scale_landmarks',
    'translate_landmarks',
    'add_gaussian_noise',
    'dropout_landmarks',
    'time_warp_sequence',
    'mixup_sequences',
    'cutmix_sequences',
    'normalize_sequence_range',
    'standardize_sequence',
    # Seed utilities
    'set_seed',
    'get_seed',
]
