# SPDX-License-Identifier: MIT
"""
Audio extraction from video files using FFmpeg.
"""

from pathlib import Path
from typing import Callable, Optional

from eluate.utils.ffmpeg import (
    FFmpegError,
    FFmpegTimeout,
    media_timeout,
    run_ffmpeg_simple,
    run_ffmpeg_with_progress,
)
from eluate.utils.validators import validate_duration

# Default sample rate (Bandit v2 uses 48kHz, so this is overridden in pipeline)
SAMPLE_RATE = 48000


def extract_audio(
    video_path: Path,
    output_path: Path,
    sample_rate: int = SAMPLE_RATE,
    progress_callback: Optional[Callable[[float], None]] = None,
    *,
    skip_duration_check: bool = False,
) -> Path:
    """
    Extract audio from video file using FFmpeg.

    Args:
        video_path: Path to input video file
        output_path: Path for output WAV file
        sample_rate: Target sample rate (default 48000)
        progress_callback: Callback for progress updates (0.0 to 1.0)
        skip_duration_check: If True, bypass the plausibility check on
            the probed duration. Used by ``--force`` at the CLI layer.

    Returns:
        Path to extracted audio file

    Raises:
        FileNotFoundError: If input video doesn't exist
        DurationOutOfRange: If probed duration is implausible and
            ``skip_duration_check`` is False.
        FFmpegError / FFmpegTimeout: On FFmpeg failure or timeout.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = get_video_duration(video_path)

    if not skip_duration_check:
        # Raises DurationOutOfRange for zero/negative/unreasonably-long
        # values. ``None`` (duration unknown) is accepted silently.
        validate_duration(duration)

    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",  # No video
        "-acodec",
        "pcm_s16le",  # 16-bit PCM (lossless)
        "-ar",
        str(sample_rate),  # Sample rate
        "-ac",
        "2",  # Stereo
        "-y",  # Overwrite output
        str(output_path),
    ]

    timeout = media_timeout(duration)

    if progress_callback and duration:
        run_ffmpeg_with_progress(
            cmd,
            duration,
            progress_callback,
            timeout=timeout,
            stage="extract",
        )
    else:
        run_ffmpeg_simple(cmd, timeout=timeout, stage="extract")

    return output_path


def get_video_duration(video_path: Path) -> Optional[float]:
    """
    Get duration of video file in seconds.

    Args:
        video_path: Path to video file

    Returns:
        Duration in seconds or None if unable to determine
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    try:
        result = run_ffmpeg_simple(cmd, timeout=30.0, stage="probe")
        return float(result.stdout.strip())
    except (FFmpegError, FFmpegTimeout, ValueError):
        return None


def check_ffmpeg_available() -> bool:
    """
    Check if FFmpeg is available on the system.

    Returns:
        True if FFmpeg is installed and accessible
    """
    try:
        run_ffmpeg_simple(["ffmpeg", "-version"], timeout=10.0, stage="version-check")
        return True
    except (FFmpegError, FFmpegTimeout, FileNotFoundError):
        return False


def get_ffmpeg_version() -> Optional[str]:
    """
    Get FFmpeg version string.

    Returns:
        Version string or None if FFmpeg not available
    """
    try:
        result = run_ffmpeg_simple(["ffmpeg", "-version"], timeout=10.0, stage="version-check")
        return result.stdout.split("\n")[0]
    except (FFmpegError, FFmpegTimeout, FileNotFoundError):
        return None
