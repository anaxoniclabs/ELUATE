# SPDX-License-Identifier: MIT
"""
Video compilation - merging processed audio with original video.
"""

from pathlib import Path
from typing import Callable, Optional

from eluate.utils.ffmpeg import (
    media_timeout,
    run_ffmpeg_simple,
    run_ffmpeg_with_progress,
)

from .extractor import get_video_duration


def compile_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    audio_codec: str = "aac",
    audio_bitrate: str = "256k",
    progress_callback: Optional[Callable[[float], None]] = None,
    duration: Optional[float] = None,
) -> Path:
    """
    Merge processed audio back with original video.

    The video stream is copied (no re-encoding) for speed.
    Only the audio is encoded.

    Args:
        video_path: Original video file
        audio_path: Processed audio file (dialogue + effects)
        output_path: Output video file
        audio_codec: Audio codec for output (default: aac)
        audio_bitrate: Audio bitrate (default: 256k)
        progress_callback: Callback for progress updates (0.0 to 1.0)

    Returns:
        Path to output video

    Raises:
        FFmpegError / FFmpegTimeout: On FFmpeg failure or timeout.
        FileNotFoundError: If input files don't exist
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if duration is None:
        duration = get_video_duration(video_path)

    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),  # Input video
        "-i",
        str(audio_path),  # Input audio
        "-map",
        "0:v",  # Take video from first input
        "-map",
        "1:a",  # Take audio from second input
        "-map",
        "0:s?",  # Subtitle tracks if present
        "-map",
        "0:d?",  # Data streams if present
        "-c:v",
        "copy",  # Copy video stream (no re-encoding)
        "-c:a",
        audio_codec,  # Encode audio
        "-b:a",
        audio_bitrate,  # Audio bitrate
        "-c:s",
        "copy",  # Copy subtitle streams
        "-c:d",
        "copy",  # Copy data streams
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
            stage="compile",
        )
    else:
        run_ffmpeg_simple(cmd, timeout=timeout, stage="compile")

    return output_path
