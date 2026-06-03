# SPDX-License-Identifier: MIT
"""
File validation utilities.
"""

from pathlib import Path
from typing import Optional

# Canonical set of recognised video extensions. Single source of truth for
# both interactive-mode validation (``validate_video_file``) and the
# ``--folder`` scan in the CLI, so the two acceptance surfaces cannot drift.
VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".3gp",
    ".3g2",
    ".ts",
    ".mts",
    ".m2ts",
}


def validate_video_file(path: str | Path) -> bool:
    """
    Validate that a path points to a valid video file.

    Args:
        path: Path to validate

    Returns:
        True if path exists and has a video extension
    """
    if not path:
        return False

    path = Path(path)

    if not path.exists():
        return False

    if not path.is_file():
        return False

    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_valid_path(path: str) -> bool:
    """
    Check if a string is a valid file path.

    Args:
        path: Path string to validate

    Returns:
        True if path is valid
    """
    if not path or not isinstance(path, str):
        return False

    # Check for null bytes or other problematic characters
    if "\x00" in path:
        return False

    # Basic length check
    if len(path) > 4096:
        return False

    return True


def format_path_for_display(path: str, max_length: int = 50) -> str:
    """
    Format a path for display, truncating if needed.

    Args:
        path: Path to format
        max_length: Maximum display length

    Returns:
        Formatted path string
    """
    if not path:
        return ""

    if len(path) <= max_length:
        return path

    # Truncate in the middle
    half = (max_length - 3) // 2
    return f"{path[:half]}...{path[-half:]}"


# --- Reliability prechecks -------------------------------------------------


class DurationOutOfRange(ValueError):
    """Video duration is zero, negative, or unreasonably long."""


class InsufficientDiskSpace(RuntimeError):
    """Target filesystem does not have enough free space for processing."""


# Minimum plausible duration: shorter than this and the file is almost
# certainly not real video content (e.g. corrupt header, empty container).
MIN_DURATION_SECONDS = 0.5

# Upper bound — configurable but defaulting to 6 hours. Extraction and
# separation both scale linearly; anything longer is almost certainly a
# misdirected input, and the user can override with ``--force`` at the CLI.
MAX_DURATION_SECONDS = 6 * 60 * 60


def validate_duration(
    duration_seconds: Optional[float],
    *,
    min_seconds: float = MIN_DURATION_SECONDS,
    max_seconds: float = MAX_DURATION_SECONDS,
) -> None:
    """Raise ``DurationOutOfRange`` if duration is implausible.

    ``None`` duration is treated as "unknown" and accepted — ffprobe
    occasionally fails to report duration on exotic containers, and we
    don't want to hard-fail those runs on the precheck alone.
    """
    if duration_seconds is None:
        return
    if duration_seconds <= 0:
        raise DurationOutOfRange(
            f"Video duration is {duration_seconds:.1f}s. File is likely "
            "empty, corrupt, or not a video."
        )
    if duration_seconds < min_seconds:
        raise DurationOutOfRange(
            f"Video duration is only {duration_seconds:.2f}s (minimum "
            f"{min_seconds:.1f}s). Too short for meaningful separation."
        )
    if duration_seconds > max_seconds:
        hours = duration_seconds / 3600
        cap_hours = max_seconds / 3600
        raise DurationOutOfRange(
            f"Video duration is {hours:.1f}h (cap: {cap_hours:.1f}h). "
            "Processing would likely take many hours. Pass --force to "
            "override if this is intentional."
        )


def estimate_required_disk_bytes(
    duration_seconds: float,
    *,
    sample_rate: int = 48000,
    channels: int = 2,
    bytes_per_sample: int = 4,
    overhead_multiplier: float = 3.5,
    minimum_bytes: int = 200 * 1024 * 1024,
) -> int:
    """Estimate bytes needed in the target directory for the full pipeline.

    Covers: extracted WAV (pcm_s16le stereo), intermediate float32 stems,
    and the output video remux. The multiplier includes headroom for
    temp files and the final container.
    """
    raw = int(duration_seconds * sample_rate * channels * bytes_per_sample)
    return max(int(raw * overhead_multiplier), minimum_bytes)


def check_disk_space(target_dir: "Path | str", required_bytes: int) -> None:
    """Raise ``InsufficientDiskSpace`` if the target filesystem is too full."""
    import shutil

    target = Path(target_dir)
    # Walk up to the nearest existing parent — ``disk_usage`` needs a real path.
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError as exc:
        raise InsufficientDiskSpace(f"Cannot inspect free space at {probe}: {exc}") from exc

    if free < required_bytes:
        need_mb = required_bytes // (1024 * 1024)
        have_mb = free // (1024 * 1024)
        raise InsufficientDiskSpace(
            f"Need ~{need_mb} MB free at {probe}, but only {have_mb} MB "
            "available. Free up space or choose a different output directory."
        )
