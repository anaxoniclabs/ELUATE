# SPDX-License-Identifier: MIT
"""
Shared pytest fixtures for Eluate tests.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_ffmpeg_available():
    """Mock FFmpeg as available."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="ffmpeg version 6.0", stderr="")
        yield mock_run


@pytest.fixture
def mock_ffmpeg_unavailable():
    """Mock FFmpeg as unavailable."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError("ffmpeg not found")
        yield mock_run


@pytest.fixture
def mock_mps_available():
    """Mock MPS (Apple Silicon GPU) as available."""
    with (
        patch("torch.backends.mps.is_available", return_value=True),
        patch("torch.backends.mps.is_built", return_value=True),
    ):
        yield


@pytest.fixture
def mock_mps_unavailable():
    """Mock MPS as unavailable (CPU fallback)."""
    with patch("torch.backends.mps.is_available", return_value=False):
        yield


@pytest.fixture
def sample_video_path(temp_dir):
    """Create a mock video file path (doesn't actually contain video data)."""
    video_file = temp_dir / "sample.mp4"
    video_file.touch()
    return video_file


@pytest.fixture
def mock_video_duration():
    """Mock video duration detection."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="120.5\n",  # 2 minutes
            stderr="",
        )
        yield mock_run
