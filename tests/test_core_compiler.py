# SPDX-License-Identifier: MIT
"""
Tests for eluate.core.compiler module.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

# compile_video delegates duration probing to the canonical helper in
# extractor; test that single source of truth here.
from eluate.core.extractor import get_video_duration as _get_duration


class TestGetDuration:
    """Tests for the ffprobe duration helper used by compile_video."""

    def test_returns_duration_in_seconds(self):
        """Should return duration as float."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="240.500000\n", returncode=0)
            result = _get_duration(Path("/test/video.mp4"))

            assert result == 240.5

    def test_returns_none_on_error(self):
        """Should return None on ffprobe error."""
        with patch("subprocess.run") as mock_run:
            from subprocess import CalledProcessError

            mock_run.side_effect = CalledProcessError(1, "ffprobe")
            result = _get_duration(Path("/test/video.mp4"))

            assert result is None

    def test_returns_none_on_invalid_output(self):
        """Should return None on non-numeric output."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="not a number\n", returncode=0)
            result = _get_duration(Path("/test/video.mp4"))

            assert result is None

    def test_handles_integer_duration(self):
        """Should handle integer duration output."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="600\n", returncode=0)
            result = _get_duration(Path("/test/video.mp4"))

            assert result == 600.0

    def test_handles_very_long_duration(self):
        """Should handle long video duration."""
        # 3 hours = 10800 seconds
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="10800.0\n", returncode=0)
            result = _get_duration(Path("/test/long_video.mp4"))

            assert result == 10800.0

    def test_handles_fractional_seconds(self):
        """Should handle precise fractional durations."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="123.456789\n", returncode=0)
            result = _get_duration(Path("/test/video.mp4"))

            assert result == 123.456789
