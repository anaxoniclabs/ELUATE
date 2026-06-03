# SPDX-License-Identifier: MIT
"""
Tests for eluate.core.extractor module.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from eluate.core.extractor import (
    SAMPLE_RATE,
    check_ffmpeg_available,
    get_ffmpeg_version,
    get_video_duration,
)


class TestCheckFfmpegAvailable:
    """Tests for check_ffmpeg_available function."""

    def test_returns_true_when_available(self, mock_ffmpeg_available):
        """Should return True when FFmpeg is installed."""
        result = check_ffmpeg_available()
        assert result is True

    def test_returns_false_when_not_found(self, mock_ffmpeg_unavailable):
        """Should return False when FFmpeg is not installed."""
        result = check_ffmpeg_available()
        assert result is False

    def test_returns_false_on_subprocess_error(self):
        """Should return False on subprocess error."""
        with patch("subprocess.run") as mock_run:
            from subprocess import CalledProcessError

            mock_run.side_effect = CalledProcessError(1, "ffmpeg")
            result = check_ffmpeg_available()
            assert result is False


class TestGetFfmpegVersion:
    """Tests for get_ffmpeg_version function."""

    def test_returns_version_string(self):
        """Should return version string when available."""
        mock_output = "ffmpeg version 6.0 Copyright (c) 2000-2023\nbuilt with..."

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=mock_output, returncode=0)
            result = get_ffmpeg_version()

            assert result == "ffmpeg version 6.0 Copyright (c) 2000-2023"

    def test_returns_none_when_not_available(self, mock_ffmpeg_unavailable):
        """Should return None when FFmpeg not available."""
        result = get_ffmpeg_version()
        assert result is None

    def test_returns_none_on_error(self):
        """Should return None on subprocess error."""
        with patch("subprocess.run") as mock_run:
            from subprocess import CalledProcessError

            mock_run.side_effect = CalledProcessError(1, "ffmpeg")
            result = get_ffmpeg_version()
            assert result is None


class TestGetVideoDuration:
    """Tests for get_video_duration function."""

    def test_returns_duration_in_seconds(self):
        """Should return duration as float."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="120.500000\n", returncode=0)
            result = get_video_duration(Path("/test/video.mp4"))

            assert result == 120.5

    def test_returns_none_on_error(self):
        """Should return None on ffprobe error."""
        with patch("subprocess.run") as mock_run:
            from subprocess import CalledProcessError

            mock_run.side_effect = CalledProcessError(1, "ffprobe")
            result = get_video_duration(Path("/test/video.mp4"))

            assert result is None

    def test_returns_none_on_invalid_output(self):
        """Should return None on non-numeric output."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="invalid\n", returncode=0)
            result = get_video_duration(Path("/test/video.mp4"))

            assert result is None

    def test_handles_integer_duration(self):
        """Should handle integer duration output."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="300\n", returncode=0)
            result = get_video_duration(Path("/test/video.mp4"))

            assert result == 300.0


class TestSampleRate:
    """Tests for SAMPLE_RATE constant."""

    def test_sample_rate_is_48khz(self):
        """Sample rate should be 48kHz for Bandit v2."""
        assert SAMPLE_RATE == 48000

    def test_sample_rate_is_int(self):
        """Sample rate should be an integer."""
        assert isinstance(SAMPLE_RATE, int)
