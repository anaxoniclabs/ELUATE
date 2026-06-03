# SPDX-License-Identifier: MIT
"""
Tests for eluate.utils.validators module.
"""

from pathlib import Path

from eluate.utils.validators import (
    format_path_for_display,
    is_valid_path,
    validate_video_file,
)


class TestValidateVideoFile:
    """Tests for validate_video_file function."""

    def test_none_path(self):
        """Should return False for None."""
        assert validate_video_file(None) is False

    def test_empty_path(self):
        """Should return False for empty string."""
        assert validate_video_file("") is False

    def test_nonexistent_file(self):
        """Should return False for nonexistent file."""
        assert validate_video_file("/nonexistent/video.mp4") is False

    def test_directory_not_file(self, temp_dir):
        """Should return False for directory."""
        assert validate_video_file(temp_dir) is False

    def test_non_video_extension(self, temp_dir):
        """Should return False for non-video extension."""
        text_file = temp_dir / "document.txt"
        text_file.touch()
        assert validate_video_file(text_file) is False

    def test_valid_mp4(self, temp_dir):
        """Should return True for .mp4 file."""
        video = temp_dir / "video.mp4"
        video.touch()
        assert validate_video_file(video) is True

    def test_valid_mkv(self, temp_dir):
        """Should return True for .mkv file."""
        video = temp_dir / "video.mkv"
        video.touch()
        assert validate_video_file(video) is True

    def test_valid_avi(self, temp_dir):
        """Should return True for .avi file."""
        video = temp_dir / "video.avi"
        video.touch()
        assert validate_video_file(video) is True

    def test_valid_mov(self, temp_dir):
        """Should return True for .mov file."""
        video = temp_dir / "video.mov"
        video.touch()
        assert validate_video_file(video) is True

    def test_valid_webm(self, temp_dir):
        """Should return True for .webm file."""
        video = temp_dir / "video.webm"
        video.touch()
        assert validate_video_file(video) is True

    def test_case_insensitive(self, temp_dir):
        """Should handle uppercase extensions."""
        video = temp_dir / "video.MP4"
        video.touch()
        assert validate_video_file(video) is True

    def test_accepts_path_object(self, temp_dir):
        """Should accept Path object."""
        video = temp_dir / "video.mp4"
        video.touch()
        assert validate_video_file(Path(video)) is True

    def test_accepts_string(self, temp_dir):
        """Should accept string path."""
        video = temp_dir / "video.mp4"
        video.touch()
        assert validate_video_file(str(video)) is True


class TestIsValidPath:
    """Tests for is_valid_path function."""

    def test_none_path(self):
        """Should return False for None."""
        assert is_valid_path(None) is False

    def test_empty_string(self):
        """Should return False for empty string."""
        assert is_valid_path("") is False

    def test_non_string(self):
        """Should return False for non-string."""
        assert is_valid_path(123) is False

    def test_null_byte(self):
        """Should return False for path with null byte."""
        assert is_valid_path("/path/with\x00null") is False

    def test_too_long(self):
        """Should return False for path over 4096 chars."""
        long_path = "a" * 5000
        assert is_valid_path(long_path) is False

    def test_valid_path(self):
        """Should return True for valid path."""
        assert is_valid_path("/Users/test/video.mp4") is True

    def test_relative_path(self):
        """Should return True for relative path."""
        assert is_valid_path("./video.mp4") is True

    def test_path_with_spaces(self):
        """Should return True for path with spaces."""
        assert is_valid_path("/Users/test user/my video.mp4") is True


class TestFormatPathForDisplay:
    """Tests for format_path_for_display function."""

    def test_empty_path(self):
        """Should return empty string for empty input."""
        assert format_path_for_display("") == ""

    def test_none_path(self):
        """Should return empty string for None."""
        assert format_path_for_display(None) == ""

    def test_short_path_unchanged(self):
        """Should not truncate short paths."""
        short_path = "/Users/test/video.mp4"
        assert format_path_for_display(short_path) == short_path

    def test_exact_max_length(self):
        """Should not truncate path at exact max length."""
        path = "a" * 50
        assert format_path_for_display(path, max_length=50) == path

    def test_truncation(self):
        """Should truncate long paths in the middle."""
        long_path = "/Users/very/long/path/to/some/deeply/nested/video.mp4"
        result = format_path_for_display(long_path, max_length=30)

        assert len(result) <= 30
        assert "..." in result

    def test_truncation_preserves_ends(self):
        """Should preserve start and end of path."""
        path = "/start/middle/end"
        result = format_path_for_display(path, max_length=15)

        # Should have beginning and end with ... in middle
        assert result.startswith("/sta")
        assert result.endswith("end")

    def test_custom_max_length(self):
        """Should respect custom max_length."""
        path = "a" * 100
        result = format_path_for_display(path, max_length=20)
        assert len(result) <= 20
