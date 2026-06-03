# SPDX-License-Identifier: MIT
"""
Tests for eluate.core.pipeline module.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from eluate.core.pipeline import (
    EluatePipeline,
    PipelineResult,
    VideoInfo,
    format_duration,
)


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_zero_seconds(self):
        """Should format 0 seconds correctly."""
        assert format_duration(0) == "0:00"

    def test_seconds_only(self):
        """Should format seconds under a minute."""
        assert format_duration(45) == "0:45"

    def test_one_minute(self):
        """Should format exactly one minute."""
        assert format_duration(60) == "1:00"

    def test_minutes_and_seconds(self):
        """Should format minutes and seconds."""
        assert format_duration(125) == "2:05"

    def test_under_one_hour(self):
        """Should format time under an hour."""
        assert format_duration(3599) == "59:59"

    def test_exactly_one_hour(self):
        """Should format exactly one hour."""
        assert format_duration(3600) == "1:00:00"

    def test_hours_minutes_seconds(self):
        """Should format hours, minutes, and seconds."""
        # 2 hours, 30 minutes, 45 seconds = 9045 seconds
        assert format_duration(9045) == "2:30:45"

    def test_pads_seconds(self):
        """Should zero-pad seconds."""
        assert format_duration(61) == "1:01"

    def test_pads_minutes_in_hour_format(self):
        """Should zero-pad minutes when showing hours."""
        assert format_duration(3665) == "1:01:05"

    def test_float_input(self):
        """Should handle float input."""
        assert format_duration(90.5) == "1:30"

    def test_large_duration(self):
        """Should handle large durations."""
        # 10 hours
        assert format_duration(36000) == "10:00:00"


class TestVideoInfo:
    """Tests for VideoInfo dataclass."""

    def test_create_video_info(self):
        """Should create VideoInfo with all fields."""
        info = VideoInfo(
            title="My Video", duration=120, duration_str="2:00", filepath=Path("/path/to/video.mp4")
        )

        assert info.title == "My Video"
        assert info.duration == 120
        assert info.duration_str == "2:00"
        assert info.filepath == Path("/path/to/video.mp4")


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_successful_result(self):
        """Should create successful result."""
        result = PipelineResult(
            success=True,
            output_path=Path("/output/video.mp4"),
            video_info=None,
            processing_time=45.5,
        )

        assert result.success is True
        assert result.output_path == Path("/output/video.mp4")
        assert result.error is None

    def test_failed_result(self):
        """Should create failed result with error."""
        result = PipelineResult(
            success=False,
            output_path=None,
            video_info=None,
            processing_time=5.0,
            error="Something went wrong",
        )

        assert result.success is False
        assert result.output_path is None
        assert result.error == "Something went wrong"

    def test_default_error_is_none(self):
        """Error should default to None."""
        result = PipelineResult(
            success=True,
            output_path=Path("/output.mp4"),
            video_info=None,
            processing_time=30.0,
        )

        assert result.error is None


class TestEluatePipeline:
    """Tests for EluatePipeline class."""

    def test_init_creates_output_dir(self, temp_dir):
        """Should create output directory on init."""
        output_dir = temp_dir / "output"
        assert not output_dir.exists()

        EluatePipeline(output_dir=output_dir)

        assert output_dir.exists()

    def test_init_with_callbacks(self, temp_dir):
        """Should accept callback functions."""
        callbacks_called = []

        pipeline = EluatePipeline(
            output_dir=temp_dir,
            on_stage_start=lambda *args: callbacks_called.append("start"),
            on_progress=lambda *args: callbacks_called.append("progress"),
            on_stage_complete=lambda *args: callbacks_called.append("complete"),
            on_error=lambda *args: callbacks_called.append("error"),
            on_video_info=lambda *args: callbacks_called.append("info"),
        )

        # Callbacks should be stored
        assert pipeline.on_stage_start is not None
        assert pipeline.on_progress is not None

    def test_init_with_model_paths(self, temp_dir):
        """Should accept model and config paths."""
        model_path = temp_dir / "model.ckpt"
        config_path = temp_dir / "config.yaml"

        pipeline = EluatePipeline(
            output_dir=temp_dir,
            model_path=model_path,
            config_path=config_path,
        )

        assert pipeline.model_path == model_path
        assert pipeline.config_path == config_path

    def test_process_nonexistent_video(self, temp_dir):
        """Should return error for nonexistent video."""
        pipeline = EluatePipeline(output_dir=temp_dir)

        result = pipeline.process(Path("/nonexistent/video.mp4"))

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_separator_lazy_loaded(self, temp_dir):
        """Separator should be lazy-loaded."""
        pipeline = EluatePipeline(output_dir=temp_dir)

        # Separator should not be loaded until accessed
        assert pipeline._separator is None


class _RecordingSeparator:
    """Stub separator that records its produce_outputs calls and writes
    placeholder bytes to each requested output path."""

    def __init__(self):
        self.calls = []

    def produce_outputs(
        self,
        audio_path,
        *,
        workspace_dir,
        mix_path=None,
        speech_path=None,
        sfx_path=None,
        progress_callback=None,
    ):
        self.calls.append(
            {
                "audio_path": Path(audio_path),
                "mix_path": Path(mix_path) if mix_path is not None else None,
                "speech_path": Path(speech_path) if speech_path is not None else None,
                "sfx_path": Path(sfx_path) if sfx_path is not None else None,
            }
        )
        for target in (mix_path, speech_path, sfx_path):
            if target is None:
                continue
            target = Path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
        if progress_callback:
            progress_callback(1.0)


def _stub_pipeline(output_dir, separator):
    """Build a pipeline with the heavy stages stubbed.

    - The separator is replaced by ``_RecordingSeparator`` (no model load).
    - ``extract_audio`` is no-op'd; the pipeline writes a placeholder
      audio file in temp.
    - ``get_video_duration`` returns a small duration so the duration
      and disk-space prechecks don't reject the synthetic input.
    """
    pipeline = EluatePipeline(output_dir=output_dir)
    # Bypass the lazy BanditSeparator construction.
    EluatePipeline.separator.fget  # ensure attribute exists
    pipeline._separator = separator  # type: ignore[assignment]
    return pipeline


@pytest.fixture
def fake_video(tmp_path):
    video = tmp_path / "MyClip.mp4"
    video.write_bytes(b"\x00\x00\x00\x20ftypisom")
    return video


@pytest.fixture
def stub_extract(monkeypatch):
    """Replace extract_audio with a no-op that creates the temp wav."""

    def fake_extract(
        video_path, output_path, sample_rate, progress_callback=None, skip_duration_check=False
    ):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
        if progress_callback:
            progress_callback(1.0)

    monkeypatch.setattr("eluate.core.pipeline.extract_audio", fake_extract)
    monkeypatch.setattr("eluate.core.pipeline.get_video_duration", lambda p: 5.0)


class TestPipelineOutputs:
    """``EluatePipeline.process`` honours ``outputs=``."""

    def test_video_default_calls_compile(self, tmp_path, fake_video, stub_extract):
        sep = _RecordingSeparator()
        pipeline = _stub_pipeline(tmp_path / "out", sep)

        with patch("eluate.core.pipeline.compile_video") as mock_compile:
            # compile_video writes the output file in production; mimic that.
            def fake_compile(video_path, audio_path, output_path, **kwargs):
                Path(output_path).write_bytes(b"\x00")

            mock_compile.side_effect = fake_compile
            result = pipeline.process(fake_video)

        assert result.success, result.error
        assert mock_compile.call_count == 1
        # Mix temp must have been requested from the separator.
        assert sep.calls[0]["mix_path"] is not None
        assert sep.calls[0]["speech_path"] is None
        assert sep.calls[0]["sfx_path"] is None

    def test_speech_only_skips_compile(self, tmp_path, fake_video, stub_extract):
        sep = _RecordingSeparator()
        out_dir = tmp_path / "out"
        pipeline = _stub_pipeline(out_dir, sep)

        with patch("eluate.core.pipeline.compile_video") as mock_compile:
            result = pipeline.process(fake_video, outputs=("speech",))

        # Skip-mux: compile_video must NOT be invoked.
        assert mock_compile.call_count == 0

        assert result.success, result.error
        assert result.output_path is None
        assert result.speech_path == out_dir / "MyClip_speech.wav"
        assert result.sfx_path is None
        assert result.speech_path.exists()

        # The output dir contains exactly the speech file.
        assert sorted(p.name for p in out_dir.iterdir()) == ["MyClip_speech.wav"]

        # Separator was asked for speech only.
        assert sep.calls[0]["mix_path"] is None
        assert sep.calls[0]["speech_path"] is not None
        assert sep.calls[0]["sfx_path"] is None

    def test_sfx_only_skips_compile(self, tmp_path, fake_video, stub_extract):
        sep = _RecordingSeparator()
        out_dir = tmp_path / "out"
        pipeline = _stub_pipeline(out_dir, sep)

        with patch("eluate.core.pipeline.compile_video") as mock_compile:
            result = pipeline.process(fake_video, outputs=("sfx",))

        assert mock_compile.call_count == 0
        assert result.output_path is None
        assert result.speech_path is None
        assert result.sfx_path == out_dir / "MyClip_sfx.wav"
        assert result.sfx_path.exists()
        assert sorted(p.name for p in out_dir.iterdir()) == ["MyClip_sfx.wav"]

    def test_speech_and_sfx_skips_compile(self, tmp_path, fake_video, stub_extract):
        sep = _RecordingSeparator()
        out_dir = tmp_path / "out"
        pipeline = _stub_pipeline(out_dir, sep)

        with patch("eluate.core.pipeline.compile_video") as mock_compile:
            result = pipeline.process(fake_video, outputs=("speech", "sfx"))

        assert mock_compile.call_count == 0
        assert result.output_path is None
        assert result.speech_path == out_dir / "MyClip_speech.wav"
        assert result.sfx_path == out_dir / "MyClip_sfx.wav"
        assert sorted(p.name for p in out_dir.iterdir()) == [
            "MyClip_sfx.wav",
            "MyClip_speech.wav",
        ]

    def test_all_three_runs_compile(self, tmp_path, fake_video, stub_extract):
        sep = _RecordingSeparator()
        out_dir = tmp_path / "out"
        pipeline = _stub_pipeline(out_dir, sep)

        with patch("eluate.core.pipeline.compile_video") as mock_compile:

            def fake_compile(video_path, audio_path, output_path, **kwargs):
                Path(output_path).write_bytes(b"\x00")

            mock_compile.side_effect = fake_compile
            result = pipeline.process(fake_video, outputs=("video", "speech", "sfx"))

        assert mock_compile.call_count == 1
        assert result.output_path is not None
        assert result.output_path.name == "MyClip_eluted.mp4"
        assert result.speech_path == out_dir / "MyClip_speech.wav"
        assert result.sfx_path == out_dir / "MyClip_sfx.wav"
