# SPDX-License-Identifier: MIT
"""
Tests for eluate.api — the public Python API.

These tests cover the public boundary established in slice 002:
``elute``, ``Session``, ``Result``, ``EluateError``. Heavy work
(preflight, pipeline, model loading) is mocked because the contract
under test is the API surface, not the underlying inference.
"""

from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import eluate
from eluate.api import EluateError, Result, Session, elute
from eluate.core.pipeline import PipelineResult, VideoInfo
from eluate.utils.preflight import PreflightConfig


def _fake_preflight(tmp_path: Path) -> PreflightConfig:
    return PreflightConfig(
        model_path=tmp_path / "model.ckpt",
        config_path=tmp_path / "config.yaml",
        arch="bandit_v2",
    )


def _make_pipeline_mock(
    output_path: Path,
    *,
    duration: int = 120,
    speech_path: Path | None = None,
    sfx_path: Path | None = None,
) -> MagicMock:
    """A pipeline mock whose ``.process`` returns a successful result.

    The mock honours the requested ``outputs`` tuple: each artifact path
    is populated only if the caller asked for that kind, so tests can
    assert the API's pass-through behaviour without running the real
    pipeline.
    """
    pipeline = MagicMock()

    def fake_process(video_path, output_path=None, *, outputs=("video",)):
        return PipelineResult(
            success=True,
            output_path=output_path if "video" in outputs else None,
            video_info=VideoInfo(
                title=Path(video_path).stem,
                duration=duration,
                duration_str="2:00",
                filepath=Path(video_path),
            ),
            processing_time=1.5,
            speech_path=speech_path if "speech" in outputs else None,
            sfx_path=sfx_path if "sfx" in outputs else None,
        )

    pipeline.process.side_effect = fake_process
    pipeline.output_dir = output_path.parent
    return pipeline


class TestPublicSurface:
    """The names re-exported from ``eluate`` are the stable contract."""

    def test_eluate_error_is_exception_subclass(self):
        assert issubclass(EluateError, Exception)

    def test_eluate_error_re_exported(self):
        assert eluate.EluateError is EluateError

    def test_elute_re_exported(self):
        assert eluate.elute is elute

    def test_session_re_exported(self):
        assert eluate.Session is Session

    def test_result_re_exported(self):
        assert eluate.Result is Result


class TestResult:
    """Result is a frozen dataclass with the documented six fields."""

    def test_six_fields_exact(self):
        names = {f.name for f in fields(Result)}
        assert names == {
            "input",
            "video",
            "speech",
            "sfx",
            "duration",
            "processing_time",
        }

    def test_is_frozen(self):
        result = Result(
            input=Path("/in.mp4"),
            video=Path("/out.mp4"),
            speech=None,
            sfx=None,
            duration=120.0,
            processing_time=10.5,
        )
        with pytest.raises(FrozenInstanceError):
            result.video = Path("/other.mp4")  # type: ignore[misc]

    def test_field_population(self):
        result = Result(
            input=Path("/in.mp4"),
            video=Path("/out.mp4"),
            speech=None,
            sfx=None,
            duration=12.5,
            processing_time=3.25,
        )
        assert result.input == Path("/in.mp4")
        assert result.video == Path("/out.mp4")
        assert result.speech is None
        assert result.sfx is None
        assert result.duration == 12.5
        assert result.processing_time == 3.25


class TestHushFunction:
    """``eluate.elute(file)`` returns a ``Result`` with the cleaned video."""

    def test_returns_result_with_expected_fields(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "video.mp4"
        video.touch()
        expected_output = tmp_path / "video_eluted.mp4"

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = _make_pipeline_mock(expected_output)

            result = elute(video)

        assert isinstance(result, Result)
        assert result.input == video.resolve()
        assert result.video == expected_output
        assert result.speech is None
        assert result.sfx is None
        assert isinstance(result.duration, float)
        assert result.duration == 120.0
        assert isinstance(result.processing_time, float)

    def test_pipeline_failure_raises_eluate_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "video.mp4"
        video.touch()

        failed_pipeline = MagicMock()
        failed_pipeline.process.return_value = PipelineResult(
            success=False,
            output_path=None,
            video_info=None,
            processing_time=0.0,
            error="boom",
        )
        failed_pipeline.output_dir = tmp_path

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = failed_pipeline

            with pytest.raises(EluateError, match="boom"):
                elute(video)


class TestSession:
    """Session owns model lifecycle and supports the context-manager protocol."""

    def test_context_manager_returns_self(self):
        session = Session()
        with session as ctx:
            assert ctx is session

    def test_context_manager_processes_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "video.mp4"
        video.touch()
        expected_output = tmp_path / "video_eluted.mp4"

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = _make_pipeline_mock(expected_output)

            with Session() as session:
                result = session.elute(video)

        assert isinstance(result, Result)
        assert result.video == expected_output

    def test_single_model_load_across_two_calls(self, tmp_path, monkeypatch):
        """Two consecutive Session.elute() calls must reuse one pipeline.

        Verified by spying on ``EluatePipeline``: the constructor is
        invoked exactly once even though ``.elute()`` is called twice,
        and ``run_preflight`` is likewise invoked once.
        """
        monkeypatch.chdir(tmp_path)
        v1 = tmp_path / "a.mp4"
        v1.touch()
        v2 = tmp_path / "b.mp4"
        v2.touch()

        out1 = tmp_path / "a_eluted.mp4"
        out2 = tmp_path / "b_eluted.mp4"
        outputs_iter = iter([out1, out2])

        pipeline_instance = MagicMock()

        def fake_process(video_path, output_path=None, *, outputs=("video",)):
            return PipelineResult(
                success=True,
                output_path=next(outputs_iter),
                video_info=VideoInfo(
                    title=Path(video_path).stem,
                    duration=60,
                    duration_str="1:00",
                    filepath=Path(video_path),
                ),
                processing_time=0.1,
            )

        pipeline_instance.process.side_effect = fake_process
        pipeline_instance.output_dir = tmp_path

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = pipeline_instance

            with Session() as session:
                r1 = session.elute(v1)
                r2 = session.elute(v2)

        assert MockPipeline.call_count == 1
        assert mock_preflight.call_count == 1
        assert pipeline_instance.process.call_count == 2
        assert r1.video == out1
        assert r2.video == out2


class TestOutputsValidation:
    """``outputs=`` is the brand-line vocabulary; rejection is load-bearing."""

    def test_music_rejected_with_brand_message(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        with pytest.raises(ValueError, match="music stem"):
            elute(video, outputs=("music",))

    def test_music_rejected_in_combination(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        with pytest.raises(ValueError, match="music stem"):
            elute(video, outputs=("video", "music"))

    def test_unknown_output_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        with pytest.raises(ValueError, match="unknown output"):
            elute(video, outputs=("vocal",))

    def test_empty_outputs_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        with pytest.raises(ValueError, match="empty"):
            elute(video, outputs=())

    def test_string_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        with pytest.raises(TypeError, match="not a single string"):
            elute(video, outputs="speech")  # type: ignore[arg-type]

    def test_list_accepted(self, tmp_path, monkeypatch):
        """Liberal in what we accept: any iterable of valid names works."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        speech_out = tmp_path / "v_speech.wav"

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = _make_pipeline_mock(
                tmp_path / "v_eluted.mp4",
                speech_path=speech_out,
            )

            result = elute(video, outputs=["speech"])  # type: ignore[arg-type]

        assert result.video is None
        assert result.speech == speech_out


class TestOutputsCombinations:
    """Each ``outputs=`` selection populates the matching ``Result`` fields."""

    def test_speech_only(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        speech_out = tmp_path / "v_speech.wav"

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = _make_pipeline_mock(
                tmp_path / "v_eluted.mp4",
                speech_path=speech_out,
                sfx_path=tmp_path / "v_sfx.wav",
            )

            result = elute(video, outputs=("speech",))

        assert result.video is None
        assert result.speech == speech_out
        assert result.sfx is None

    def test_sfx_only(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        sfx_out = tmp_path / "v_sfx.wav"

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = _make_pipeline_mock(
                tmp_path / "v_eluted.mp4",
                speech_path=tmp_path / "v_speech.wav",
                sfx_path=sfx_out,
            )

            result = elute(video, outputs=("sfx",))

        assert result.video is None
        assert result.speech is None
        assert result.sfx == sfx_out

    def test_speech_and_sfx(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        speech_out = tmp_path / "v_speech.wav"
        sfx_out = tmp_path / "v_sfx.wav"

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = _make_pipeline_mock(
                tmp_path / "v_eluted.mp4",
                speech_path=speech_out,
                sfx_path=sfx_out,
            )

            result = elute(video, outputs=("speech", "sfx"))

        assert result.video is None
        assert result.speech == speech_out
        assert result.sfx == sfx_out

    def test_all_three(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        video_out = tmp_path / "v_eluted.mp4"
        speech_out = tmp_path / "v_speech.wav"
        sfx_out = tmp_path / "v_sfx.wav"

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = _make_pipeline_mock(
                video_out,
                speech_path=speech_out,
                sfx_path=sfx_out,
            )

            result = elute(video, outputs=("video", "speech", "sfx"))

        assert result.video == video_out
        assert result.speech == speech_out
        assert result.sfx == sfx_out

    def test_default_is_video_only(self, tmp_path, monkeypatch):
        """Calling ``elute(file)`` without ``outputs=`` matches the CLI default."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        video_out = tmp_path / "v_eluted.mp4"

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = _make_pipeline_mock(
                video_out,
                speech_path=tmp_path / "v_speech.wav",
                sfx_path=tmp_path / "v_sfx.wav",
            )

            result = elute(video)

        assert result.video == video_out
        assert result.speech is None
        assert result.sfx is None

    def test_session_elute_accepts_outputs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        speech_out = tmp_path / "v_speech.wav"

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = _make_pipeline_mock(
                tmp_path / "v_eluted.mp4",
                speech_path=speech_out,
            )

            with Session() as session:
                result = session.elute(video, outputs=("speech",))

        assert result.video is None
        assert result.speech == speech_out


class TestOutputDirKwarg:
    """``output_dir`` controls where outputs land."""

    def test_default_is_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            pipeline = _make_pipeline_mock(tmp_path / "v_eluted.mp4")
            MockPipeline.return_value = pipeline

            elute(video)

        # The pipeline was constructed with output_dir=CWD.
        kwargs = MockPipeline.call_args.kwargs
        assert kwargs["output_dir"] == tmp_path

    def test_explicit_output_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        out_dir = tmp_path / "elsewhere"

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = _make_pipeline_mock(out_dir / "v_eluted.mp4")

            elute(video, output_dir=out_dir)

        kwargs = MockPipeline.call_args.kwargs
        assert kwargs["output_dir"] == out_dir

    def test_creates_missing_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        out_dir = tmp_path / "deep" / "nested" / "out"
        assert not out_dir.exists()

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = _make_pipeline_mock(out_dir / "v_eluted.mp4")

            elute(video, output_dir=out_dir)

        assert out_dir.is_dir()

    def test_pathlike_string_accepted(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        out_dir = tmp_path / "out_str"

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = _make_pipeline_mock(out_dir / "v_eluted.mp4")

            elute(video, output_dir=str(out_dir))

        assert out_dir.is_dir()


class TestOverwritePolicy:
    """``overwrite=False`` raises on collision; ``overwrite=True`` proceeds."""

    def test_collision_raises_file_exists(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        # Pre-create the colliding video output.
        existing = tmp_path / "v_eluted.mp4"
        existing.write_bytes(b"old")

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            pipeline = _make_pipeline_mock(existing)
            MockPipeline.return_value = pipeline

            with pytest.raises(FileExistsError) as exc_info:
                elute(video)

            # Pipeline must not be invoked when collision is detected.
            assert pipeline.process.call_count == 0

        # Error message identifies the colliding path.
        assert str(existing) in str(exc_info.value)

    def test_collision_in_multi_output_raises_before_work(self, tmp_path, monkeypatch):
        """Even when only ONE of the requested paths collides, no work runs."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        # speech is fine; sfx already exists.
        existing_sfx = tmp_path / "v_sfx.wav"
        existing_sfx.write_bytes(b"old")

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            pipeline = _make_pipeline_mock(
                tmp_path / "v_eluted.mp4",
                speech_path=tmp_path / "v_speech.wav",
                sfx_path=existing_sfx,
            )
            MockPipeline.return_value = pipeline

            with pytest.raises(FileExistsError) as exc_info:
                elute(video, outputs=("speech", "sfx"))

            assert pipeline.process.call_count == 0

        assert "v_sfx.wav" in str(exc_info.value)
        # The non-colliding speech path must NOT have been written.
        assert not (tmp_path / "v_speech.wav").exists()

    def test_overwrite_true_proceeds(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        existing = tmp_path / "v_eluted.mp4"
        existing.write_bytes(b"old")

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            pipeline = _make_pipeline_mock(existing)
            MockPipeline.return_value = pipeline

            result = elute(video, overwrite=True)

        assert pipeline.process.call_count == 1
        assert result.video == existing


class TestForceKwarg:
    """``force=True`` bypasses both the duration validator and disk-space precheck."""

    def test_force_propagates_to_pipeline(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            pipeline = _make_pipeline_mock(tmp_path / "v_eluted.mp4")
            MockPipeline.return_value = pipeline

            elute(video, force=True)

        # The pipeline's force flag is set to True before process() runs.
        assert pipeline.force is True

    def test_default_force_is_false(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            pipeline = _make_pipeline_mock(tmp_path / "v_eluted.mp4")
            MockPipeline.return_value = pipeline

            elute(video)

        assert pipeline.force is False

    def test_force_does_not_bypass_collision_check(self, tmp_path, monkeypatch):
        """Collision policy is independent of force."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        existing = tmp_path / "v_eluted.mp4"
        existing.write_bytes(b"old")

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = _make_pipeline_mock(existing)

            with pytest.raises(FileExistsError):
                elute(video, force=True)

    def test_real_pipeline_force_bypasses_disk_space(self, tmp_path, monkeypatch):
        """``force=True`` bypasses the disk-space precheck.

        Companion to the duration-bypass test: the issue's acceptance
        criterion calls out *both* prechecks explicitly, so each is
        verified independently against a real ``EluatePipeline``.
        """
        from eluate.core.pipeline import EluatePipeline
        from eluate.utils.validators import InsufficientDiskSpace

        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.write_bytes(b"\x00\x00\x00\x20ftypisom")

        def fake_extract(
            video_path, output_path, sample_rate, progress_callback=None, skip_duration_check=False
        ):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        class _StubSep:
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
                for target in (mix_path, speech_path, sfx_path):
                    if target is None:
                        continue
                    Path(target).parent.mkdir(parents=True, exist_ok=True)
                    Path(target).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        monkeypatch.setattr("eluate.core.pipeline.get_video_duration", lambda p: 5.0)
        monkeypatch.setattr("eluate.core.pipeline.extract_audio", fake_extract)

        def always_full(target_dir, required_bytes):
            raise InsufficientDiskSpace("no space left")

        monkeypatch.setattr("eluate.core.pipeline.check_disk_space", always_full)

        pipeline = EluatePipeline(output_dir=tmp_path)
        pipeline._separator = _StubSep()  # type: ignore[assignment]

        # Without force, the disk-space precheck fails.
        result_strict = pipeline.process(video, outputs=("speech",))
        assert not result_strict.success
        assert "no space left" in (result_strict.error or "")

        # With force=True, the same patched precheck is bypassed.
        pipeline.force = True
        result_forced = pipeline.process(video, outputs=("speech",))
        assert result_forced.success, result_forced.error

    def test_real_pipeline_force_bypasses_duration(self, tmp_path, monkeypatch):
        """End-to-end: with a duration that would normally raise
        ``DurationOutOfRange``, ``force=True`` lets the run continue.

        Drives a real ``EluatePipeline`` (not a MagicMock) so the
        precheck branch in ``pipeline.process`` is exercised. Heavy
        stages are stubbed.
        """
        from eluate.core.pipeline import EluatePipeline

        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.write_bytes(b"\x00\x00\x00\x20ftypisom")

        # Stub the heavy stages.
        def fake_extract(
            video_path, output_path, sample_rate, progress_callback=None, skip_duration_check=False
        ):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        class _StubSep:
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
                for target in (mix_path, speech_path, sfx_path):
                    if target is None:
                        continue
                    Path(target).parent.mkdir(parents=True, exist_ok=True)
                    Path(target).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        # Duration that would trigger DurationOutOfRange (way above cap).
        monkeypatch.setattr("eluate.core.pipeline.get_video_duration", lambda p: 10 * 3600)
        monkeypatch.setattr("eluate.core.pipeline.extract_audio", fake_extract)

        # Build a real pipeline directly and inject the stub separator.
        pipeline = EluatePipeline(output_dir=tmp_path)
        pipeline._separator = _StubSep()  # type: ignore[assignment]

        # Without force, the duration precheck should fail.
        result_strict = pipeline.process(video, outputs=("speech",))
        assert not result_strict.success
        assert "duration" in (result_strict.error or "").lower()

        # With force=True (mutated mid-test, as the API does), it should pass.
        pipeline.force = True
        result_forced = pipeline.process(video, outputs=("speech",))
        assert result_forced.success, result_forced.error


class TestSessionLevelDefaults:
    """``Session(...)`` constructor kwargs become per-call defaults."""

    def test_session_overwrite_default_used(self, tmp_path, monkeypatch):
        """Session(overwrite=True) lets all subsequent .elute() calls overwrite."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        existing = tmp_path / "v_eluted.mp4"
        existing.write_bytes(b"old")

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            pipeline = _make_pipeline_mock(existing)
            MockPipeline.return_value = pipeline

            with Session(overwrite=True) as session:
                result = session.elute(video)

        assert pipeline.process.call_count == 1
        assert result.video == existing

    def test_session_force_default_used(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            pipeline = _make_pipeline_mock(tmp_path / "v_eluted.mp4")
            MockPipeline.return_value = pipeline

            with Session(force=True) as session:
                session.elute(video)

        assert pipeline.force is True

    def test_session_output_dir_default_used(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        out_dir = tmp_path / "session_dir"

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = _make_pipeline_mock(out_dir / "v_eluted.mp4")

            with Session(output_dir=out_dir) as session:
                session.elute(video)

        kwargs = MockPipeline.call_args.kwargs
        assert kwargs["output_dir"] == out_dir
        assert out_dir.is_dir()


class TestPerCallOverridesSession:
    """Per-call ``Session.elute()`` kwargs override session-level defaults."""

    def test_per_call_overwrite_overrides_session_false(self, tmp_path, monkeypatch):
        """Session(overwrite=False) + .elute(overwrite=True) → overwrites."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        existing = tmp_path / "v_eluted.mp4"
        existing.write_bytes(b"old")

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            pipeline = _make_pipeline_mock(existing)
            MockPipeline.return_value = pipeline

            with Session(overwrite=False) as session:
                session.elute(video, overwrite=True)

        assert pipeline.process.call_count == 1

    def test_per_call_overwrite_false_overrides_session_true(self, tmp_path, monkeypatch):
        """Session(overwrite=True) + .elute(overwrite=False) → raises on collision."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        existing = tmp_path / "v_eluted.mp4"
        existing.write_bytes(b"old")

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = _make_pipeline_mock(existing)

            with Session(overwrite=True) as session:
                with pytest.raises(FileExistsError):
                    session.elute(video, overwrite=False)

    def test_per_call_force_overrides_session(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            pipeline = _make_pipeline_mock(tmp_path / "v_eluted.mp4")
            MockPipeline.return_value = pipeline

            with Session(force=True) as session:
                session.elute(video, force=False)

        assert pipeline.force is False

    def test_per_call_output_dir_overrides_session(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        session_dir = tmp_path / "session"
        per_call_dir = tmp_path / "per_call"

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = _make_pipeline_mock(per_call_dir / "v_eluted.mp4")

            with Session(output_dir=session_dir) as session:
                session.elute(video, output_dir=per_call_dir)

        # Per-call dir was created.
        assert per_call_dir.is_dir()
        # The pipeline was constructed targeting the per-call dir.
        assert MockPipeline.call_args.kwargs["output_dir"] == per_call_dir


class TestCodecKwargs:
    """``audio_codec`` and ``audio_bitrate`` flow through to the pipeline."""

    def test_defaults_match_cli(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            pipeline = _make_pipeline_mock(tmp_path / "v_eluted.mp4")
            MockPipeline.return_value = pipeline

            elute(video)

        assert pipeline.audio_codec == "aac"
        assert pipeline.audio_bitrate == "256k"

    def test_alac_codec_propagates(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            pipeline = _make_pipeline_mock(tmp_path / "v_eluted.mp4")
            MockPipeline.return_value = pipeline

            elute(video, audio_codec="alac")

        assert pipeline.audio_codec == "alac"

    def test_audio_bitrate_propagates(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            pipeline = _make_pipeline_mock(tmp_path / "v_eluted.mp4")
            MockPipeline.return_value = pipeline

            elute(video, audio_bitrate="320k")

        assert pipeline.audio_bitrate == "320k"

    def test_session_codec_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            pipeline = _make_pipeline_mock(tmp_path / "v_eluted.mp4")
            MockPipeline.return_value = pipeline

            with Session(audio_codec="flac", audio_bitrate="0") as session:
                session.elute(video)

        assert pipeline.audio_codec == "flac"
        assert pipeline.audio_bitrate == "0"

    def test_per_call_codec_overrides_session(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            pipeline = _make_pipeline_mock(tmp_path / "v_eluted.mp4")
            MockPipeline.return_value = pipeline

            with Session(audio_codec="flac") as session:
                session.elute(video, audio_codec="alac", audio_bitrate="192k")

        assert pipeline.audio_codec == "alac"
        assert pipeline.audio_bitrate == "192k"

    def test_codec_reaches_compile_video(self, tmp_path, monkeypatch):
        """Argv-level passthrough: ``compile_video`` is called with the codec.

        Drives a real ``EluatePipeline`` (heavy stages stubbed) and spies on
        ``compile_video`` to assert the codec kwarg flowed all the way
        through the pipeline boundary.
        """
        from eluate.core.pipeline import EluatePipeline

        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.write_bytes(b"\x00\x00\x00\x20ftypisom")

        def fake_extract(
            video_path, output_path, sample_rate, progress_callback=None, skip_duration_check=False
        ):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        class _StubSep:
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
                for target in (mix_path, speech_path, sfx_path):
                    if target is None:
                        continue
                    Path(target).parent.mkdir(parents=True, exist_ok=True)
                    Path(target).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        compile_calls: list[dict] = []

        def fake_compile(
            *,
            video_path,
            audio_path,
            output_path,
            audio_codec,
            audio_bitrate,
            progress_callback=None,
            duration=None,
        ):
            compile_calls.append({"audio_codec": audio_codec, "audio_bitrate": audio_bitrate})
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"\x00\x00\x00\x20ftypisom")

        monkeypatch.setattr("eluate.core.pipeline.get_video_duration", lambda p: 5.0)
        monkeypatch.setattr("eluate.core.pipeline.extract_audio", fake_extract)
        monkeypatch.setattr("eluate.core.pipeline.compile_video", fake_compile)

        pipeline = EluatePipeline(output_dir=tmp_path, audio_codec="alac", audio_bitrate="192k")
        pipeline._separator = _StubSep()  # type: ignore[assignment]

        result = pipeline.process(video, outputs=("video",))
        assert result.success, result.error
        assert compile_calls == [{"audio_codec": "alac", "audio_bitrate": "192k"}]


def _install_progressful_stubs(monkeypatch):
    """Wire stub stages onto ``eluate.core.pipeline`` that each emit
    progress(0.5) and progress(1.0) on their internal progress_callback.

    Returns nothing; the caller drives a real ``EluatePipeline`` (via
    ``eluate.elute`` or ``Session.elute``) and observes the public
    ``on_progress`` shim from outside.
    """

    def fake_extract(
        video_path, output_path, sample_rate, progress_callback=None, skip_duration_check=False
    ):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
        if progress_callback:
            progress_callback(0.5)
            progress_callback(1.0)

    class _StubSep:
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
            for target in (mix_path, speech_path, sfx_path):
                if target is None:
                    continue
                Path(target).parent.mkdir(parents=True, exist_ok=True)
                Path(target).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
            if progress_callback:
                progress_callback(0.5)
                progress_callback(1.0)

    def fake_compile(
        *,
        video_path,
        audio_path,
        output_path,
        audio_codec,
        audio_bitrate,
        progress_callback=None,
        duration=None,
    ):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"\x00\x00\x00\x20ftypisom")
        if progress_callback:
            progress_callback(0.5)
            progress_callback(1.0)

    monkeypatch.setattr("eluate.core.pipeline.get_video_duration", lambda p: 5.0)
    monkeypatch.setattr("eluate.core.pipeline.extract_audio", fake_extract)
    monkeypatch.setattr("eluate.core.pipeline.compile_video", fake_compile)
    # Skip the lazy BanditSeparator construction: bypass the property and
    # plant a stub instance the real pipeline can call.
    from eluate.core.pipeline import EluatePipeline

    monkeypatch.setattr(
        EluatePipeline,
        "separator",
        property(lambda self: _StubSep()),
    )


class TestOnProgress:
    """``on_progress`` collapses pipeline callbacks into one public callable."""

    def test_elute_api_drives_callback(self, tmp_path, monkeypatch):
        """``elute(file, on_progress=cb)`` end-to-end fires monotonic [0,1].

        Drives a real ``EluatePipeline`` through the public ``elute``
        function. Heavy stages are stubbed but each invokes its internal
        ``progress_callback`` so the API-layer adapter wiring is on the
        actual hot path; if a future refactor forgets to assign
        ``pipeline.on_progress``, this test breaks.
        """
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.write_bytes(b"\x00\x00\x00\x20ftypisom")
        _install_progressful_stubs(monkeypatch)

        events: list[tuple[float, str]] = []
        with patch("eluate.api.run_preflight") as mock_preflight:
            mock_preflight.return_value = _fake_preflight(tmp_path)
            elute(video, on_progress=lambda f, s: events.append((f, s)))

        assert events, "callback never fired during a real run"
        fractions = [f for f, _ in events]
        assert fractions == sorted(fractions), fractions  # monotonic
        assert all(0.0 <= f <= 1.0 for f in fractions), fractions
        assert fractions[-1] == 1.0
        seen_stages = {s for _, s in events}
        assert "extract" in seen_stages
        assert "compile" in seen_stages

    def test_on_progress_none_real_pipeline_silent(self, tmp_path, monkeypatch):
        """With ``on_progress=None``, a real run with progressful stubs
        emits zero events to a sentinel. Belt-and-braces against any
        leak through the adapter into user space."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.write_bytes(b"\x00\x00\x00\x20ftypisom")
        _install_progressful_stubs(monkeypatch)

        # Sentinel: anything that ever calls user code would have to
        # surface here. The adapter swallows everything when cb is None.
        observed: list = []

        # Patch the adapter callback slot via Session(on_progress=None)
        # path; observe by also wiring a hostile sentinel as the
        # session-level callback to confirm per-call None overrides it.
        with patch("eluate.api.run_preflight") as mock_preflight:
            mock_preflight.return_value = _fake_preflight(tmp_path)
            with Session(on_progress=lambda *a: observed.append(a)) as session:
                session.elute(video, on_progress=None)

        assert observed == []

    def test_session_reassigns_callbacks_per_call(self, tmp_path, monkeypatch):
        """Two calls on one Session: callback set, then None. Second silent.

        Pins the "always reassign pipeline callbacks" guarantee — if a
        future refactor only assigns when ``on_progress`` is truthy, the
        first call's adapter would leak into the second call and this
        test would catch it.
        """
        monkeypatch.chdir(tmp_path)
        v1 = tmp_path / "a.mp4"
        v1.write_bytes(b"\x00\x00\x00\x20ftypisom")
        v2 = tmp_path / "b.mp4"
        v2.write_bytes(b"\x00\x00\x00\x20ftypisom")
        _install_progressful_stubs(monkeypatch)

        first: list[tuple[float, str]] = []
        second: list[tuple[float, str]] = []

        with patch("eluate.api.run_preflight") as mock_preflight:
            mock_preflight.return_value = _fake_preflight(tmp_path)
            with Session() as session:
                session.elute(
                    v1,
                    overwrite=True,
                    on_progress=lambda f, s: first.append((f, s)),
                )
                # Second call: no on_progress. Must stay silent even
                # though the same pipeline object is reused.
                session.elute(v2, overwrite=True, on_progress=None)
                # Sanity: append to ``second`` only if anything fires —
                # nothing should.

                # Re-arm with a per-call sentinel for the third call.
                session.elute(
                    v1,
                    overwrite=True,
                    on_progress=lambda f, s: second.append((f, s)),
                )

        assert first and first[-1][0] == 1.0
        assert second and second[-1][0] == 1.0
        # The middle call's silence is implicit: no list captured it.

    def test_session_on_progress_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            pipeline = _make_pipeline_mock(tmp_path / "v_eluted.mp4")
            MockPipeline.return_value = pipeline

            events: list[tuple[float, str]] = []
            with Session(on_progress=lambda f, s: events.append((f, s))) as session:
                session.elute(video)

        # The MagicMock-backed pipeline.process never invokes our progress
        # plumbing, so we get only the post-success final emission.
        assert events == [(1.0, "complete")]

    def test_per_call_on_progress_overrides_session(self, tmp_path, monkeypatch):
        """A per-call callback wins; the session-level one stays silent."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            pipeline = _make_pipeline_mock(tmp_path / "v_eluted.mp4")
            MockPipeline.return_value = pipeline

            session_events: list[tuple[float, str]] = []
            per_call_events: list[tuple[float, str]] = []

            with Session(on_progress=lambda f, s: session_events.append((f, s))) as session:
                session.elute(
                    video,
                    on_progress=lambda f, s: per_call_events.append((f, s)),
                )

        assert per_call_events == [(1.0, "complete")]
        assert session_events == []

    def test_per_call_on_progress_none_overrides_session(self, tmp_path, monkeypatch):
        """Passing on_progress=None per call silences a session-level callback."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            pipeline = _make_pipeline_mock(tmp_path / "v_eluted.mp4")
            MockPipeline.return_value = pipeline

            session_events: list[tuple[float, str]] = []
            with Session(on_progress=lambda f, s: session_events.append((f, s))) as session:
                session.elute(video, on_progress=None)

        assert session_events == []

    def test_no_final_emission_on_failure(self, tmp_path, monkeypatch):
        """Failure path raises without firing the final 1.0."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        failed_pipeline = MagicMock()
        failed_pipeline.process.return_value = PipelineResult(
            success=False,
            output_path=None,
            video_info=None,
            processing_time=0.0,
            error="boom",
        )
        failed_pipeline.output_dir = tmp_path

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = failed_pipeline

            events: list[tuple[float, str]] = []
            with pytest.raises(EluateError, match="boom"):
                elute(video, on_progress=lambda f, s: events.append((f, s)))

        assert events == [], "final 1.0 must not fire when the pipeline failed"


class TestInternalCallbacksArePrivate:
    """The pipeline's five internal callback names are not part of the API."""

    @pytest.mark.parametrize(
        "kwarg",
        ["on_stage_start", "on_stage_complete", "on_error", "on_video_info"],
    )
    def test_internal_callback_kwargs_rejected_on_elute(self, kwarg, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        with pytest.raises(TypeError):
            elute(video, **{kwarg: lambda *a, **k: None})

    @pytest.mark.parametrize(
        "kwarg",
        ["on_stage_start", "on_stage_complete", "on_error", "on_video_info"],
    )
    def test_internal_callback_kwargs_rejected_on_session(self, kwarg):
        with pytest.raises(TypeError):
            Session(**{kwarg: lambda *a, **k: None})

    @pytest.mark.parametrize(
        "kwarg",
        ["on_stage_start", "on_stage_complete", "on_error", "on_video_info"],
    )
    def test_internal_callback_kwargs_rejected_on_session_elute(self, kwarg, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        with Session() as session:
            with pytest.raises(TypeError):
                session.elute(video, **{kwarg: lambda *a, **k: None})

    def test_internal_callbacks_not_in_eluate_namespace(self):
        # Public re-exports only — internal callback names must not leak.
        for name in ("on_stage_start", "on_stage_complete", "on_error", "on_video_info"):
            assert not hasattr(eluate, name)


class TestDeviceKwarg:
    """``device`` selects inference hardware; session-level only."""

    @pytest.mark.parametrize("bad", ["gpu", "metal", "", "CUDA", "auto", 0])
    def test_invalid_device_rejected_on_elute(self, bad, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        with pytest.raises(ValueError, match="valid options"):
            elute(video, device=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", ["gpu", "metal", "", "CUDA", "auto", 0])
    def test_invalid_device_rejected_on_session(self, bad):
        with pytest.raises(ValueError, match="valid options"):
            Session(device=bad)  # type: ignore[arg-type]

    def test_default_device_is_none(self, tmp_path, monkeypatch):
        """Default ``device=None`` — pipeline gets ``device_override=None``,
        so the separator's own auto-detect (CUDA > MPS > CPU) wins."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = _make_pipeline_mock(tmp_path / "v_eluted.mp4")

            elute(video)

        kwargs = MockPipeline.call_args.kwargs
        assert kwargs["device_override"] is None

    @pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
    def test_explicit_device_propagates(self, device, tmp_path, monkeypatch):
        """The string flows through to ``EluatePipeline(device_override=...)``."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = _make_pipeline_mock(tmp_path / "v_eluted.mp4")

            elute(video, device=device)

        kwargs = MockPipeline.call_args.kwargs
        assert kwargs["device_override"] == device

    def test_session_device_propagates(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = _make_pipeline_mock(tmp_path / "v_eluted.mp4")

            with Session(device="cpu") as session:
                session.elute(video)

        kwargs = MockPipeline.call_args.kwargs
        assert kwargs["device_override"] == "cpu"

    def test_session_elute_rejects_device_with_explanatory_typeerror(self, tmp_path, monkeypatch):
        """``Session.elute(device=...)`` raises ``TypeError`` and the
        message names the constraint, not just the unknown kwarg."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        with Session(device="cpu") as session:
            with pytest.raises(TypeError, match="locked at Session construction"):
                session.elute(video, device="cpu")  # type: ignore[call-arg]

    def test_session_elute_rejects_device_even_when_session_is_default(self, tmp_path, monkeypatch):
        """Even when the session was constructed without device=, the
        per-call device is rejected — there is no path that allows it."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        with Session() as session:
            with pytest.raises(TypeError, match="locked at Session construction"):
                session.elute(video, device="cuda")  # type: ignore[call-arg]

    def test_forced_cpu_constructs_separator_on_cpu(self, tmp_path, monkeypatch):
        """End-to-end: with ``device="cpu"``, the lazy separator load
        produces a torch.device("cpu"), regardless of MPS/CUDA presence.

        Drives a real ``EluatePipeline`` and intercepts ``BanditSeparator``
        construction so we don't actually load weights, but we observe
        the ``device=`` argument the pipeline computes from the override.
        """
        import torch

        from eluate.core.pipeline import EluatePipeline

        captured: dict = {}

        class _StubSep:
            def __init__(self, *, config_path, checkpoint_path, arch, device=None):
                captured["device"] = device

            def produce_outputs(self, *args, **kwargs):  # pragma: no cover
                raise AssertionError("not called in this test")

        monkeypatch.setattr("eluate.core.pipeline.BanditSeparator", _StubSep)

        pipeline = EluatePipeline(output_dir=tmp_path, device_override="cpu")
        # Trigger lazy separator construction.
        _ = pipeline.separator

        assert captured["device"] == torch.device("cpu")

    def test_auto_does_not_force_a_device(self, tmp_path, monkeypatch):
        """``device_override=None`` leaves the separator's own auto-detect
        in charge: ``BanditSeparator`` is constructed with ``device=None``."""
        from eluate.core.pipeline import EluatePipeline

        captured: dict = {}

        class _StubSep:
            def __init__(self, *, config_path, checkpoint_path, arch, device=None):
                captured["device"] = device

            def produce_outputs(self, *args, **kwargs):  # pragma: no cover
                raise AssertionError("not called in this test")

        monkeypatch.setattr("eluate.core.pipeline.BanditSeparator", _StubSep)

        pipeline = EluatePipeline(output_dir=tmp_path, device_override=None)
        _ = pipeline.separator

        assert captured["device"] is None

    def test_forced_mps_routes_to_mps_or_skipped_on_cpu_only(self, tmp_path, monkeypatch):
        """``device="mps"`` resolves to ``torch.device("mps")`` when MPS is
        available; on CPU-only hosts this branch is skipped cleanly."""
        import torch

        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            pytest.skip("MPS not available on this host")

        from eluate.core.pipeline import EluatePipeline

        captured: dict = {}

        class _StubSep:
            def __init__(self, *, config_path, checkpoint_path, arch, device=None):
                captured["device"] = device

            def produce_outputs(self, *args, **kwargs):  # pragma: no cover
                raise AssertionError("not called in this test")

        monkeypatch.setattr("eluate.core.pipeline.BanditSeparator", _StubSep)

        pipeline = EluatePipeline(output_dir=tmp_path, device_override="mps")
        _ = pipeline.separator

        assert captured["device"] == torch.device("mps")

    def test_forced_cuda_routes_to_cuda_or_skipped(self, tmp_path, monkeypatch):
        """``device="cuda"`` resolves to ``torch.device("cuda")`` when CUDA
        is available; on hosts without CUDA the branch is skipped."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available on this host")

        from eluate.core.pipeline import EluatePipeline

        captured: dict = {}

        class _StubSep:
            def __init__(self, *, config_path, checkpoint_path, arch, device=None):
                captured["device"] = device

            def produce_outputs(self, *args, **kwargs):  # pragma: no cover
                raise AssertionError("not called in this test")

        monkeypatch.setattr("eluate.core.pipeline.BanditSeparator", _StubSep)

        pipeline = EluatePipeline(output_dir=tmp_path, device_override="cuda")
        _ = pipeline.separator

        assert captured["device"] == torch.device("cuda")


class TestCheckpointKwarg:
    """``checkpoint`` selects the bandit-v2 language variant; session-locked."""

    @pytest.mark.parametrize("bad", ["english", "ENG", "", "xx", "japanese", 0, None])
    def test_invalid_checkpoint_rejected_on_elute(self, bad, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()
        with pytest.raises(ValueError, match="valid options"):
            elute(video, checkpoint=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", ["english", "ENG", "", "xx", "japanese", 0, None])
    def test_invalid_checkpoint_rejected_on_session(self, bad):
        with pytest.raises(ValueError, match="valid options"):
            Session(checkpoint=bad)  # type: ignore[arg-type]

    def test_default_checkpoint_is_multi(self, tmp_path, monkeypatch):
        """Default ``checkpoint="multi"`` matches the CLI default."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = _make_pipeline_mock(tmp_path / "v_eluted.mp4")

            elute(video)

        # run_preflight is called positionally with the checkpoint key.
        assert mock_preflight.call_args.args[0] == "multi"

    @pytest.mark.parametrize("checkpoint", ["multi", "eng", "deu", "fra", "spa", "cmn", "fao"])
    def test_explicit_checkpoint_propagates_to_preflight(self, checkpoint, tmp_path, monkeypatch):
        """The checkpoint string flows through to ``run_preflight``."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = _make_pipeline_mock(tmp_path / "v_eluted.mp4")

            elute(video, checkpoint=checkpoint)

        assert mock_preflight.call_args.args[0] == checkpoint

    def test_session_checkpoint_propagates(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = _make_pipeline_mock(tmp_path / "v_eluted.mp4")

            with Session(checkpoint="eng") as session:
                session.elute(video)

        assert mock_preflight.call_args.args[0] == "eng"

    def test_session_elute_rejects_checkpoint_with_explanatory_typeerror(
        self, tmp_path, monkeypatch
    ):
        """``Session.elute(checkpoint=...)`` raises ``TypeError`` and the
        message names the constraint, not just the unknown kwarg."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        with Session(checkpoint="eng") as session:
            with pytest.raises(TypeError, match="locked at Session construction"):
                session.elute(video, checkpoint="eng")  # type: ignore[call-arg]

    def test_session_elute_rejects_checkpoint_even_when_session_is_default(
        self, tmp_path, monkeypatch
    ):
        """Even when the session was constructed without checkpoint=,
        the per-call checkpoint is rejected — there is no path that allows it."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        with Session() as session:
            with pytest.raises(TypeError, match="locked at Session construction"):
                session.elute(video, checkpoint="multi")  # type: ignore[call-arg]

    def test_eng_checkpoint_loads_english_weights(self, tmp_path, monkeypatch):
        """End-to-end: ``checkpoint="eng"`` causes the English-tuned
        bandit-v2 weights path to be requested via ``get_checkpoint_path``.

        Drives the real ``run_preflight`` (with vendor + ffmpeg checks
        stubbed) and intercepts the download so we don't hit the network;
        the assertion is on the resolved checkpoint path's filename."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        # Stub ffmpeg presence (vendor preflight check no longer exists).
        monkeypatch.setattr("eluate.core.extractor.check_ffmpeg_available", lambda: True)

        # Steer model/config paths into tmp_path so nothing touches ~/.eluate.
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        eng_ckpt = models_dir / "checkpoint-eng.ckpt"
        # Pretend the checkpoint already exists so download isn't attempted.
        eng_ckpt.write_bytes(b"fake")
        config_yaml = tmp_path / "bandit_v2.yaml"
        config_yaml.write_text("dummy: true")

        captured: dict = {}

        def fake_get_ckpt_path(key, model="bandit-v2"):
            captured["checkpoint_key"] = key
            return models_dir / f"checkpoint-{key}.ckpt"

        monkeypatch.setattr("eluate.utils.preflight.get_checkpoint_path", fake_get_ckpt_path)
        monkeypatch.setattr("eluate.utils.preflight.get_config_path", lambda: config_yaml)

        with patch("eluate.core.pipeline.EluatePipeline") as MockPipeline:
            MockPipeline.return_value = _make_pipeline_mock(tmp_path / "v_eluted.mp4")
            elute(video, checkpoint="eng")

        assert captured["checkpoint_key"] == "eng"
        # The pipeline was constructed with the English checkpoint path.
        kwargs = MockPipeline.call_args.kwargs
        assert kwargs["model_path"] == eng_ckpt


class TestExceptionHierarchy:
    """Slice 008: typed eluate.* errors and stdlib pass-through.

    The contract is the translation table:
        low-level ``DurationOutOfRange``  → ``eluate.DurationOutOfRange``
        low-level ``InsufficientDiskSpace`` → ``eluate.InsufficientDiskSpace``
        missing checkpoint                → ``eluate.ModelNotInstalledError``
        ``FileNotFoundError``             → ``FileNotFoundError`` (unwrapped)
        ``PermissionError``               → ``PermissionError`` (unwrapped)
        ``KeyboardInterrupt``             → ``KeyboardInterrupt`` (unwrapped)
    """

    def test_duration_out_of_range_inherits_eluate_error(self):
        from eluate import DurationOutOfRange

        assert issubclass(DurationOutOfRange, eluate.EluateError)

    def test_insufficient_disk_space_inherits_eluate_error(self):
        from eluate import InsufficientDiskSpace

        assert issubclass(InsufficientDiskSpace, eluate.EluateError)

    def test_model_not_installed_inherits_eluate_error(self):
        from eluate import ModelNotInstalledError

        assert issubclass(ModelNotInstalledError, eluate.EluateError)

    def test_all_three_re_exported_from_eluate(self):
        from eluate.api import (
            DurationOutOfRange,
            InsufficientDiskSpace,
            ModelNotInstalledError,
        )

        assert eluate.DurationOutOfRange is DurationOutOfRange
        assert eluate.InsufficientDiskSpace is InsufficientDiskSpace
        assert eluate.ModelNotInstalledError is ModelNotInstalledError

    def test_eluate_error_catches_all_subclasses(self):
        for cls in (
            eluate.DurationOutOfRange,
            eluate.InsufficientDiskSpace,
            eluate.ModelNotInstalledError,
        ):
            with pytest.raises(eluate.EluateError):
                raise cls("boom")

    def test_low_level_duration_translates_to_public(self, tmp_path, monkeypatch):
        """A low-level ``validators.DurationOutOfRange`` from the pipeline
        surfaces as ``eluate.DurationOutOfRange`` with ``__cause__`` set."""
        from eluate.utils.validators import (
            DurationOutOfRange as LowLevelDuration,
        )

        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        original = LowLevelDuration("video too long")

        def fake_process(video_path, output_path=None, *, outputs=("video",)):
            return PipelineResult(
                success=False,
                output_path=None,
                video_info=None,
                processing_time=0.0,
                error=str(original),
                error_exc=original,
            )

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            pipeline = MagicMock()
            pipeline.process.side_effect = fake_process
            pipeline.output_dir = tmp_path
            MockPipeline.return_value = pipeline

            with pytest.raises(eluate.DurationOutOfRange) as excinfo:
                elute(video)

        assert excinfo.value.__cause__ is original
        # Public exception is also catchable as EluateError.
        assert isinstance(excinfo.value, eluate.EluateError)

    def test_low_level_insufficient_disk_translates_to_public(self, tmp_path, monkeypatch):
        from eluate.utils.validators import (
            InsufficientDiskSpace as LowLevelDisk,
        )

        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        original = LowLevelDisk("no space left at /tmp")

        def fake_process(video_path, output_path=None, *, outputs=("video",)):
            return PipelineResult(
                success=False,
                output_path=None,
                video_info=None,
                processing_time=0.0,
                error=str(original),
                error_exc=original,
            )

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            pipeline = MagicMock()
            pipeline.process.side_effect = fake_process
            pipeline.output_dir = tmp_path
            MockPipeline.return_value = pipeline

            with pytest.raises(eluate.InsufficientDiskSpace) as excinfo:
                elute(video)

        assert excinfo.value.__cause__ is original
        assert isinstance(excinfo.value, eluate.EluateError)

    def test_missing_checkpoint_raises_model_not_installed(self, tmp_path, monkeypatch):
        """When the checkpoint is absent, the API refuses to auto-download
        and raises ``ModelNotInstalledError`` with a remediation hint."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        # Stub ffmpeg so preflight only fails on the checkpoint check.
        monkeypatch.setattr("eluate.core.extractor.check_ffmpeg_available", lambda: True)

        # Point the checkpoint at a path that does not exist.
        missing_ckpt = tmp_path / "not-installed.ckpt"
        monkeypatch.setattr(
            "eluate.utils.preflight.get_checkpoint_path",
            lambda key, model="bandit-v2": missing_ckpt,
        )

        with pytest.raises(eluate.ModelNotInstalledError, match="eluate setup"):
            elute(video)

    def test_model_not_installed_caught_as_eluate_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        monkeypatch.setattr("eluate.core.extractor.check_ffmpeg_available", lambda: True)
        monkeypatch.setattr(
            "eluate.utils.preflight.get_checkpoint_path",
            lambda key, model="bandit-v2": tmp_path / "absent.ckpt",
        )

        with pytest.raises(eluate.EluateError):
            elute(video)

    def test_missing_input_raises_stdlib_file_not_found(self, tmp_path, monkeypatch):
        """``elute(missing_file)`` raises stdlib ``FileNotFoundError`` —
        not wrapped, so callers can ``except FileNotFoundError`` directly."""
        monkeypatch.chdir(tmp_path)

        with pytest.raises(FileNotFoundError):
            elute(tmp_path / "does_not_exist.mp4")

    def test_missing_input_is_not_eluate_error(self, tmp_path, monkeypatch):
        """The unwrapped contract: ``FileNotFoundError`` is not a
        ``EluateError``. ``except EluateError`` must NOT swallow it."""
        monkeypatch.chdir(tmp_path)

        try:
            elute(tmp_path / "does_not_exist.mp4")
        except eluate.EluateError:
            pytest.fail("FileNotFoundError must propagate unwrapped")
        except FileNotFoundError:
            pass

    def test_readonly_output_parent_raises_permission_error(self, tmp_path, monkeypatch):
        """End-to-end: a real read-only parent dir causes the API's own
        ``mkdir(parents=True)`` at api.py:313 to raise stdlib
        ``PermissionError`` before the pipeline is ever invoked."""
        import os
        import stat

        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        readonly_parent = tmp_path / "readonly"
        readonly_parent.mkdir()
        target = readonly_parent / "out"  # mkdir-ing this requires write on the parent
        os.chmod(readonly_parent, stat.S_IRUSR | stat.S_IXUSR)
        try:
            with pytest.raises(PermissionError):
                elute(video, output_dir=target)
        finally:
            # Restore so tmp_path cleanup can proceed.
            os.chmod(readonly_parent, stat.S_IRWXU)

    def test_pipeline_permission_error_propagates_unwrapped(self, tmp_path, monkeypatch):
        """A ``PermissionError`` raised inside the pipeline (e.g. a
        read-only output dir during stem write) reaches the caller as
        ``PermissionError``, not wrapped in ``EluateError``."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        original = PermissionError(13, "read-only filesystem")

        def fake_process(video_path, output_path=None, *, outputs=("video",)):
            return PipelineResult(
                success=False,
                output_path=None,
                video_info=None,
                processing_time=0.0,
                error=str(original),
                error_exc=original,
            )

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            pipeline = MagicMock()
            pipeline.process.side_effect = fake_process
            pipeline.output_dir = tmp_path
            MockPipeline.return_value = pipeline

            with pytest.raises(PermissionError) as excinfo:
                elute(video)

        # Same instance — not re-raised, not wrapped.
        assert excinfo.value is original

    def test_keyboard_interrupt_propagates_unwrapped(self, tmp_path, monkeypatch):
        """``KeyboardInterrupt`` raised mid-run is not caught and
        re-raised as ``EluateError`` — the user's Ctrl-C must surface."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        def boom(*args, **kwargs):
            raise KeyboardInterrupt()

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            pipeline = MagicMock()
            pipeline.process.side_effect = boom
            pipeline.output_dir = tmp_path
            MockPipeline.return_value = pipeline

            with pytest.raises(KeyboardInterrupt):
                elute(video)

    def test_unknown_low_level_error_falls_back_to_eluate_error(self, tmp_path, monkeypatch):
        """An unrecognised exception in ``error_exc`` becomes
        ``EluateError`` so the public surface stays a closed set."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "v.mp4"
        video.touch()

        original = RuntimeError("something internal blew up")

        def fake_process(video_path, output_path=None, *, outputs=("video",)):
            return PipelineResult(
                success=False,
                output_path=None,
                video_info=None,
                processing_time=0.0,
                error=str(original),
                error_exc=original,
            )

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            pipeline = MagicMock()
            pipeline.process.side_effect = fake_process
            pipeline.output_dir = tmp_path
            MockPipeline.return_value = pipeline

            with pytest.raises(eluate.EluateError) as excinfo:
                elute(video)

        # Plain EluateError, not one of the typed subclasses.
        assert type(excinfo.value) is eluate.EluateError


class TestStdlibLogging:
    """Slice 009: library uses stdlib ``logging``; silent by default.

    Acceptance: no Rich-direct calls leak from ``eluate.api``; library
    code routes user-visible events through ``logging.getLogger("eluate.*")``;
    stage start/complete are observable via ``caplog``.
    """

    def test_logger_names_follow_eluate_hierarchy(self):
        import eluate.api
        import eluate.core.pipeline
        import eluate.utils.preflight

        assert eluate.api.logger.name == "eluate.api"
        assert eluate.core.pipeline.logger.name == "eluate.core.pipeline"
        assert eluate.utils.preflight.logger.name == "eluate.utils.preflight"

    def test_api_module_does_not_import_rich(self):
        import eluate.api as api_module

        # The library boundary must not pull Rich into the import graph
        # of ``eluate.api`` — Rich is the CLI's rendering layer.
        assert not hasattr(api_module, "Console")

    def test_silent_by_default_under_capsys(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "video.mp4"
        video.touch()
        expected_output = tmp_path / "video_eluted.mp4"

        with (
            patch("eluate.api.run_preflight") as mock_preflight,
            patch("eluate.core.pipeline.EluatePipeline") as MockPipeline,
        ):
            mock_preflight.return_value = _fake_preflight(tmp_path)
            MockPipeline.return_value = _make_pipeline_mock(expected_output)

            elute(video)

        captured = capsys.readouterr()
        # No logging configured → eluate emits nothing on stdout/stderr.
        assert captured.out == ""
        assert captured.err == ""

    def test_pipeline_emits_stage_events_at_info(self, tmp_path, caplog):
        """Real ``EluatePipeline.process`` emits start/complete INFO events.

        Stage methods log through ``eluate.core.pipeline``; we drive a real
        pipeline with the inner I/O routines mocked so the logging seam
        fires without touching ffmpeg or the model.
        """
        import logging

        from eluate.core.pipeline import EluatePipeline, VideoInfo

        video = tmp_path / "video.mp4"
        video.touch()

        pipeline = EluatePipeline(output_dir=tmp_path, force=True)

        fake_separator = MagicMock()
        fake_separator.produce_outputs = MagicMock()

        with (
            patch("eluate.core.pipeline.get_video_duration", return_value=120.0),
            patch("eluate.core.pipeline.extract_audio") as mock_extract,
            patch("eluate.core.pipeline.compile_video") as mock_compile,
            patch.object(
                EluatePipeline,
                "separator",
                new_callable=lambda: property(lambda self: fake_separator),
            ),
            caplog.at_level(logging.INFO, logger="eluate.core.pipeline"),
        ):
            mock_extract.return_value = None
            mock_compile.return_value = None
            result = pipeline.process(video)

        assert result.success
        assert isinstance(result.video_info, VideoInfo)

        events = [
            (rec.name, rec.levelno, rec.getMessage())
            for rec in caplog.records
            if rec.name == "eluate.core.pipeline"
        ]
        messages = [msg for _, _, msg in events]
        for expected in (
            "stage extract started",
            "stage extract completed",
            "stage separate started",
            "stage separate completed",
            "stage compile started",
            "stage compile completed",
        ):
            assert expected in messages, f"missing log event: {expected!r}"
        for _, level, _ in events:
            assert level == logging.INFO

    def test_library_does_not_call_basic_config(self):
        """Library code must never call ``logging.basicConfig`` itself."""
        import pathlib

        for module_path in (
            pathlib.Path("eluate/api.py"),
            pathlib.Path("eluate/core/pipeline.py"),
            pathlib.Path("eluate/utils/preflight.py"),
        ):
            text = module_path.read_text()
            assert "basicConfig" not in text, f"{module_path} must not call logging.basicConfig"
