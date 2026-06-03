# SPDX-License-Identifier: MIT
"""
End-to-end pipeline test.

Drives ``EluatePipeline.process()`` on a tiny real MP4 fixture generated
on the fly with FFmpeg. The separator stage is stubbed (copies input to
output) so the test runs without a 450 MB model checkpoint, but the
extract and compile stages hit real FFmpeg and real files — exactly the
path the external audit flagged as uncovered.

Also covers the public ``eluate.elute()`` API on the same fixture, so
regressions in the API↔pipeline plumbing (outputs short-circuit, target
path resolution) surface here too.

Skipped when FFmpeg isn't on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from eluate.core.pipeline import EluatePipeline
from eluate.utils.preflight import PreflightConfig

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available on PATH",
)


def _make_test_video(path: Path, duration: float = 2.0) -> None:
    """Write a tiny MP4 with colour bars and a sine-wave audio track."""
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration}:size=128x72:rate=10",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        "-shortest",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


class _StubSeparator:
    """Drop-in for BanditSeparator that just copies audio through.

    Keeps the real pipeline wiring (extract → separate → compile) but
    removes the model dependency so the test can run in CI. ``mix_path``
    receives the input audio verbatim; speech/sfx (if requested) are
    copied as well, so a speech-only run lands the same audio at
    ``<stem>_speech.wav``.
    """

    def produce_outputs(
        self,
        audio_path: Path,
        *,
        workspace_dir: Path,
        mix_path: Path | None = None,
        speech_path: Path | None = None,
        sfx_path: Path | None = None,
        progress_callback=None,
    ) -> None:
        for target in (mix_path, speech_path, sfx_path):
            if target is None:
                continue
            target = Path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(audio_path, target)
        if progress_callback:
            progress_callback(1.0)


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    input_video = tmp_path / "input.mp4"
    _make_test_video(input_video)
    assert input_video.stat().st_size > 0

    output_dir = tmp_path / "out"

    # Bypass the lazy BanditSeparator construction — no checkpoint needed.
    monkeypatch.setattr(
        EluatePipeline,
        "separator",
        property(lambda self: _StubSeparator()),
    )

    pipeline = EluatePipeline(output_dir=output_dir, force=True)
    result = pipeline.process(input_video)

    assert result.success, result.error
    assert result.output_path is not None
    assert result.output_path.exists(), "output mp4 was not written"
    assert result.output_path.stat().st_size > 0

    # Sanity-check: the output is a real MP4 that ffprobe can parse and it
    # still has both a video and an audio stream.
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(result.output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream_types = set(probe.stdout.split())
    assert "video" in stream_types
    assert "audio" in stream_types


def test_pipeline_alac_codec_passthrough(tmp_path, monkeypatch):
    """``audio_codec="alac"`` produces an ALAC-encoded audio track.

    Companion to the unit-level argv-spy test in ``tests/test_api.py``:
    this one drives the full pipeline through real FFmpeg and verifies
    the resulting file with ``ffprobe`` so a regression in the ffmpeg
    invocation is caught at the codec-name level.
    """
    input_video = tmp_path / "input.mp4"
    _make_test_video(input_video)
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        EluatePipeline,
        "separator",
        property(lambda self: _StubSeparator()),
    )

    pipeline = EluatePipeline(
        output_dir=output_dir,
        force=True,
        audio_codec="alac",
        audio_bitrate="0",
    )
    # ALAC requires a container that supports it; mp4 does. Use the
    # default ``_eluted.mp4`` suffix.
    result = pipeline.process(input_video)
    assert result.success, result.error
    assert result.output_path is not None and result.output_path.exists()

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(result.output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.stdout.strip() == "alac"


def _fake_preflight(tmp_path: Path) -> PreflightConfig:
    """Stand-in for ``run_preflight`` that returns dummy paths.

    The pipeline's ``separator`` property is monkeypatched out in these
    tests, so ``model_path``/``config_path`` are never opened — they
    only need to be valid Path values to satisfy ``EluatePipeline``'s
    constructor.
    """
    return PreflightConfig(
        model_path=tmp_path / "model.ckpt",
        config_path=tmp_path / "config.yaml",
        arch="bandit_v2",
    )


def _patch_api_for_e2e(monkeypatch, tmp_path: Path) -> None:
    """Stub the model-loading half of the API stack.

    Patches ``run_preflight`` (so no checkpoint download is attempted)
    and ``EluatePipeline.separator`` (so no torch/checkpoint load runs).
    Everything else — Session wiring, target-path resolution, the
    extract and compile stages — runs for real.
    """
    monkeypatch.setattr(
        "eluate.api.run_preflight",
        lambda *args, **kwargs: _fake_preflight(tmp_path),
    )
    monkeypatch.setattr(
        EluatePipeline,
        "separator",
        property(lambda self: _StubSeparator()),
    )


def test_api_speech_only_writes_wav_skips_mp4(tmp_path, monkeypatch):
    """``eluate.elute(..., outputs=("speech",))`` produces ``<stem>_speech.wav``
    and skips the mux: no ``<stem>_eluted.mp4`` is written.

    Drives the full public stack (``eluate.elute`` → ``Session`` →
    ``EluatePipeline``). The target-path convention and the speech-only
    short-circuit in ``EluatePipeline.process`` are both load-bearing
    for the public API contract; this test is the only one that exercises
    them end-to-end on a real video fixture.
    """
    import eluate

    input_video = tmp_path / "song.mp4"
    _make_test_video(input_video)
    output_dir = tmp_path / "out"

    _patch_api_for_e2e(monkeypatch, tmp_path)

    result = eluate.elute(
        input_video,
        outputs=("speech",),
        output_dir=output_dir,
        force=True,
    )

    expected_speech = output_dir / "song_speech.wav"
    expected_video = output_dir / "song_eluted.mp4"

    assert result.video is None
    assert result.speech == expected_speech
    assert expected_speech.exists(), "speech.wav was not written"
    assert expected_speech.stat().st_size > 0
    assert not expected_video.exists(), (
        "speech-only run produced a _eluted.mp4 — the mux short-circuit "
        "in EluatePipeline.process is broken."
    )


def test_api_speech_only_faster_than_full_pipeline(tmp_path, monkeypatch):
    """Speech-only finishes faster than the full ``("video",)`` run.

    Both runs share the (stubbed, instant) separator and the (real)
    extract stage. The full run additionally executes the FFmpeg mux
    in ``_stage_compile``, which on a 5-second fixture costs hundreds
    of milliseconds — comfortably above sub-100 ms timing jitter, so
    the strict comparison catches a regression where speech-only stops
    skipping the compile stage without flaking on CI.
    """
    import eluate

    input_video = tmp_path / "clip.mp4"
    # 5 seconds (vs the 2-second default elsewhere) so the mux step is
    # measurable above timing jitter.
    _make_test_video(input_video, duration=5.0)

    _patch_api_for_e2e(monkeypatch, tmp_path)

    full_result = eluate.elute(
        input_video,
        outputs=("video",),
        output_dir=tmp_path / "full",
        force=True,
    )
    speech_result = eluate.elute(
        input_video,
        outputs=("speech",),
        output_dir=tmp_path / "speech_only",
        force=True,
    )

    assert full_result.video is not None and full_result.video.exists()
    assert speech_result.speech is not None and speech_result.speech.exists()

    assert speech_result.processing_time < full_result.processing_time, (
        f"speech-only ({speech_result.processing_time:.3f}s) was not faster "
        f"than full pipeline ({full_result.processing_time:.3f}s) — the "
        f"speech-only path is likely running the compile stage when it "
        f"should be skipping it."
    )
