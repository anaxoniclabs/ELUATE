# SPDX-License-Identifier: MIT
"""
CLI smoke test.

Drives ``eluate`` through ``cli.main`` on a tiny real MP4 fixture. The
preflight (model download) and separator (torch + checkpoint) stages
are stubbed so the test runs in CI without a 450 MB checkpoint, but
every other layer — argument parsing, CLI→API plumbing, the API's
session lifecycle, the pipeline's extract and compile stages — runs
end-to-end with real FFmpeg.

A regression in any of those layers (CLI flag mapping, progress wiring,
mux path) would fail this test.

Skipped when FFmpeg isn't on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
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
    """Drop-in for ``BanditSeparator`` that copies audio through.

    Identical to the stub in ``test_pipeline_e2e.py``; duplicated here
    so each test file is self-contained and runnable in isolation.
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


def _fake_preflight(tmp_path: Path) -> PreflightConfig:
    return PreflightConfig(
        model_path=tmp_path / "model.ckpt",
        config_path=tmp_path / "config.yaml",
        arch="bandit_v2",
    )


def test_cli_smoke_produces_eluted_mp4(tmp_path, monkeypatch):
    """``eluate <fixture> -o <out>`` produces a probeable ``_eluted.mp4``.

    Both layers run preflight: the CLI runs it directly to render the
    Rich download UI on first run, and the API's ``Session`` runs it
    again internally; both call sites are stubbed. The separator is
    stubbed at the class level so the API's lazy load returns the stub.
    """
    from eluate import cli

    input_video = tmp_path / "song.mp4"
    _make_test_video(input_video)

    output_path = tmp_path / "song_eluted.mp4"

    def fake_preflight(*args, **kwargs):
        return _fake_preflight(tmp_path)

    monkeypatch.setattr("eluate.api.run_preflight", fake_preflight)
    monkeypatch.setattr("eluate.cli.run_preflight", fake_preflight)
    monkeypatch.setattr(
        EluatePipeline,
        "separator",
        property(lambda self: _StubSeparator()),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["eluate", str(input_video), "-o", str(output_path), "--force"],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 0, "eluate CLI exited non-zero"

    assert output_path.exists(), "expected _eluted.mp4 was not produced"
    assert output_path.stat().st_size > 0

    # Validate the mux output is a real MP4 with both a video and audio
    # stream. ffprobe failing here points to a regression in the compile
    # stage rather than a path-resolution bug.
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream_types = set(probe.stdout.split())
    assert "video" in stream_types
    assert "audio" in stream_types
