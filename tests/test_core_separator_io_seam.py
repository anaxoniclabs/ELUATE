# SPDX-License-Identifier: MIT
"""
Tests for BanditSeparator._demix_streaming via the injected-writers seam.

These tests bypass the real model checkpoint — a FakeModel returns zeros —
so they run in CI without ELUATE_RUN_MODEL_TESTS=1. They verify the
ring-buffer accumulation, flushing, and output-length correctness.
"""

import types

import numpy as np
import pytest
import torch

from eluate.core.separator import BanditSeparator


class InMemoryWriter:
    """Drop-in replacement for soundfile.SoundFile used by _demix_streaming."""

    def __init__(self):
        self._blocks: list[np.ndarray] = []

    def write(self, data: np.ndarray) -> None:
        self._blocks.append(data.copy())

    def close(self) -> None:
        pass

    @property
    def total_samples(self) -> int:
        return sum(b.shape[0] for b in self._blocks)

    @property
    def data(self) -> np.ndarray:
        if not self._blocks:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(self._blocks, axis=0)


def _make_separator(
    chunk_size: int = 4096,
    num_overlap: int = 4,
    batch_size: int = 4,
    stems: tuple[str, ...] = ("speech", "music", "sfx"),
) -> BanditSeparator:
    """Build a BanditSeparator with a zero-returning model, bypassing __init__."""
    sep = object.__new__(BanditSeparator)
    # Use private backing attrs — config and model are read-only lazy properties.
    sep._config = types.SimpleNamespace(
        inference=types.SimpleNamespace(
            chunk_size=chunk_size,
            num_overlap=num_overlap,
            batch_size=batch_size,
        ),
        training=types.SimpleNamespace(instruments=list(stems)),
        audio=types.SimpleNamespace(chunk_size=chunk_size),
    )
    sep.model_sample_rate = 48000
    sep.arch = "bandit_v2"
    sep.device = torch.device("cpu")

    num_stems = len(stems)

    class FakeModel:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            B, C, T = x.shape
            return torch.zeros(B, num_stems, C, T)

    sep._model = FakeModel()
    return sep


def _run_demix(sep: BanditSeparator, mix: np.ndarray) -> dict[str, InMemoryWriter]:
    stems = list(sep.config.training.instruments)
    writers: dict[str, InMemoryWriter] = {name: InMemoryWriter() for name in stems}
    sep._demix_streaming(mix, writers)  # type: ignore[arg-type]
    return writers


class TestDemixStreamingOutputLength:
    def test_exact_multiple_of_step(self):
        """Track whose length is an exact multiple of the step."""
        sep = _make_separator(chunk_size=4096, num_overlap=4, batch_size=4)
        step = 4096 // 4
        n = step * 10
        mix = np.zeros((2, n), dtype=np.float32)
        writers = _run_demix(sep, mix)
        for name, w in writers.items():
            assert w.total_samples == n, f"{name}: expected {n}, got {w.total_samples}"

    def test_arbitrary_length(self):
        """Output length equals input length for an arbitrary sample count."""
        sep = _make_separator(chunk_size=4096, num_overlap=4, batch_size=4)
        sr = sep.model_sample_rate
        n = sr * 3 + 7919  # 3 seconds plus an odd prime
        mix = np.zeros((2, n), dtype=np.float32)
        writers = _run_demix(sep, mix)
        for name, w in writers.items():
            assert w.total_samples == n

    def test_short_track_single_chunk(self):
        """A track shorter than one chunk completes without hang or wrong count."""
        sep = _make_separator(chunk_size=48000, num_overlap=4, batch_size=4)
        n = 48000 // 2  # 0.5 s — shorter than chunk_size
        mix = np.zeros((2, n), dtype=np.float32)
        writers = _run_demix(sep, mix)
        for name, w in writers.items():
            assert w.total_samples == n

    def test_mono_input(self):
        """Single-channel (mono) input is handled correctly."""
        sep = _make_separator(chunk_size=4096, num_overlap=4, batch_size=4)
        n = 48000 * 2
        mix = np.zeros((1, n), dtype=np.float32)
        writers = _run_demix(sep, mix)
        for name, w in writers.items():
            assert w.total_samples == n


class TestDemixStreamingZeroModel:
    def test_zero_model_produces_zero_output(self):
        """A zero-returning model yields all-zero output after overlap-add."""
        sep = _make_separator(chunk_size=2048, num_overlap=2, batch_size=2)
        mix = np.zeros((2, 48000), dtype=np.float32)
        writers = _run_demix(sep, mix)
        for name, w in writers.items():
            assert np.allclose(w.data, 0.0), f"{name}: expected zeros from zero model"


class TestDemixStreamingProgressCallback:
    def test_callback_reaches_1(self):
        """Progress callback reports 1.0 exactly once at the end."""
        sep = _make_separator(chunk_size=4096, num_overlap=4, batch_size=4)
        mix = np.zeros((2, 48000), dtype=np.float32)
        stems = list(sep.config.training.instruments)
        writers = {name: InMemoryWriter() for name in stems}

        reports: list[float] = []
        sep._demix_streaming(mix, writers, progress_callback=reports.append)  # type: ignore[arg-type]

        assert len(reports) > 0
        assert reports[-1] == pytest.approx(1.0)
        assert all(0.0 <= r <= 1.0 for r in reports)


class TestSelectiveStemWriters:
    def test_unrequested_stems_use_null_writer(self, tmp_path):
        """Only requested stems should create temp files on disk."""
        sep = _make_separator(stems=("speech", "music", "sfx"))

        paths, writers = sep._open_stem_writers(
            tmp_path,
            channels=2,
            persisted_stems={"speech", "sfx"},
        )
        try:
            assert set(paths) == {"speech", "sfx"}
            assert "_eluate_stem_music.wav" not in {p.name for p in paths.values()}
            writers["music"].write(np.zeros((16, 2), dtype=np.float32))
        finally:
            for writer in writers.values():
                writer.close()

        assert not (tmp_path / "_eluate_stem_music.wav").exists()

    def test_produce_outputs_persists_only_stems_needed_for_video(self, tmp_path, monkeypatch):
        """Video output needs speech+sfx, not the discarded music stem."""
        sep = _make_separator(stems=("speech", "music", "sfx"))
        captured: dict[str, set[str] | None] = {}

        monkeypatch.setattr(
            sep,
            "_load_audio",
            lambda audio_path: np.zeros((2, 48000), dtype=np.float32),
        )

        def fake_open_stem_writers(workspace_dir, channels, persisted_stems=None):
            captured["persisted_stems"] = persisted_stems
            paths = {name: tmp_path / f"{name}.wav" for name in persisted_stems}
            writers = {name: InMemoryWriter() for name in sep.config.training.instruments}
            return paths, writers

        monkeypatch.setattr(sep, "_open_stem_writers", fake_open_stem_writers)
        monkeypatch.setattr(sep, "_demix_streaming", lambda *args, **kwargs: None)
        monkeypatch.setattr(sep, "_stream_mix_normalize_write", lambda *args, **kwargs: None)

        sep.produce_outputs(
            tmp_path / "audio.wav",
            workspace_dir=tmp_path,
            mix_path=tmp_path / "no_music.wav",
        )

        assert captured["persisted_stems"] == {"speech", "sfx"}

    def test_produce_outputs_speech_only_does_not_require_sfx_temp(self, tmp_path, monkeypatch):
        """Speech-only output should not create or look up an SFX temp file."""
        sep = _make_separator(stems=("speech", "music", "sfx"))
        captured: dict[str, set[str] | None] = {}

        monkeypatch.setattr(
            sep,
            "_load_audio",
            lambda audio_path: np.zeros((2, 48000), dtype=np.float32),
        )

        def fake_open_stem_writers(workspace_dir, channels, persisted_stems=None):
            captured["persisted_stems"] = persisted_stems
            paths = {name: tmp_path / f"{name}.wav" for name in persisted_stems}
            writers = {name: InMemoryWriter() for name in sep.config.training.instruments}
            return paths, writers

        monkeypatch.setattr(sep, "_open_stem_writers", fake_open_stem_writers)
        monkeypatch.setattr(sep, "_demix_streaming", lambda *args, **kwargs: None)
        monkeypatch.setattr(sep, "_recode_wav", lambda *args, **kwargs: None)

        sep.produce_outputs(
            tmp_path / "audio.wav",
            workspace_dir=tmp_path,
            speech_path=tmp_path / "speech.wav",
        )

        assert captured["persisted_stems"] == {"speech"}


class TestDeviceCacheClearing:
    def test_cache_clear_runs_on_configured_interval(self, monkeypatch):
        """Cache clearing is throttled instead of running after every batch."""
        sep = _make_separator(chunk_size=4096, num_overlap=2, batch_size=1)
        sep.device_cache_clear_interval = 2
        mix = np.zeros((2, 4096 * 3), dtype=np.float32)
        writers = {name: InMemoryWriter() for name in sep.config.training.instruments}

        clear_calls = []
        progress_reports = []
        monkeypatch.setattr(
            "eluate.core.separator.clear_device_cache",
            lambda device: clear_calls.append(device),
        )

        sep._demix_streaming(mix, writers, progress_callback=progress_reports.append)  # type: ignore[arg-type]

        assert len(progress_reports) > 0
        assert len(clear_calls) == len(progress_reports) // 2
