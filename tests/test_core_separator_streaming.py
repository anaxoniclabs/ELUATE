# SPDX-License-Identifier: MIT
"""
Parity test: streaming demix vs. reference demix.

The streaming implementation is the runtime path; the reference is kept only
as the numerical oracle compared against here. Requires a real model
checkpoint and is therefore gated behind the ``ELUATE_RUN_MODEL_TESTS``
environment variable. Enable with::

    ELUATE_RUN_MODEL_TESTS=1 pytest tests/test_core_separator_streaming.py
"""

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytestmark = pytest.mark.skipif(
    os.environ.get("ELUATE_RUN_MODEL_TESTS") != "1",
    reason="Model-loading parity test; set ELUATE_RUN_MODEL_TESTS=1 to run.",
)


def _synthesize_mix(sample_rate: int, duration_seconds: float) -> np.ndarray:
    """Deterministic stereo test signal: a frequency sweep + a drifting tone."""
    n = int(sample_rate * duration_seconds)
    t = np.linspace(0, duration_seconds, n, endpoint=False, dtype=np.float64)
    f0, f1 = 220.0, 1200.0
    sweep = np.sin(2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / duration_seconds * t**2))
    tone = 0.3 * np.sin(2 * np.pi * 440.0 * t)
    left = 0.5 * sweep + tone
    right = 0.5 * sweep - tone
    return np.stack([left, right], axis=0).astype(np.float32)


def test_streaming_matches_reference(tmp_path: Path) -> None:
    from eluate.core.separator import BanditSeparator

    separator = BanditSeparator()
    _ = separator.model  # force load

    mix = _synthesize_mix(separator.model_sample_rate, duration_seconds=20.0)

    reference = separator._demix_reference(mix)

    channels = mix.shape[0]
    stem_paths, writers = separator._open_stem_writers(tmp_path, channels)
    try:
        separator._demix_streaming(mix, writers)
    finally:
        for w in writers.values():
            w.close()

    streamed = {}
    for name, path in stem_paths.items():
        data, _ = sf.read(str(path), dtype="float32", always_2d=True)
        streamed[name] = data.T  # (channels, samples)

    assert set(streamed.keys()) == set(reference.keys())
    tolerance = 1e-5
    for name in reference:
        ref = reference[name]
        stream = streamed[name]
        assert stream.shape == ref.shape, (
            f"stem {name}: shape {stream.shape} vs reference {ref.shape}"
        )
        rms = float(np.sqrt(np.mean((stream - ref) ** 2)))
        assert rms < tolerance, f"stem {name}: RMS diff {rms:.3e} exceeds {tolerance:.0e}"
