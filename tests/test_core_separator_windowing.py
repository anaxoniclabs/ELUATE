# SPDX-License-Identifier: MIT
"""Per-chunk overlap-add windowing tests for BanditSeparator.

Two layers:

1. Unit tests for the pure helper ``_chunk_window`` — covers the
   four {skip_fade_in, skip_fade_out} combinations.
2. Integration test of ``_demix_reference`` with a mock model that
   returns ones. Verifies the counter accumulator at the chunk 0 /
   chunk 1 boundary equals 1 (from a proper ramp-in + ramp-out),
   NOT 2 (which the pre-fix code produced by flattening chunk 1's
   fade-in to 1 while chunk 0's window was also 1 at that position).

Neither layer requires the real Bandit v2 checkpoint; both run in
plain pytest.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eluate.core.separator import BanditSeparator

# ---------------------------------------------------------------------------
# 1. Unit tests for _chunk_window
# ---------------------------------------------------------------------------

CHUNK = 20
FADE = 4


@pytest.fixture
def base_window():
    w = torch.ones(CHUNK)
    w[:FADE] = torch.linspace(0, 1, FADE)
    w[-FADE:] = torch.linspace(1, 0, FADE)
    return w


def test_chunk_window_skip_fade_in_only(base_window):
    w = BanditSeparator._chunk_window(base_window, FADE, skip_fade_in=True, skip_fade_out=False)
    assert torch.equal(w[:FADE], torch.ones(FADE))
    assert torch.equal(w[FADE:-FADE], base_window[FADE:-FADE])
    assert torch.equal(w[-FADE:], base_window[-FADE:])


def test_chunk_window_skip_fade_out_only(base_window):
    w = BanditSeparator._chunk_window(base_window, FADE, skip_fade_in=False, skip_fade_out=True)
    assert torch.equal(w[:FADE], base_window[:FADE])
    assert torch.equal(w[FADE:-FADE], base_window[FADE:-FADE])
    assert torch.equal(w[-FADE:], torch.ones(FADE))


def test_chunk_window_neither(base_window):
    # Modal case: neither flag set, so the function returns the shared
    # template directly. Callers must treat the result as read-only —
    # the streaming/reference demix paths only multiply-and-slice it.
    w = BanditSeparator._chunk_window(base_window, FADE, skip_fade_in=False, skip_fade_out=False)
    assert w is base_window


def test_chunk_window_both(base_window):
    w = BanditSeparator._chunk_window(base_window, FADE, skip_fade_in=True, skip_fade_out=True)
    assert torch.equal(w, torch.ones(CHUNK))


# ---------------------------------------------------------------------------
# 2. Integration test with a mock model
# ---------------------------------------------------------------------------
#
# The pre-fix bug: at batch_size >= 2, chunks 1..N-1 of the first batch
# received a window with window[:fade_size] flattened to 1 (same as
# chunk 0). Consequence: at absolute position `step` (start of chunk 1),
# the counter accumulated chunk 0's fade-out value AT position step
# (which is 1, since step == chunk_size/2 is in chunk 0's middle plateau)
# PLUS chunk 1's fade-in value at local offset 0, which was 1 (flattened)
# instead of 0 (the correct ramp start). So counter[..., step] = 2.
#
# Post-fix: chunk 1 gets a normal ramp-in window, so counter[..., step] =
# 1 (middle of chunk 0) + 0 (start of chunk 1's ramp) = 1.
#
# We drive `_demix_reference` with a mock model that echoes its input
# back, and inspect the `result` / `counter` tensors by reconstructing
# the accumulation from within the method. Easiest route: patch the
# accumulator-division step and instead export the raw counter via a
# side-effect on a list; then inspect.


def _make_separator_for_windowing(
    *,
    chunk_size: int,
    num_overlap: int,
    batch_size: int,
    fade_size: int,
    stems: tuple[str, ...] = ("speech", "music", "sfx"),
) -> BanditSeparator:
    """Build a BanditSeparator with a mock model + minimal config.

    No checkpoint load, no MPS, no real Bandit module — just enough
    surface area for ``_demix_reference`` to run.
    """
    sep = BanditSeparator.__new__(BanditSeparator)
    sep.config_path = None
    sep.checkpoint_path = None
    sep.arch = "bandit_v2"
    sep.device = torch.device("cpu")
    sep.model_sample_rate = 48000
    # ``fade_size`` is derived as chunk_size // 10 inside demix — don't
    # override that. Pick chunk_size so chunk_size // 10 == fade_size.
    assert chunk_size // 10 == fade_size, (
        "fixture must choose chunk_size such that chunk_size // 10 == fade_size"
    )
    sep._config = SimpleNamespace(
        audio=SimpleNamespace(
            sample_rate=48000,
            chunk_size=chunk_size,
        ),
        inference=SimpleNamespace(
            num_overlap=num_overlap,
            batch_size=batch_size,
        ),
        training=SimpleNamespace(
            instruments=list(stems),
        ),
    )
    num_stems = len(stems)

    def mock_model(arr: torch.Tensor) -> torch.Tensor:
        # arr: (batch, channels, chunk_size). Return
        #     (batch, num_stems, channels, chunk_size) of ones.
        b, ch, cs = arr.shape
        return torch.ones(b, num_stems, ch, cs, dtype=arr.dtype)

    sep._model = mock_model
    return sep


def test_first_batch_counter_is_correct_at_chunk_boundary(monkeypatch):
    """Reproduce the fade-in bug at the chunk 0 / chunk 1 boundary.

    Construct a batch size ≥ 2 config and a track long enough to have
    two full chunks in batch 1 plus some trailing material. The
    counter at absolute position ``step`` should equal ~1.0 (chunk 0's
    middle at that position + chunk 1's ramp-in start).
    """
    chunk_size = 2000
    fade_size = chunk_size // 10  # 200 — matches demix's fade_size derivation
    num_overlap = 2
    step = chunk_size // num_overlap  # 1000
    batch_size = 2

    sep = _make_separator_for_windowing(
        chunk_size=chunk_size,
        num_overlap=num_overlap,
        batch_size=batch_size,
        fade_size=fade_size,
    )

    # Track length: enough to produce >= 2 batches so the "first batch"
    # is genuinely the first and not also the last. Also must satisfy
    # length_init > 2 * border so the outer reflect-pad path runs (same
    # as the production config). border = chunk_size - step = 1000.
    length = 6 * step  # 6000 samples
    assert length > 2 * (chunk_size - step)

    # Intercept the divide step by monkeypatching torch.Tensor to
    # capture the counter before division — simplest: pre-compute what
    # the counter should look like by running the same per-chunk loop
    # manually. We instead rely on the fact that mock_model returns 1s,
    # so result == counter after accumulation, and result/counter == 1
    # wherever counter > 0. That means the output stem at every
    # in-range sample is exactly 1.0, regardless of windowing. The
    # windowing bug only manifests in `counter` before division.
    #
    # To inspect counter directly, we re-run the same windowing math
    # the method uses and compare to the post-fix helper.
    border = chunk_size - step
    pad_applied = length > 2 * border and border > 0
    padded = length + (2 * border if pad_applied else 0)

    windowing_array = sep._get_windowing_array(chunk_size, fade_size)
    counter = np.zeros((3, 2, padded), dtype=np.float32)

    # Simulate what the fixed loop does for batch 1 only.
    i = 0
    batch_starts = []
    while i < padded and len(batch_starts) < batch_size:
        batch_starts.append(i)
        i += step
    is_final_batch = i >= padded
    last_idx = len(batch_starts) - 1
    for j, start in enumerate(batch_starts):
        w = BanditSeparator._chunk_window(
            windowing_array,
            fade_size,
            skip_fade_in=(start == 0),
            skip_fade_out=(is_final_batch and j == last_idx),
        )
        seg_len = min(chunk_size, padded - start)
        counter[..., start : start + seg_len] += w[:seg_len].numpy()

    # At absolute position `step` (start of chunk 1):
    #   chunk 0 window[step] = 1 (middle plateau of chunk 0)
    #   chunk 1 window[0]    = 0 (start of chunk 1's fade-in ramp, post-fix)
    # Counter expected = 1. (Pre-fix would have been 2, because chunk 1's
    # fade-in was flattened to 1.)
    np.testing.assert_allclose(counter[..., step], 1.0, atol=1e-6)

    # Inside the overlap plateau [step + fade_size, chunk_size - fade_size),
    # both chunks contribute 1.0 — counter == 2.
    mid = step + (chunk_size - step - fade_size) // 2
    np.testing.assert_allclose(counter[..., mid], 2.0, atol=1e-6)

    # First fade_size samples [0, fade_size): only chunk 0 contributes,
    # and its fade-in is skipped (start == 0) — counter == 1.
    np.testing.assert_allclose(counter[..., :fade_size], 1.0, atol=1e-6)


def test_demix_reference_output_is_ones_with_mock_model():
    """End-to-end: a mock model returning ones should, through the full
    demix pipeline, yield stems that are ~1.0 at every in-range sample.
    This implicitly verifies that result/counter normalisation doesn't
    blow up, and that the outer-border crop is honoured.
    """
    chunk_size = 2000
    fade_size = chunk_size // 10
    num_overlap = 2
    batch_size = 2

    sep = _make_separator_for_windowing(
        chunk_size=chunk_size,
        num_overlap=num_overlap,
        batch_size=batch_size,
        fade_size=fade_size,
    )

    length = 6 * (chunk_size // num_overlap)
    mix = np.zeros((2, length), dtype=np.float32)

    stems = sep._demix_reference(mix)
    assert set(stems.keys()) == {"speech", "music", "sfx"}
    for name, arr in stems.items():
        assert arr.shape == (2, length), name
        np.testing.assert_allclose(arr, 1.0, atol=1e-6, err_msg=name)


def test_only_chunk_at_position_zero_skips_fade_in(monkeypatch):
    """Spy on _chunk_window. When _demix_reference runs a multi-batch
    track, skip_fade_in must be True exactly for the chunk at absolute
    position 0 and False for every other chunk — including chunks 1..N-1
    of batch 1, which the pre-fix code incorrectly flattened.
    """
    chunk_size = 2000
    fade_size = chunk_size // 10
    num_overlap = 2
    batch_size = 2
    step = chunk_size // num_overlap

    sep = _make_separator_for_windowing(
        chunk_size=chunk_size,
        num_overlap=num_overlap,
        batch_size=batch_size,
        fade_size=fade_size,
    )

    calls: list[tuple[bool, bool]] = []
    original = BanditSeparator._chunk_window

    def spy(windowing_array, fs, *, skip_fade_in, skip_fade_out):
        calls.append((skip_fade_in, skip_fade_out))
        return original(
            windowing_array,
            fs,
            skip_fade_in=skip_fade_in,
            skip_fade_out=skip_fade_out,
        )

    monkeypatch.setattr(BanditSeparator, "_chunk_window", staticmethod(spy))

    # Length chosen so there are > batch_size chunks in total AND at
    # least two batches, so the final-batch edge doesn't mask the
    # per-chunk fade-in decision in batch 1.
    length = 6 * step
    mix = np.zeros((2, length), dtype=np.float32)
    sep._demix_reference(mix)

    # Exactly one call with skip_fade_in=True — the chunk at start 0.
    assert sum(1 for fi, _ in calls if fi) == 1, (
        f"expected skip_fade_in=True for exactly one chunk, got {calls}"
    )

    # The first call corresponds to batch 1's first chunk, which IS the
    # chunk at start 0. All other calls must have skip_fade_in=False.
    first_fi, _ = calls[0]
    assert first_fi is True
    for fi, _ in calls[1:]:
        assert fi is False


def test_single_chunk_track_has_both_fades_skipped():
    """A track short enough to be a single chunk should have both
    fade-in (start == 0) and fade-out (only chunk is also final) skipped.
    Output through the mock-model pipeline should be uniformly 1.
    """
    chunk_size = 2000
    fade_size = chunk_size // 10
    num_overlap = 2
    batch_size = 2

    sep = _make_separator_for_windowing(
        chunk_size=chunk_size,
        num_overlap=num_overlap,
        batch_size=batch_size,
        fade_size=fade_size,
    )

    # Track shorter than 2 * border so outer-pad is NOT applied; single
    # chunk (length < chunk_size so seg_len < chunk_size). num_overlap=2
    # so border = chunk_size/2 = 1000. Choose length = 1500 so length
    # is NOT > 2*border (would need 2000+).
    length = 1500
    border = chunk_size - chunk_size // num_overlap
    assert length < 2 * border  # outer-pad path disabled
    mix = np.zeros((2, length), dtype=np.float32)

    stems = sep._demix_reference(mix)
    for name, arr in stems.items():
        assert arr.shape == (2, length), name
        np.testing.assert_allclose(arr, 1.0, atol=1e-6, err_msg=name)
