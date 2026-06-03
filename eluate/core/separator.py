# SPDX-License-Identifier: MIT
"""
Bandit v2 audio separator for music removal.

Uses ZFTurbo's Music-Source-Separation-Training framework
to separate audio into speech, music, and sound effects.
"""

from pathlib import Path
from typing import Callable, Dict, Optional, Protocol

import librosa
import numpy as np
import soundfile as sf
import torch

from eluate.utils.device import (
    clear_device_cache,
    configure_mps_settings,
    get_optimal_device,
)
from eluate.utils.telemetry import record_event

# Audio constants - Bandit v2 uses 48kHz
SAMPLE_RATE = 48000


class _StemWriter(Protocol):
    def write(self, data: np.ndarray) -> None: ...

    def close(self) -> None: ...


class _NullStemWriter:
    """Drop output blocks for stems that are not needed by this run."""

    def write(self, data: np.ndarray) -> None:
        return None

    def close(self) -> None:
        return None


class BanditSeparator:
    """
    Bandit v2 audio separator for documentary processing.

    Separates audio into three stems:
    - speech: Dialogue, narration, voice
    - music: Background music, score
    - sfx: Sound effects, ambient sounds

    Uses the ZFTurbo Music-Source-Separation-Training framework
    with chunked processing for long audio files.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
        arch: str = "bandit_v2",
        device_cache_clear_interval: int = 8,
    ):
        """
        Initialize the separator.

        Args:
            config_path: Path to model config YAML (uses default if None)
            checkpoint_path: Path to model checkpoint (.ckpt)
            device: Torch device (auto-detected if None)
            arch: ZFTurbo model-type string. Only "bandit_v2" is supported.
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.arch = arch
        self.device_cache_clear_interval = max(0, int(device_cache_clear_interval))

        if device is None:
            configure_mps_settings()
            self.device = get_optimal_device()
        else:
            self.device = device

        self._model = None
        self._config = None
        # Populated from config.audio.sample_rate after load. Always 48 kHz
        # for Bandit v2.
        self.model_sample_rate: int = SAMPLE_RATE

    @property
    def model(self):
        """Lazy-load the model on first access."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def config(self):
        """Lazy-load the config on first access."""
        if self._config is None:
            self._load_model()
        return self._config

    def _load_model(self):
        """Load Bandit v2 model from checkpoint."""
        from eluate._vendor.mss_training.utils.settings import get_model_from_config

        config_path = self.config_path
        checkpoint_path = self.checkpoint_path

        if config_path is None or checkpoint_path is None:
            from eluate.utils.paths import get_model_paths

            default_checkpoint, default_config = get_model_paths()
            if config_path is None:
                config_path = default_config
            if checkpoint_path is None:
                checkpoint_path = default_checkpoint

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {checkpoint_path}\n"
                f"Please run the install script to download the model."
            )

        model, config = get_model_from_config(self.arch, str(config_path))

        # weights_only=False is required by the Bandit v2 checkpoint format
        # (it contains pickled config objects, not just tensors). This means
        # torch.load will execute arbitrary code from the pickle — only load
        # checkpoints you trust. Eluate mitigates by: (1) fetching only from
        # the pinned Zenodo records in paths.py, (2) verifying SHA256 when a
        # digest is declared in CHECKPOINT_SHA256. See README "Security".
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {
                k.replace("model.", "", 1): v
                for k, v in state_dict.items()
                if k.startswith("model.")
            }

        model.load_state_dict(state_dict)
        model = model.float()
        model = model.to(self.device)
        model.eval()

        self._model = model
        self._config = config
        # Cache the model's native sample rate — may differ from pipeline SR.
        self.model_sample_rate = int(config.audio.sample_rate)

    def _load_audio(self, audio_path: Path) -> np.ndarray:
        """
        Load audio as float32 stereo at the model's native sample rate,
        shape (channels, samples).

        Fast path: soundfile (zero resampling, no codec overhead) when the
        file's sample rate already matches. Otherwise resample with librosa.
        """
        target_sr = self.model_sample_rate
        try:
            data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
            if sr == target_sr:
                mix = np.ascontiguousarray(data.T)  # (samples, channels) -> (channels, samples)
                if mix.shape[0] == 1:
                    mix = np.concatenate([mix, mix], axis=0)
                elif mix.shape[0] > 2:
                    mix = mix[:2]
                return mix
        except Exception:
            pass

        mix, _ = librosa.load(audio_path, sr=target_sr, mono=False)
        if len(mix.shape) == 1:
            mix = np.stack([mix, mix], axis=0)
        return mix

    def _get_windowing_array(self, chunk_size: int, fade_size: int) -> torch.Tensor:
        """Create windowing array with linear fade-in/fade-out for overlap-add."""
        window = torch.ones(chunk_size)
        window[:fade_size] = torch.linspace(0, 1, fade_size)
        window[-fade_size:] = torch.linspace(1, 0, fade_size)
        return window

    @staticmethod
    def _chunk_window(
        windowing_array: torch.Tensor,
        fade_size: int,
        *,
        skip_fade_in: bool,
        skip_fade_out: bool,
    ) -> torch.Tensor:
        """Per-chunk overlap-add window.

        Returns the shared ramp-in/ramp-out template when neither flag is
        set (the modal case for interior chunks) — callers must treat the
        result as read-only. When a flag is set, returns a fresh clone
        with the corresponding ramp flattened to 1. ``skip_fade_in`` is
        True only for the chunk whose absolute start position is 0 (the
        true start of the track). ``skip_fade_out`` is True only for the
        last chunk that reaches the track end. Both may be True for a
        single-chunk track.
        """
        if not skip_fade_in and not skip_fade_out:
            return windowing_array
        window = windowing_array.clone()
        if skip_fade_in:
            window[:fade_size] = 1
        if skip_fade_out:
            window[-fade_size:] = 1
        return window

    def _demix_reference(
        self,
        mix: np.ndarray,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Reference chunked-inference implementation, kept as the numerical
        parity oracle for tests. **Test-only — do not call from runtime
        paths.** Allocates two full-track float32 accumulators, which is
        exactly the memory behavior the streaming path is designed to avoid.

        Mirrors vendor/mss-training/utils/model_utils.py::demix.
        """
        config = self.config
        model = self.model

        # Read inference params from config (same as vendor demix() generic mode)
        if hasattr(config.inference, "chunk_size"):
            chunk_size = config.inference.chunk_size
        else:
            chunk_size = config.audio.chunk_size
        num_overlap = config.inference.num_overlap
        batch_size = config.inference.batch_size

        fade_size = chunk_size // 10
        step = chunk_size // num_overlap
        border = chunk_size - step

        mix_tensor = torch.tensor(mix, dtype=torch.float32)
        length_init = mix_tensor.shape[-1]
        windowing_array = self._get_windowing_array(chunk_size, fade_size)

        if length_init > 2 * border and border > 0:
            mix_tensor = torch.nn.functional.pad(mix_tensor, (border, border), mode="reflect")

        stems = config.training.instruments
        num_stems = len(stems)
        result = torch.zeros((num_stems,) + mix_tensor.shape, dtype=torch.float32)
        counter = torch.zeros((num_stems,) + mix_tensor.shape, dtype=torch.float32)

        i = 0
        batch_data = []
        batch_locations = []
        processed_batches = 0

        with torch.inference_mode():
            while i < mix_tensor.shape[1]:
                part = mix_tensor[:, i : i + chunk_size].to(self.device)
                chunk_len = part.shape[-1]

                pad_mode = "reflect" if chunk_len > chunk_size // 2 else "constant"
                part = torch.nn.functional.pad(
                    part, (0, chunk_size - chunk_len), mode=pad_mode, value=0
                )

                batch_data.append(part)
                batch_locations.append((i, chunk_len))
                i += step

                if len(batch_data) >= batch_size or i >= mix_tensor.shape[1]:
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr).cpu()

                    # Per-chunk windowing: skip fade-in only at the true track
                    # start (start == 0) and fade-out only on the chunk that
                    # reaches the track end. Diverges from vendor's
                    # `if i - step == 0` (model_utils.py:126), which is
                    # unreachable at batch_size > 1 and leaves chunks 1..N of
                    # batch 1 with an intact ramp — see docs/bench/memory-benchmark.md.
                    is_final_batch = i >= mix_tensor.shape[1]
                    last_idx = len(batch_locations) - 1
                    for j, (start, seg_len) in enumerate(batch_locations):
                        window = self._chunk_window(
                            windowing_array,
                            fade_size,
                            skip_fade_in=(start == 0),
                            skip_fade_out=(is_final_batch and j == last_idx),
                        )
                        result[..., start : start + seg_len] += (
                            x[j, ..., :seg_len] * window[..., :seg_len]
                        )
                        counter[..., start : start + seg_len] += window[..., :seg_len]

                    batch_data.clear()
                    batch_locations.clear()
                    processed_batches += 1

                    if progress_callback:
                        progress_callback(min(i / mix_tensor.shape[1], 1.0))

        estimated_sources: np.ndarray = (result / counter).numpy()
        np.nan_to_num(estimated_sources, copy=False, nan=0.0)

        if length_init > 2 * border and border > 0:
            estimated_sources = estimated_sources[..., border:-border]

        return {name: estimated_sources[idx] for idx, name in enumerate(stems)}

    def _open_stem_writers(
        self,
        workspace_dir: Path,
        channels: int,
        persisted_stems: Optional[set[str]] = None,
    ) -> tuple[Dict[str, Path], Dict[str, _StemWriter]]:
        """Create temp WAV writers for requested model stems.

        Returns (temp_paths, writers). Callers must close each writer
        (and own the resulting files). Unrequested stems get a null writer
        so model output shapes stay unchanged without paying temp-file I/O.
        """
        stems = list(self.config.training.instruments)
        if persisted_stems is None:
            persisted_stems = set(stems)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        temp_paths = {
            name: workspace_dir / f"_eluate_stem_{name}.wav"
            for name in stems
            if name in persisted_stems
        }
        writers: Dict[str, _StemWriter] = {}
        for name in stems:
            if name in temp_paths:
                writers[name] = sf.SoundFile(
                    str(temp_paths[name]),
                    mode="w",
                    samplerate=self.model_sample_rate,
                    channels=channels,
                    format="WAV",
                    subtype="FLOAT",
                )
            else:
                writers[name] = _NullStemWriter()
        return temp_paths, writers

    def _demix_streaming(
        self,
        mix: np.ndarray,
        writers: Dict[str, _StemWriter],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        """
        Streaming chunked inference: writes finalized samples into the
        provided per-stem writers as the loop advances, so the resident
        accumulator stays O(chunk_size + headroom) regardless of track length.

        The loop advances `i` sequentially; after a batch, samples at
        absolute padded-space positions `< i` are final (no future chunk
        can write below `i`, which is the next chunk's start). We flush
        those into ``writers``, shift the ring buffer, zero the freshly
        exposed region, and keep going. Outer-border samples are skipped
        at write time so each writer receives only the real region.

        Callers own writer creation and closure; see ``_open_stem_writers``.

        Args:
            mix: Audio array of shape (channels, samples) at model_sample_rate
            writers: Per-stem open SoundFile writers (keyed by stem name in
                     the same order as config.training.instruments)
            progress_callback: Called with progress fraction (0.0–1.0) per batch
        """
        config = self.config
        model = self.model

        if hasattr(config.inference, "chunk_size"):
            chunk_size = config.inference.chunk_size
        else:
            chunk_size = config.audio.chunk_size
        num_overlap = config.inference.num_overlap
        batch_size = config.inference.batch_size

        fade_size = chunk_size // 10
        step = chunk_size // num_overlap
        border = chunk_size - step

        # Share memory with the input ndarray rather than copy (torch.tensor
        # makes a copy; from_numpy is a view). Ensure C-contiguous float32 so
        # the view is valid without an implicit copy.
        if mix.dtype != np.float32:
            mix = mix.astype(np.float32, copy=False)
        assert mix.flags["C_CONTIGUOUS"], "mix must be C-contiguous; _load_audio guarantees this"
        mix_base = torch.from_numpy(mix)
        length_init = mix_base.shape[-1]
        channels = mix_base.shape[0]
        windowing_array = self._get_windowing_array(chunk_size, fade_size)

        pad_applied = length_init > 2 * border and border > 0
        if pad_applied:
            # Precompute the two small border-reflection regions (border
            # samples each, ~1.5 MB at border=192000 stereo float32) rather
            # than allocating a full padded copy of the track. torch's
            # reflect pad convention: padded[k] = mix[pad_left - k] for
            # k in [0, pad_left); padded[L + pad_left + k] = mix[L - 2 - k]
            # for k in [0, pad_right).
            left_pad = mix_base[:, 1 : border + 1].flip(-1).contiguous()
            right_pad = (
                mix_base[:, length_init - border - 1 : length_init - 1].flip(-1).contiguous()
            )
            real_start = border
            real_end = border + length_init
            padded_length = length_init + 2 * border
        else:
            left_pad = None
            right_pad = None
            real_start = 0
            real_end = length_init
            padded_length = length_init

        def _padded_chunk(start: int, want: int) -> tuple[torch.Tensor, int]:
            """Build a chunk (channels, chunk_len) from the virtually padded
            mix starting at padded-space position `start`. `chunk_len` is
            `min(want, padded_length - start)` and may be < want near the
            end; callers pad the remainder as before.
            """
            end = min(start + want, padded_length)
            chunk_len = end - start
            if not pad_applied:
                return mix_base[:, start:end], chunk_len
            # Reached only when pad_applied; left_pad and right_pad are set
            # above. Narrow for the type checker with an explicit raise so
            # ``python -O`` can't strip the check like an ``assert`` would.
            if left_pad is None or right_pad is None:
                raise RuntimeError("left_pad/right_pad unset while pad_applied")
            parts: list[torch.Tensor] = []
            pos = start
            if pos < real_start:
                take = min(end, real_start)
                parts.append(left_pad[:, pos:take])
                pos = take
            if pos < real_end and pos < end:
                take = min(end, real_end)
                parts.append(mix_base[:, pos - real_start : take - real_start])
                pos = take
            if pos < end:
                parts.append(right_pad[:, pos - real_end : end - real_end])
            if len(parts) == 1:
                return parts[0], chunk_len
            return torch.cat(parts, dim=-1), chunk_len

        stems = list(config.training.instruments)
        num_stems = len(stems)

        # Ring size: fit one batch of writes plus a chunk of headroom,
        # with a 60-second floor so small-batch configs still have slack.
        min_ring = batch_size * step + chunk_size + step
        window_samples = max(
            int(self.model_sample_rate * 60),
            min_ring * 2,
        )

        result = torch.zeros((num_stems, channels, window_samples), dtype=torch.float32)
        counter = torch.zeros_like(result)

        record_event(
            "demix.config",
            {
                "arch": self.arch,
                "chunk_size": chunk_size,
                "num_overlap": num_overlap,
                "batch_size": batch_size,
                "fade_size": fade_size,
                "step": step,
                "border": border,
                "num_stems": num_stems,
                "channels": channels,
                "model_sample_rate": self.model_sample_rate,
                "device": str(self.device),
                "length_init": length_init,
                "padded_length": padded_length,
                "window_samples": window_samples,
                "streaming": True,
            },
        )

        ring_base_abs = 0  # invariant: ring_base_abs == flushed_abs
        flushed_abs = 0

        def flush_up_to(finalize_abs: int) -> None:
            nonlocal ring_base_abs, flushed_abs
            if finalize_abs <= flushed_abs:
                return
            new_final = finalize_abs - flushed_abs

            r = result[..., :new_final]
            c = counter[..., :new_final]
            # Clamp counter to avoid 0/0; regions with c==0 get zeroed.
            safe_c = torch.where(c > 0, c, torch.ones_like(c))
            normalized = (r / safe_c).numpy()
            zero_mask = (c == 0).numpy()
            if zero_mask.any():
                normalized[zero_mask] = 0.0
            np.nan_to_num(normalized, copy=False, nan=0.0)

            write_start = max(flushed_abs, real_start)
            write_end = min(flushed_abs + new_final, real_end)
            if write_end > write_start:
                local_start = write_start - flushed_abs
                local_end = write_end - flushed_abs
                for stem_idx, name in enumerate(stems):
                    block = normalized[stem_idx, :, local_start:local_end].T
                    writers[name].write(np.ascontiguousarray(block))

            # Slide ring forward by new_final.
            tail = window_samples - new_final
            if tail > 0:
                result[..., :tail] = result[..., new_final : new_final + tail].clone()
                counter[..., :tail] = counter[..., new_final : new_final + tail].clone()
                result[..., tail:] = 0
                counter[..., tail:] = 0
            else:
                result.zero_()
                counter.zero_()
            flushed_abs += new_final
            ring_base_abs += new_final

        i = 0
        batch_data: list[torch.Tensor] = []
        batch_locations: list[tuple[int, int]] = []
        processed_batches = 0

        with torch.inference_mode():
            while i < padded_length:
                part_cpu, chunk_len = _padded_chunk(i, chunk_size)
                part = part_cpu.to(self.device)

                pad_mode = "reflect" if chunk_len > chunk_size // 2 else "constant"
                part = torch.nn.functional.pad(
                    part, (0, chunk_size - chunk_len), mode=pad_mode, value=0
                )

                batch_data.append(part)
                batch_locations.append((i, chunk_len))
                i += step

                if len(batch_data) >= batch_size or i >= padded_length:
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr).cpu()

                    # See _demix_reference for rationale on per-chunk
                    # windowing vs the vendor's batch-level condition.
                    is_final_batch = i >= padded_length
                    last_idx = len(batch_locations) - 1
                    for j, (start, seg_len) in enumerate(batch_locations):
                        window = self._chunk_window(
                            windowing_array,
                            fade_size,
                            skip_fade_in=(start == 0),
                            skip_fade_out=(is_final_batch and j == last_idx),
                        )
                        ro = start - ring_base_abs
                        result[..., ro : ro + seg_len] += (
                            x[j, ..., :seg_len] * window[..., :seg_len]
                        )
                        counter[..., ro : ro + seg_len] += window[..., :seg_len]

                    batch_data.clear()
                    batch_locations.clear()
                    processed_batches += 1

                    flush_up_to(min(i, padded_length))
                    clear_interval = getattr(self, "device_cache_clear_interval", 8)
                    if clear_interval and processed_batches % clear_interval == 0:
                        clear_device_cache(self.device)

                    if progress_callback:
                        progress_callback(min(i / padded_length, 1.0))

        # Any remainder (shouldn't happen after the i>=padded_length
        # branch flushes, but belt-and-braces):
        flush_up_to(padded_length)

    def produce_outputs(
        self,
        audio_path: Path,
        *,
        workspace_dir: Path,
        mix_path: Optional[Path] = None,
        speech_path: Optional[Path] = None,
        sfx_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        """
        Run separation once and write only the requested outputs.

        ``mix_path`` receives the music-removed mix (speech + sfx, peak
        normalised). ``speech_path`` and ``sfx_path`` receive the raw
        stems (PCM_16 WAV at the model sample rate). Any of the three
        may be ``None`` to skip that output. The temp float-WAV stems
        used during separation are written under ``workspace_dir`` and
        unlinked before return.

        At least one of ``mix_path``, ``speech_path``, ``sfx_path`` must
        be provided.
        """
        if mix_path is None and speech_path is None and sfx_path is None:
            raise ValueError(
                "produce_outputs requires at least one of mix_path, speech_path, or sfx_path"
            )

        audio_path = Path(audio_path)
        workspace_dir = Path(workspace_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)

        mix = self._load_audio(audio_path)
        channels = mix.shape[0]
        stems = list(self.config.training.instruments)
        sfx_key = "sfx" if "sfx" in stems else ("effects" if "effects" in stems else None)
        persisted_stems: set[str] = set()
        if mix_path is not None or speech_path is not None:
            persisted_stems.add("speech")
        if sfx_key is not None and (mix_path is not None or sfx_path is not None):
            persisted_stems.add(sfx_key)

        stem_temp_paths, writers = self._open_stem_writers(
            workspace_dir,
            channels,
            persisted_stems=persisted_stems,
        )
        try:
            try:
                self._demix_streaming(mix, writers, progress_callback)
            finally:
                for w in writers.values():
                    w.close()
            del mix

            speech_temp = stem_temp_paths.get("speech")
            sfx_temp = stem_temp_paths.get(sfx_key) if sfx_key else None

            if mix_path is not None:
                if speech_temp is None:
                    raise RuntimeError("Model did not produce a speech stem; cannot write mix_path")
                mix_path = Path(mix_path)
                mix_path.parent.mkdir(parents=True, exist_ok=True)
                self._stream_mix_normalize_write(
                    speech_path=speech_temp,
                    effects_path=sfx_temp,
                    output_path=mix_path,
                    output_sample_rate=SAMPLE_RATE,
                )

            if speech_path is not None:
                if speech_temp is None:
                    raise RuntimeError(
                        "Model did not produce a speech stem; cannot write speech_path"
                    )
                speech_path = Path(speech_path)
                speech_path.parent.mkdir(parents=True, exist_ok=True)
                self._recode_wav(speech_temp, speech_path, SAMPLE_RATE)

            if sfx_path is not None:
                if sfx_temp is None:
                    raise RuntimeError("Model did not produce an SFX stem; cannot write sfx_path")
                sfx_path = Path(sfx_path)
                sfx_path.parent.mkdir(parents=True, exist_ok=True)
                self._recode_wav(sfx_temp, sfx_path, SAMPLE_RATE)
        finally:
            for temp_path in stem_temp_paths.values():
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except OSError:
                    pass

        if progress_callback:
            progress_callback(1.0)

    @staticmethod
    def _recode_wav(
        src_path: Path,
        dst_path: Path,
        sample_rate: int,
        block_seconds: int = 10,
    ) -> None:
        """Stream-copy a WAV file with soundfile's default subtype (PCM_16)."""
        with sf.SoundFile(str(src_path)) as src:
            blocksize = sample_rate * block_seconds
            with sf.SoundFile(
                str(dst_path),
                mode="w",
                samplerate=sample_rate,
                channels=src.channels,
                format="WAV",
            ) as dst:
                for block in src.blocks(blocksize=blocksize, dtype="float32"):
                    dst.write(block)

    @staticmethod
    def _stream_mix_normalize_write(
        speech_path: Path,
        effects_path: Optional[Path],
        output_path: Path,
        output_sample_rate: int,
        block_seconds: int = 10,
    ) -> None:
        """Two-pass streaming mix + peak-normalize + write.

        Pass 1 scans speech (+effects) in sync to find the peak magnitude.
        Pass 2 sums again, applies the gain, writes blocks to the output.
        Resident memory: a few blocks of float32 audio.

        The two passes re-read the temp stem WAVs from disk. This is
        intentional: the alternative — accumulating the mix peak during
        ``_demix_streaming``'s flush — would teach the numerically
        sensitive demix loop which stems sum into the output mix, coupling
        it to a concern it has no business knowing. The re-read is a
        sequential scan of local temp files (~2 s even on an 84-min input),
        a sub-1% slice of a multi-minute inference run, so the coupling
        isn't worth the marginal I/O. Keep it two-pass.
        """
        peak = 0.0
        with sf.SoundFile(str(speech_path)) as s_src:
            blocksize = output_sample_rate * block_seconds
            if effects_path is not None:
                with sf.SoundFile(str(effects_path)) as e_src:
                    for s_block, e_block in zip(
                        s_src.blocks(blocksize=blocksize, dtype="float32"),
                        e_src.blocks(blocksize=blocksize, dtype="float32"),
                    ):
                        mixed = s_block + e_block
                        peak = max(peak, float(np.abs(mixed).max()))
            else:
                for s_block in s_src.blocks(blocksize=blocksize, dtype="float32"):
                    peak = max(peak, float(np.abs(s_block).max()))

        gain = 0.95 / peak if peak > 1.0 else 1.0

        with sf.SoundFile(str(speech_path)) as s_src:
            blocksize = output_sample_rate * block_seconds
            with sf.SoundFile(
                str(output_path),
                mode="w",
                samplerate=output_sample_rate,
                channels=s_src.channels,
                format="WAV",
            ) as dst:
                if effects_path is not None:
                    with sf.SoundFile(str(effects_path)) as e_src:
                        for s_block, e_block in zip(
                            s_src.blocks(blocksize=blocksize, dtype="float32"),
                            e_src.blocks(blocksize=blocksize, dtype="float32"),
                        ):
                            mixed = s_block + e_block
                            if gain != 1.0:
                                mixed = mixed * gain
                            dst.write(mixed)
                else:
                    for s_block in s_src.blocks(blocksize=blocksize, dtype="float32"):
                        if gain != 1.0:
                            s_block = s_block * gain
                        dst.write(s_block)


def check_model_available() -> bool:
    """Check if the default Bandit v2 model checkpoint is available."""
    from eluate.utils.paths import get_model_paths

    checkpoint_path, _ = get_model_paths()
    return checkpoint_path.exists()
