# SPDX-License-Identifier: MIT
"""
Eluate public Python API.

This module is the entire stable contract: ``elute``, ``Session``,
``Result``, ``EluateError``. Everything else under ``eluate.*``
(``eluate.core``, ``eluate.utils``, ``eluate.ui``) is internal and may
change in any release.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Tuple, Union

if TYPE_CHECKING:
    from .core.pipeline import EluatePipeline

from .core.pipeline import resolve_output_paths
from .utils.paths import CHECKPOINT_KEYS
from .utils.preflight import PreflightConfig, run_preflight
from .utils.validators import (
    DurationOutOfRange as _PipelineDurationOutOfRange,
)
from .utils.validators import (
    InsufficientDiskSpace as _PipelineInsufficientDiskSpace,
)

logger = logging.getLogger("eluate.api")

_VALID_OUTPUTS: Tuple[str, ...] = ("video", "speech", "sfx")
_VALID_DEVICES: Tuple[str, ...] = ("cuda", "mps", "cpu")
_VALID_CHECKPOINTS: Tuple[str, ...] = tuple(CHECKPOINT_KEYS)

ProgressCallback = Callable[[float, str], None]

# Equal-thirds mapping from per-stage 0→100 progress to a monotonic global
# fraction in [0, 1]. Order matches EluatePipeline's stage sequence so a
# caller wiring tqdm/Rich/etc. sees fractions strictly non-decreasing
# across the extract → separate → compile boundary.
_STAGE_BOUNDS: Tuple[Tuple[str, float, float], ...] = (
    ("extract", 0.0, 0.25),
    ("load_model", 0.25, 0.25),
    ("separate", 0.50, 0.25),
    ("compile", 0.75, 0.25),
)
_STAGE_INDEX = {name: (offset, weight) for name, offset, weight in _STAGE_BOUNDS}


class _ProgressAdapter:
    """Public ``on_progress`` shim over the pipeline's five private callbacks.

    The pipeline emits per-stage progress in 0–100 plus stage_start/
    stage_complete/error/video_info events. The public API contract is one
    callable receiving a global fraction in [0, 1] and a stage label, with
    monotonic non-decreasing fractions and a final 1.0 on success. This
    class folds the former into the latter.
    """

    def __init__(self, on_progress: Optional[ProgressCallback]) -> None:
        self._cb = on_progress
        self._current_stage: str = ""
        self._stage_offset = 0.0
        self._stage_weight = 1.0

    def on_stage_start(self, stage_id: str, total: Optional[float] = None) -> None:
        self._current_stage = stage_id
        bounds = _STAGE_INDEX.get(stage_id)
        if bounds is not None:
            self._stage_offset, self._stage_weight = bounds
        # Emit a 0% tick at the stage boundary so UIs can flip stages and
        # start animating immediately, even before the first real progress
        # event lands. Model load between extract and separate can take many
        # seconds with no work-progress signal — without this, the UI looks
        # frozen on the previous stage.
        if self._cb is not None:
            self._cb(self._stage_offset, self._current_stage)

    def on_progress(self, progress: float, detail: str = "") -> None:
        if self._cb is None:
            return
        local = max(0.0, min(1.0, float(progress) / 100.0))
        fraction = self._stage_offset + local * self._stage_weight
        self._cb(max(0.0, min(1.0, fraction)), self._current_stage)

    def on_stage_complete(self, stage_id: str) -> None:
        return None

    def on_error(self, stage_id: str, message: str) -> None:
        return None

    def on_video_info(self, title: str, duration: str) -> None:
        return None

    def emit_final(self) -> None:
        if self._cb is None:
            return
        self._cb(1.0, self._current_stage or "complete")


# Sentinel distinguishing "caller did not pass this kwarg" from
# "caller explicitly passed False/None/etc." on Session.elute(). Keeps
# the public type as bool/PathLike instead of Optional[...] so callers
# don't have to think about None semantics.
class _Unset:
    def __repr__(self) -> str:
        return "<unset>"


_UNSET = _Unset()


def _validate_outputs(outputs: Iterable[str]) -> Tuple[str, ...]:
    """Coerce ``outputs`` to a validated tuple drawn from ``{"video","speech","sfx"}``.

    Rejects ``"music"`` explicitly: eluate is a removal tool, not a stem
    separator, and the music stem is never exposed.
    """
    if isinstance(outputs, str):
        raise TypeError("outputs must be a tuple of strings, not a single string")
    try:
        coerced = tuple(outputs)
    except TypeError as exc:
        raise TypeError(
            f"outputs must be an iterable of strings, got {type(outputs).__name__}"
        ) from exc
    if not coerced:
        raise ValueError("outputs must not be empty; valid outputs are 'video', 'speech', 'sfx'")
    if "music" in coerced:
        raise ValueError(
            "eluate does not expose the music stem; valid outputs are 'video', 'speech', 'sfx'"
        )
    invalid = [o for o in coerced if o not in _VALID_OUTPUTS]
    if invalid:
        raise ValueError(
            f"unknown output(s) {invalid!r}; valid outputs are 'video', 'speech', 'sfx'"
        )
    return coerced


def _validate_checkpoint(checkpoint: str) -> str:
    """Validate the ``checkpoint`` kwarg string against ``CHECKPOINT_KEYS``."""
    if not isinstance(checkpoint, str) or checkpoint not in _VALID_CHECKPOINTS:
        raise ValueError(
            f"invalid checkpoint {checkpoint!r}; valid options are {sorted(_VALID_CHECKPOINTS)}"
        )
    return checkpoint


def _validate_device(device: Optional[str]) -> Optional[str]:
    """Validate the ``device`` kwarg string.

    ``None`` means auto-detect (CUDA > MPS > CPU). Any other value must
    be one of ``"cuda" | "mps" | "cpu"``. Availability of the named
    device is checked later when the model loads — this function only
    rejects the string itself.
    """
    if device is None:
        return None
    if not isinstance(device, str) or device not in _VALID_DEVICES:
        raise ValueError(f"invalid device {device!r}; valid options are None, 'cuda', 'mps', 'cpu'")
    return device


def _check_collisions(targets: dict, *, overwrite: bool) -> None:
    """Raise ``FileExistsError`` if any target path already exists.

    Collision policy is independent of ``force``: ``force=True`` only
    bypasses the duration validator and disk-space precheck, never the
    overwrite guard. The error message identifies the colliding path so
    callers can act on it without parsing.
    """
    if overwrite:
        return
    for path in targets.values():
        if path.exists():
            raise FileExistsError(
                f"Output path already exists: {path}. "
                "Pass overwrite=True to replace existing files."
            )


class EluateError(Exception):
    """Base exception for all eluate-specific failures.

    Subclasses (:class:`DurationOutOfRange`, :class:`InsufficientDiskSpace`,
    :class:`ModelNotInstalledError`, :class:`FFmpegNotFoundError`) cover the
    failure modes callers typically want to react to specifically. Catching
    ``EluateError`` catches all of them.

    Stdlib exceptions (``FileNotFoundError``, ``PermissionError``,
    ``KeyboardInterrupt``) propagate **unwrapped** — eluate does not
    re-package errors that already have the right Python name.
    """


class DurationOutOfRange(EluateError):
    """Input video duration is zero, negative, or above the supported cap.

    Pass ``force=True`` to bypass the cap when running on long inputs
    intentionally.
    """


class InsufficientDiskSpace(EluateError):
    """Target filesystem does not have enough free space for the run.

    Pass ``force=True`` to skip the precheck if you are willing to risk
    a mid-run write failure.
    """


class ModelNotInstalledError(EluateError):
    """Required checkpoint is not present on this machine.

    The API does not auto-download checkpoints. Run ``eluate setup`` to
    download them once, then re-run.
    """


class FFmpegNotFoundError(EluateError):
    """FFmpeg is required but was not found on ``PATH``.

    Eluate shells out to FFmpeg for audio extraction and muxing. Install
    it (``brew install ffmpeg`` on macOS, ``apt-get install ffmpeg`` on
    Debian/Ubuntu) and ensure it is on ``PATH``, then re-run.
    """


def _raise_translated(pipeline_result) -> None:
    """Re-raise a failed ``PipelineResult`` as the right public exception.

    Translation table:

    - low-level ``DurationOutOfRange`` → :class:`DurationOutOfRange`
    - low-level ``InsufficientDiskSpace`` → :class:`InsufficientDiskSpace`
    - ``FileNotFoundError`` / ``PermissionError`` → re-raised unwrapped
    - anything else → :class:`EluateError`

    The original exception is chained via ``__cause__`` (``raise ... from
    exc``) so traceback debuggers and ``logging.exception`` see both
    layers.
    """
    exc = pipeline_result.error_exc
    msg = pipeline_result.error or "Eluate pipeline failed"

    if isinstance(exc, _PipelineDurationOutOfRange):
        raise DurationOutOfRange(msg) from exc
    if isinstance(exc, _PipelineInsufficientDiskSpace):
        raise InsufficientDiskSpace(msg) from exc
    if isinstance(exc, (FileNotFoundError, PermissionError)):
        raise exc
    raise EluateError(msg)


@dataclass(frozen=True)
class Result:
    """Outcome of a successful eluate run.

    ``None`` on a stem field means *the caller did not request that
    output*, never *a failure occurred* — failures raise.
    """

    input: Path
    video: Optional[Path]
    speech: Optional[Path]
    sfx: Optional[Path]
    duration: float
    processing_time: float


PathLike = Union[str, Path]


class Session:
    """Stateful eluate session that amortises model loading across calls.

    The model is loaded on the first ``.elute()`` call and reused for every
    subsequent call within the session, so batch loops finish in minutes
    rather than hours. Use as a context manager to make the lifetime
    explicit::

        with eluate.Session() as s:
            for v in videos:
                s.elute(v)

    Constructor kwargs (``output_dir``, ``overwrite``, ``force``) act as
    session-level defaults. ``Session.elute()`` accepts the same kwargs
    as per-call overrides that win when both are set.

    A ``Session`` is **not** safe for concurrent use: ``elute()`` mutates
    shared pipeline state (force, codec, progress callbacks) per call, so
    calls must be serialised. Use one ``Session`` per thread for parallel
    work. ``close()`` (called automatically on ``__exit__``) releases the
    model and frees accelerator memory; the session stays usable after
    ``close()`` — the next ``elute()`` reloads the model.
    """

    def __init__(
        self,
        *,
        output_dir: Optional[PathLike] = None,
        overwrite: bool = False,
        force: bool = False,
        audio_codec: str = "aac",
        audio_bitrate: str = "256k",
        on_progress: Optional[ProgressCallback] = None,
        device: Optional[str] = None,
        checkpoint: str = "multi",
    ) -> None:
        self._default_output_dir: Optional[Path] = (
            Path(output_dir).expanduser() if output_dir is not None else None
        )
        self._default_overwrite = overwrite
        self._default_force = force
        self._default_audio_codec = audio_codec
        self._default_audio_bitrate = audio_bitrate
        self._default_on_progress = on_progress
        # device is session-level only: it determines which weights load
        # and onto which hardware. Changing it mid-session would require
        # a model reload — disallowed; see Session.elute().
        self._device: Optional[str] = _validate_device(device)
        # checkpoint is likewise session-locked — it selects which
        # bandit-v2 language variant's weights load.
        self._checkpoint: str = _validate_checkpoint(checkpoint)

        self._preflight: Optional[PreflightConfig] = None
        # Imported lazily to keep ``import eluate`` light; the annotation is
        # a string under ``from __future__ import annotations`` so the name
        # is only needed by type checkers (see the TYPE_CHECKING import).
        self._pipeline: Optional["EluatePipeline"] = None

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
        return None

    def close(self) -> None:
        """Release the model and free accelerator memory.

        Idempotent. The session stays usable: the next ``elute()`` reloads
        the model. Called automatically when used as a context manager.
        """
        if self._pipeline is not None:
            self._pipeline.close()

    def elute(
        self,
        input: PathLike,
        *,
        outputs: Iterable[str] = ("video",),
        output_dir: Union[PathLike, _Unset] = _UNSET,
        overwrite: Union[bool, _Unset] = _UNSET,
        force: Union[bool, _Unset] = _UNSET,
        audio_codec: Union[str, _Unset] = _UNSET,
        audio_bitrate: Union[str, _Unset] = _UNSET,
        on_progress: Union[Optional[ProgressCallback], _Unset] = _UNSET,
        device: Union[Optional[str], _Unset] = _UNSET,
        checkpoint: Union[str, _Unset] = _UNSET,
    ) -> Result:
        """Run eluate on a single video and return a :class:`Result`.

        ``outputs`` selects which artifacts to produce — any combination
        of ``"video"``, ``"speech"``, ``"sfx"``. ``"music"`` is rejected.

        ``output_dir`` directs where outputs land (default CWD; created
        if missing). ``overwrite=False`` (default) raises
        ``FileExistsError`` if any requested output path already exists.
        ``force=True`` bypasses the duration validator and disk-space
        precheck (matches CLI ``--force``). Per-call values override
        session-level defaults.
        """
        if not isinstance(device, _Unset):
            raise TypeError(
                "device cannot be set per-call on Session.elute(); it is "
                "locked at Session construction because changing it would "
                "require reloading the model. Pass device=... to Session()."
            )
        if not isinstance(checkpoint, _Unset):
            raise TypeError(
                "checkpoint cannot be set per-call on Session.elute(); it is "
                "locked at Session construction because changing it would "
                "require reloading the model. Pass checkpoint=... to Session()."
            )

        validated_outputs = _validate_outputs(outputs)

        resolved_output_dir = self._resolve_output_dir(output_dir)
        resolved_overwrite = self._resolve_bool(overwrite, self._default_overwrite)
        resolved_force = self._resolve_bool(force, self._default_force)
        resolved_audio_codec = self._resolve_value(audio_codec, self._default_audio_codec)
        resolved_audio_bitrate = self._resolve_value(audio_bitrate, self._default_audio_bitrate)
        resolved_on_progress = self._resolve_value(on_progress, self._default_on_progress)

        input_path = Path(input).expanduser().resolve()

        # Stdlib FileNotFoundError propagates unwrapped — callers reading
        # ``except FileNotFoundError`` should not need to know about
        # ``EluateError``.
        if not input_path.exists():
            raise FileNotFoundError(f"Video file not found: {input_path}")

        # Create the output directory before resolving target paths so a
        # missing-but-creatable dir is not flagged as a collision.
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

        targets = resolve_output_paths(input_path.stem, resolved_output_dir, validated_outputs)
        # Collision policy is independent of force.
        _check_collisions(targets, overwrite=resolved_overwrite)

        pipeline = self._ensure_pipeline(resolved_output_dir)
        # Per-call overrides flow through pipeline state. These attributes
        # are read at the start of each process() / _stage_compile() call,
        # so mutation is safe across reused sessions.
        pipeline.force = resolved_force
        pipeline.audio_codec = resolved_audio_codec
        pipeline.audio_bitrate = resolved_audio_bitrate

        adapter = _ProgressAdapter(resolved_on_progress)
        # Always reassign — a previous call's adapter must not leak when
        # the current call passes on_progress=None.
        pipeline.on_stage_start = adapter.on_stage_start
        pipeline.on_progress = adapter.on_progress
        pipeline.on_stage_complete = adapter.on_stage_complete
        pipeline.on_error = adapter.on_error
        pipeline.on_video_info = adapter.on_video_info

        pipeline_result = pipeline.process(
            input_path,
            output_path=targets.get("video"),
            outputs=validated_outputs,
        )

        if not pipeline_result.success:
            _raise_translated(pipeline_result)

        adapter.emit_final()

        duration = float(pipeline_result.video_info.duration) if pipeline_result.video_info else 0.0

        return Result(
            input=input_path,
            video=pipeline_result.output_path,
            speech=pipeline_result.speech_path,
            sfx=pipeline_result.sfx_path,
            duration=duration,
            processing_time=float(pipeline_result.processing_time),
        )

    def _resolve_output_dir(self, override: Union[PathLike, _Unset]) -> Path:
        if not isinstance(override, _Unset):
            return Path(override).expanduser()
        if self._default_output_dir is not None:
            return self._default_output_dir
        return Path.cwd()

    @staticmethod
    def _resolve_bool(override: Union[bool, _Unset], default: bool) -> bool:
        if isinstance(override, _Unset):
            return default
        return bool(override)

    @staticmethod
    def _resolve_value(override, default):
        if isinstance(override, _Unset):
            return default
        return override

    def _ensure_pipeline(self, output_dir: Path):
        from .core.pipeline import EluatePipeline

        if self._preflight is None:
            # download_if_missing=False so a missing checkpoint surfaces
            # as ``ModelNotInstalledError`` instead of triggering a
            # background download from the Python API. No console is
            # passed: library code routes user-visible events through
            # ``logging`` (silent by default) rather than Rich.
            self._preflight = run_preflight(self._checkpoint, download_if_missing=False)

        if self._pipeline is None:
            self._pipeline = EluatePipeline(
                output_dir=output_dir,
                model_path=self._preflight.model_path,
                config_path=self._preflight.config_path,
                arch=self._preflight.arch,
                device_override=self._device,
            )
        elif self._pipeline.output_dir != output_dir:
            self._pipeline.output_dir = output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

        return self._pipeline


def elute(
    input: PathLike,
    *,
    outputs: Iterable[str] = ("video",),
    output_dir: Optional[PathLike] = None,
    overwrite: bool = False,
    force: bool = False,
    audio_codec: str = "aac",
    audio_bitrate: str = "256k",
    on_progress: Optional[ProgressCallback] = None,
    device: Optional[str] = None,
    checkpoint: str = "multi",
) -> Result:
    """One-shot equivalent of ``Session(...).elute(input, outputs=outputs)``.

    Loads the model, processes one file, and tears down. For more than
    one file in a row, prefer :class:`Session` — it loads the model
    once instead of per call.

    See :meth:`Session.elute` for the semantics of ``output_dir``,
    ``overwrite``, ``force``, ``audio_codec``, ``audio_bitrate``, and
    ``on_progress``. ``device`` selects inference hardware: ``None``
    auto-detects CUDA > MPS > CPU; explicit ``"cuda" | "mps" | "cpu"``
    forces that device. ``checkpoint`` selects the bandit-v2 language
    variant: default ``"multi"`` matches the CLI; other valid values are
    drawn from :data:`eluate.utils.paths.CHECKPOINT_KEYS`.
    """
    with Session(
        output_dir=output_dir,
        overwrite=overwrite,
        force=force,
        audio_codec=audio_codec,
        audio_bitrate=audio_bitrate,
        on_progress=on_progress,
        device=device,
        checkpoint=checkpoint,
    ) as session:
        return session.elute(input, outputs=outputs)
