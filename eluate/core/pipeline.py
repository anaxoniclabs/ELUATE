# SPDX-License-Identifier: MIT
"""
Pipeline orchestrator - coordinates all processing stages.
"""

import logging
import math
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from eluate.utils.telemetry import record_stage
from eluate.utils.validators import (
    DurationOutOfRange,
    InsufficientDiskSpace,
    check_disk_space,
    estimate_required_disk_bytes,
    validate_duration,
)

from .compiler import compile_video
from .extractor import extract_audio, get_video_duration
from .separator import SAMPLE_RATE as SEPARATOR_SAMPLE_RATE
from .separator import BanditSeparator

logger = logging.getLogger("eluate.core.pipeline")


def _safe_title(stem: str) -> str:
    """Sanitise a video filename stem for use in output filenames.

    Same rule used by both ``EluatePipeline`` and ``eluate.api`` so the
    API-layer collision precheck and the pipeline-layer writes target
    identical paths.
    """
    safe = "".join(c for c in stem if c.isalnum() or c in " -_").strip()[:80]
    return safe or "output"


def resolve_output_paths(
    input_stem: str,
    output_dir: Path,
    outputs: Tuple[str, ...],
) -> dict:
    """Return ``{kind: path}`` for the requested outputs.

    The suffix convention (``_eluted.mp4`` / ``_speech.wav`` / ``_sfx.wav``)
    is part of the v1.x public contract; this helper is the single source
    of truth so the API and pipeline cannot drift.
    """
    title = _safe_title(input_stem)
    paths: dict = {}
    if "video" in outputs:
        paths["video"] = output_dir / f"{title}_eluted.mp4"
    if "speech" in outputs:
        paths["speech"] = output_dir / f"{title}_speech.wav"
    if "sfx" in outputs:
        paths["sfx"] = output_dir / f"{title}_sfx.wav"
    return paths


@dataclass
class VideoInfo:
    """Information about the video being processed."""

    title: str
    duration: int  # seconds
    duration_str: str
    filepath: Path


@dataclass
class PipelineResult:
    """Result of pipeline execution.

    ``error_exc`` carries the original exception so the API layer can
    translate it into a typed ``eluate.*`` exception (or re-raise stdlib
    exceptions like ``FileNotFoundError``/``PermissionError`` unwrapped)
    instead of stringifying everything into ``EluateError``.
    """

    success: bool
    output_path: Optional[Path]
    video_info: Optional[VideoInfo]
    processing_time: float  # seconds
    error: Optional[str] = None
    speech_path: Optional[Path] = None
    sfx_path: Optional[Path] = None
    error_exc: Optional[BaseException] = None


class EluatePipeline:
    """
    Main pipeline that orchestrates all processing stages.

    Stages:
    1. Extract - Extract audio track from video
    2. Separate - Remove music using Bandit v2
    3. Compile - Merge processed audio back with video
    """

    def __init__(
        self,
        output_dir: Path,
        model_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        arch: str = "bandit_v2",
        on_stage_start: Optional[Callable[[str, Optional[float]], None]] = None,
        on_progress: Optional[Callable[[float, str], None]] = None,
        on_stage_complete: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str, str], None]] = None,
        on_video_info: Optional[Callable[[str, str], None]] = None,
        force: bool = False,
        device_override: Optional[str] = None,
        audio_codec: str = "aac",
        audio_bitrate: str = "256k",
    ):
        """
        Initialize pipeline.

        Args:
            output_dir: Final output directory
            model_path: Path to model checkpoint
            config_path: Path to model config
            arch: Model architecture (always "bandit_v2")
            on_stage_start: Callback(stage_id, total) when stage begins
            on_progress: Callback(progress, detail) for progress updates
            on_stage_complete: Callback(stage_id) when stage completes
            on_error: Callback(stage_id, message) on error
            on_video_info: Callback(title, duration) when video info available
            audio_codec: FFmpeg audio codec for the output mux. Default "aac".
            audio_bitrate: Audio bitrate string for lossy codecs. Default "256k".
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = model_path
        self.config_path = config_path
        self.arch = arch

        # Callbacks
        self.on_stage_start = on_stage_start or (lambda *args: None)
        self.on_progress = on_progress or (lambda *args: None)
        self.on_stage_complete = on_stage_complete or (lambda *args: None)
        self.on_error = on_error or (lambda *args: None)
        self.on_video_info = on_video_info or (lambda *args: None)

        # Skip plausibility / resource prechecks when True. Used by --force
        # at the CLI layer to let the user run on exotic inputs anyway.
        self.force = force

        # Optional device override ("auto" | "cuda" | "mps" | "cpu"). None or
        # "auto" means the separator's own auto-detection applies.
        self.device_override = device_override

        self.audio_codec = audio_codec
        self.audio_bitrate = audio_bitrate

        # Lazy-loaded separator (heavy initialization)
        self._separator: Optional[BanditSeparator] = None

    @property
    def separator(self) -> BanditSeparator:
        """Lazy-load the Bandit separator."""
        if self._separator is None:
            device = None
            if self.device_override and self.device_override != "auto":
                from eluate.utils.device import get_optimal_device

                device = get_optimal_device(self.device_override)
            self._separator = BanditSeparator(
                config_path=self.config_path,
                checkpoint_path=self.model_path,
                arch=self.arch,
                device=device,
            )
        return self._separator

    def close(self) -> None:
        """Release the separator's model and free accelerator memory.

        Idempotent. The pipeline stays usable afterwards: the next
        ``process()`` lazily rebuilds the separator.
        """
        if self._separator is not None:
            self._separator.close()
            self._separator = None

    def process(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        *,
        outputs: Tuple[str, ...] = ("video",),
    ) -> PipelineResult:
        """
        Process a local video file through the full pipeline.

        Args:
            video_path: Path to local video file
            output_path: Optional custom output path for the cleaned video
            outputs: Tuple drawn from ``{"video", "speech", "sfx"}`` selecting
                which artifacts to write. Assumed pre-validated by the API
                layer (``eluate.api._validate_outputs``); the pipeline does
                not re-check vocabulary. The mix+mux compile step is skipped
                when ``"video"`` is absent.

        Returns:
            PipelineResult with success status and output paths
        """
        produce_video = "video" in outputs
        produce_speech = "speech" in outputs
        produce_sfx = "sfx" in outputs
        start_time = time.time()
        video_path = Path(video_path)

        # Validate input file
        if not video_path.exists():
            msg = f"Video file not found: {video_path}"
            self.on_error("extract", msg)
            return PipelineResult(
                success=False,
                output_path=None,
                video_info=None,
                processing_time=0,
                error=msg,
                error_exc=FileNotFoundError(msg),
            )

        # Get video info
        probed_duration = get_video_duration(video_path)
        duration = probed_duration or 0
        hours, remainder = divmod(int(duration), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            duration_str = f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            duration_str = f"{minutes}:{seconds:02d}"

        video_info = VideoInfo(
            title=video_path.stem,
            duration=int(duration),
            duration_str=duration_str,
            filepath=video_path,
        )

        # Notify about video info
        self.on_video_info(video_info.title, duration_str)

        # Plausibility checks. --force bypasses both.
        if not self.force:
            try:
                validate_duration(probed_duration)
            except DurationOutOfRange as exc:
                self.on_error("precheck", str(exc))
                return PipelineResult(
                    success=False,
                    output_path=None,
                    video_info=video_info,
                    processing_time=time.time() - start_time,
                    error=str(exc),
                    error_exc=exc,
                )

            if probed_duration:
                required = estimate_required_disk_bytes(probed_duration)
                try:
                    check_disk_space(self.output_dir, required)
                except InsufficientDiskSpace as exc:
                    self.on_error("precheck", str(exc))
                    return PipelineResult(
                        success=False,
                        output_path=None,
                        video_info=video_info,
                        processing_time=time.time() - start_time,
                        error=str(exc),
                        error_exc=exc,
                    )

        # Resolve the user-facing output paths up front so the workspace
        # only holds intermediate artifacts.
        speech_out: Optional[Path] = None
        sfx_out: Optional[Path] = None
        if produce_speech:
            speech_out = self._resolve_stem_output(video_info.title, "speech")
        if produce_sfx:
            sfx_out = self._resolve_stem_output(video_info.title, "sfx")

        # Create temp directory for intermediate files
        with tempfile.TemporaryDirectory(prefix="eluate_") as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Stage 1: Extract audio
                with record_stage("extract") as tel:
                    tel.add("duration_seconds", video_info.duration)
                    audio_path = self._stage_extract(video_path, temp_path)

                # Stage 2: Load the separator model. Surfaced as its own
                # stage so the UI doesn't appear frozen while torch loads
                # weights between extraction and separation.
                with record_stage("load_model"):
                    self._stage_load_model()

                # Stage 3: Separate. Writes the music-removed mix (if video
                # is requested) and the requested raw stems (speech/sfx) in
                # one model pass.
                mix_temp: Optional[Path] = temp_path / "no_music.wav" if produce_video else None
                with record_stage("separate") as tel:
                    tel.add("duration_seconds", video_info.duration)
                    tel.add("model_arch", self.arch)
                    self._stage_separate(
                        audio_path,
                        mix_path=mix_temp,
                        speech_path=speech_out,
                        sfx_path=sfx_out,
                    )

                # Stage 3: Compile final video — skipped entirely when
                # "video" is not requested.
                final_output: Optional[Path] = None
                if produce_video:
                    assert mix_temp is not None
                    with record_stage("compile") as tel:
                        tel.add("duration_seconds", video_info.duration)
                        final_output = self._stage_compile(
                            video_path,
                            mix_temp,
                            video_info.title,
                            output_path,
                            probed_duration,
                        )

                processing_time = time.time() - start_time

                return PipelineResult(
                    success=True,
                    output_path=final_output,
                    video_info=video_info,
                    processing_time=processing_time,
                    speech_path=speech_out,
                    sfx_path=sfx_out,
                )

            except KeyboardInterrupt:
                raise

            except Exception as e:
                processing_time = time.time() - start_time
                error_msg = str(e)
                self.on_error("pipeline", error_msg)
                return PipelineResult(
                    success=False,
                    output_path=None,
                    video_info=video_info,
                    processing_time=processing_time,
                    error=error_msg,
                    error_exc=e,
                )

    def _stage_extract(self, video_path: Path, temp_dir: Path) -> Path:
        """Stage 1: Extract audio from video."""
        logger.info("stage extract started")
        self.on_stage_start("extract", 100)

        audio_path = temp_dir / "audio.wav"

        def progress_cb(p: float):
            self.on_progress(p * 100, "")

        extract_audio(
            video_path=video_path,
            output_path=audio_path,
            sample_rate=SEPARATOR_SAMPLE_RATE,  # Use separator's sample rate
            progress_callback=progress_cb,
            skip_duration_check=self.force,
        )

        self.on_stage_complete("extract")
        logger.info("stage extract completed")
        return audio_path

    def _stage_load_model(self) -> None:
        """Stage 2: Force the lazy model load with a smoothly ticking bar.

        torch.load exposes no internal progress signal. We run the load on
        a background thread and tick an asymptotic curve from 0 → 95% on
        the main thread, then snap to 100% the moment the loader finishes.
        Reaches ~95% by ~18 s of elapsed time, which covers a cold load on
        most machines. If the model is already cached on the Session, the
        loader returns immediately and the bar barely flickers before 100%.
        """
        logger.info("stage load_model started")
        self.on_stage_start("load_model", 100)

        done = threading.Event()
        error_holder: list[BaseException] = []

        def loader() -> None:
            try:
                # Real BanditSeparator exposes ``model`` as a lazy property;
                # test doubles may not. Either way, accessing it forces the
                # load, and a missing attribute means there's nothing to warm.
                getattr(self.separator, "model", None)
            except BaseException as exc:
                error_holder.append(exc)
            finally:
                done.set()

        t = threading.Thread(target=loader, name="eluate-model-load", daemon=True)
        t.start()

        start = time.monotonic()
        while not done.is_set():
            elapsed = time.monotonic() - start
            p = (1.0 - math.exp(-elapsed / 6.0)) * 95.0
            self.on_progress(p, "")
            done.wait(timeout=0.05)

        if error_holder:
            raise error_holder[0]

        self.on_progress(100.0, "")
        self.on_stage_complete("load_model")
        logger.info("stage load_model completed")

    def _stage_separate(
        self,
        audio_path: Path,
        *,
        mix_path: Optional[Path],
        speech_path: Optional[Path],
        sfx_path: Optional[Path],
    ) -> None:
        """Stage 2: Run separation; write only the requested artifacts."""
        logger.info("stage separate started")
        self.on_stage_start("separate", 100)

        def progress_cb(p: float):
            self.on_progress(p * 100, "")

        self.separator.produce_outputs(
            audio_path=audio_path,
            workspace_dir=audio_path.parent,
            mix_path=mix_path,
            speech_path=speech_path,
            sfx_path=sfx_path,
            progress_callback=progress_cb,
        )

        self.on_stage_complete("separate")
        logger.info("stage separate completed")

    def _resolve_stem_output(self, title: str, kind: str) -> Path:
        """Build ``output_dir/<safe_title>_<kind>.wav``.

        Collision policy is enforced one layer up at the API boundary
        (``eluate.api`` honours ``overwrite=``); this writes through to
        the canonical name and lets the writer truncate on overwrite.
        """
        return resolve_output_paths(title, self.output_dir, (kind,))[kind]

    def _stage_compile(
        self,
        video_path: Path,
        audio_path: Path,
        title: str,
        custom_output: Optional[Path] = None,
        duration: Optional[float] = None,
    ) -> Path:
        """Stage 3: Compile final video with processed audio."""
        logger.info("stage compile started")
        self.on_stage_start("compile", 100)

        if custom_output:
            output_path = Path(custom_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = resolve_output_paths(title, self.output_dir, ("video",))["video"]
            safe_title = _safe_title(title)

            counter = 1
            base_safe_title = safe_title
            while output_path.exists():
                safe_title = f"{base_safe_title}_{counter}"
                output_path = self.output_dir / f"{safe_title}_eluted.mp4"
                counter += 1

        def progress_cb(p: float):
            self.on_progress(p * 100, "")

        compile_video(
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_path,
            audio_codec=self.audio_codec,
            audio_bitrate=self.audio_bitrate,
            progress_callback=progress_cb,
            duration=duration,
        )

        self.on_stage_complete("compile")
        logger.info("stage compile completed")
        return output_path


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"
