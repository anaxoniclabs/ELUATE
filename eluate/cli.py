# SPDX-License-Identifier: MIT
"""
Eluate CLI - Main entry point.

Remove background music from video files with a beautiful terminal interface.
"""

import os
import warnings

# Must be set before any torch import so MPS ops that lack a native kernel
# can transparently fall back to CPU instead of raising at runtime.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch")
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchaudio")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from . import __version__
from .ui.ascii_art import get_header_panel
from .ui.components import (
    create_console,
    error_panel,
    info_panel,
    success_panel,
)
from .ui.progress import EluateProgress
from .ui.theme import Colors

# Heavy import (pipeline -> separator -> torch -> vendor submodule) deferred
# to ``_elute_one``. This keeps ``eluate --version`` and
# ``eluate info`` snappy for users who don't need inference.
from .utils.paths import (
    CHECKPOINT_KEYS,
    get_app_dir,
    get_checkpoint_path,
    get_output_dir,
)
from .utils.preflight import ensure_checkpoint, run_preflight
from .utils.telemetry import record_run_start
from .utils.validators import VIDEO_EXTENSIONS, validate_video_file

# Equal-quarter offsets matching ``eluate.api._STAGE_BOUNDS``. The API hands
# us a global fraction in [0, 1] tagged with the active stage label; we
# invert that back into a per-stage 0–100 update so the existing
# Rich progress bar (one task per stage) keeps its current behaviour.
_STAGE_OFFSETS = {
    "extract": 0.0,
    "load_model": 0.25,
    "separate": 0.50,
    "compile": 0.75,
}


def _clean_dragged_path(raw: str) -> str:
    """Normalize a path string pasted / dragged into a terminal prompt.

    Terminals on macOS/Linux escape spaces as ``\\ `` when the user drags
    a file into an interactive prompt. Shells that pass through quotes
    also leave wrapping ``"`` or ``'`` around the path. Handle both,
    plus the backslash-escape of common shell metacharacters:
    space, parenthesis, ampersand, single/double quote, dollar sign,
    semicolon, and backslash itself. Anything else is left alone.
    """
    s = raw.strip()
    # Strip one layer of matching outer quotes if present.
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1]
    # Normalize escaped shell metacharacters (bash/zsh-style, one level).
    for ch in (" ", "(", ")", "&", "'", '"', "$", ";", "\\"):
        s = s.replace("\\" + ch, ch)
    return s


def _free_device_cache_safely() -> None:
    """Best-effort accelerator cache flush between batch items.

    Imports ``torch`` lazily so the helper can live in this module without
    paying the torch-import cost for CLI paths that never touch the pipeline.
    """
    import gc

    gc.collect()
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _resolve_target_path(
    input_path: Path,
    output_path: Optional[Path],
    default_output_dir: Path,
) -> Path:
    """Pick the final user-facing output path, auto-numbering on collision.

    Mirrors the legacy CLI policy:
    - With ``-o``: use the path verbatim (silently overwrites if present).
    - Without ``-o``: write to ``default_output_dir/<safe_stem>_eluted.mp4``,
      auto-numbering (`_1`, `_2`, …) when a file with that name already
      exists so prior runs are never destroyed.
    """
    from .core.pipeline import _safe_title

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    default_output_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = _safe_title(input_path.stem)
    candidate = default_output_dir / f"{safe_stem}_eluted.mp4"
    counter = 1
    while candidate.exists():
        candidate = default_output_dir / f"{safe_stem}_{counter}_eluted.mp4"
        counter += 1
    return candidate


def _make_progress_callback(progress: EluateProgress, state: dict):
    """Build an ``on_progress(fraction, stage)`` shim onto ``EluateProgress``.

    The API exposes a single fraction-in-[0,1]+stage-label callback. The
    existing Rich UI is built around per-stage begin/update/complete
    events. This shim:

    - calls ``begin_stage`` the first time a new stage label arrives
      (so PENDING flips to ACTIVE on the first progress tick)
    - calls ``complete_stage`` on the previous stage when the label
      changes
    - converts the global fraction back into a 0–100 per-stage value so
      the progress bar fills as before
    """

    def on_progress(fraction: float, stage: str) -> None:
        previous = state["current_stage"]
        if stage != previous:
            if previous in _STAGE_OFFSETS:
                progress.complete_stage(previous)
            if stage in _STAGE_OFFSETS:
                progress.begin_stage(stage, 100)
            state["current_stage"] = stage

        if stage in _STAGE_OFFSETS:
            local = (fraction - _STAGE_OFFSETS[stage]) * 4.0
            local = max(0.0, min(1.0, local)) * 100.0
            progress.update_progress(completed=local)

    return on_progress


def _elute_one(
    session,
    input_path: Path,
    output_path: Optional[Path],
    console: Console,
    default_output_dir: Path,
) -> bool:
    """Process a single video using ``session``. Returns True if successful."""
    from . import EluateError
    from .core.pipeline import format_duration

    if not input_path.exists():
        console.print(error_panel("File not found", f"Video file does not exist: {input_path}"))
        return False

    if not input_path.is_file():
        console.print(error_panel("Invalid input", f"Path is not a file: {input_path}"))
        return False

    final_target = _resolve_target_path(input_path, output_path, default_output_dir)

    progress = EluateProgress(console)
    state: dict[str, Optional[str]] = {"current_stage": None}
    on_progress_cb = _make_progress_callback(progress, state)

    progress.start()
    result = None
    # Write through a hidden temp subdirectory of the final target's parent
    # so the rename is atomic on the same filesystem and any pre-existing
    # file at the auto-numbered path is never partially overwritten by the
    # API's ``overwrite=True`` write. Cleanup runs on success and failure.
    target_parent = final_target.parent
    with tempfile.TemporaryDirectory(prefix=".eluate_cli_", dir=target_parent) as tmp:
        tmp_dir = Path(tmp)
        try:
            result = session.elute(
                input_path,
                outputs=("video",),
                output_dir=tmp_dir,
                overwrite=True,
                on_progress=on_progress_cb,
            )
            current_stage = state["current_stage"]
            if current_stage is not None and current_stage in _STAGE_OFFSETS:
                progress.complete_stage(current_stage)
        except KeyboardInterrupt:
            progress.stop()
            console.print(f"\n[{Colors.WARNING}]Processing cancelled[/]")
            return False
        except EluateError as e:
            progress.stop()
            console.print(error_panel("Processing failed", str(e)))
            return False
        except Exception as e:
            progress.stop()
            console.print(error_panel("Processing failed", str(e)))
            return False
        finally:
            progress.stop()

        if result is None or result.video is None:
            console.print(error_panel("Processing failed", "Unknown error"))
            return False

        # ``shutil.move`` handles cross-device moves gracefully (falls
        # back to copy+unlink), and overwrites the destination on the
        # same filesystem. ``-o`` callers explicitly accepted overwrite
        # already; the auto-numbered default path is guaranteed
        # non-existent by ``_resolve_target_path``.
        try:
            shutil.move(str(result.video), str(final_target))
        except OSError as e:
            console.print(error_panel("Processing failed", f"Could not write output: {e}"))
            return False

    console.print()

    duration_str = format_duration(result.duration)
    console.print(
        success_panel(
            output_path=str(final_target),
            duration=duration_str,
            processing_time=format_duration(result.processing_time),
        )
    )
    return True


def _build_session(
    *,
    force: bool,
    device_override: Optional[str],
    audio_codec: str,
    audio_bitrate: str,
    checkpoint: str,
    preflight_config=None,
):
    """Construct a ``eluate.Session`` configured from CLI flags.

    ``checkpoint`` and ``device`` are session-locked at the API layer
    (changing either would require reloading the model), which matches
    the CLI's per-invocation single-checkpoint, single-device model.
    """
    from . import Session

    session = Session(
        force=force,
        audio_codec=audio_codec,
        audio_bitrate=audio_bitrate,
        device=device_override,
        checkpoint=checkpoint,
    )
    if preflight_config is not None:
        session._preflight = preflight_config
    return session


def _process_sequence(
    video_paths: List[Path],
    console: Console,
    session,
    default_output_dir: Path,
) -> tuple[int, int]:
    """Process a flat list of video paths sequentially. Returns (success, fail)."""
    success_count = 0
    fail_count = 0
    total = len(video_paths)

    for i, video_path in enumerate(video_paths, 1):
        console.print(f"[{Colors.TEXT_SECONDARY}]─── Video {i}/{total} ───[/]")
        console.print(f"[{Colors.TEXT_MUTED}]{video_path}[/]")
        console.print()

        if _elute_one(session, video_path, None, console, default_output_dir):
            success_count += 1
        else:
            fail_count += 1

        # Free accelerator caches between items so peak memory stays bounded
        # over long batch runs. Safe on CPU (no-op).
        _free_device_cache_safely()

        console.print()

    console.print(f"[{Colors.PRIMARY}]─── Complete ───[/]")
    console.print(f"[{Colors.SUCCESS}]Successful: {success_count}[/]")
    if fail_count > 0:
        console.print(f"[{Colors.ERROR}]Failed: {fail_count}[/]")
    console.print()

    return success_count, fail_count


def _process_chunked(
    video_paths: List[Path],
    batch_size: int,
    console: Console,
    session,
    default_output_dir: Path,
    source_label: str = "",
) -> tuple[int, int]:
    """
    Process video paths in chunks, pausing between batches for user confirmation.

    Args:
        video_paths: Ordered list of video paths to process
        batch_size: Maximum videos per batch
        source_label: Optional label shown in the header (e.g. folder path)

    Returns:
        (total_success, total_fail)
    """
    total_videos = len(video_paths)
    total_batches = (total_videos + batch_size - 1) // batch_size

    if source_label:
        console.print(f"[{Colors.TEXT_MUTED}]{source_label}[/]")
    console.print(
        f"[{Colors.PRIMARY}]Processing {total_videos} videos in "
        f"{total_batches} batches of up to {batch_size}[/]"
    )
    console.print()

    total_success = 0
    total_fail = 0

    for batch_num, chunk_start in enumerate(range(0, total_videos, batch_size), 1):
        chunk_end = min(chunk_start + batch_size, total_videos)
        chunk = video_paths[chunk_start:chunk_end]
        chunk_size = len(chunk)

        console.print(
            f"[{Colors.PRIMARY}]═══ Batch {batch_num}/{total_batches} ({chunk_size} videos) ═══[/]"
        )
        console.print()

        batch_success = 0
        batch_fail = 0

        for i, video_path in enumerate(chunk, 1):
            global_index = chunk_start + i
            console.print(
                f"[{Colors.TEXT_SECONDARY}]─── Video {global_index}/{total_videos} "
                f"(Batch {batch_num}, #{i}) ───[/]"
            )
            console.print(f"[{Colors.TEXT_MUTED}]{video_path}[/]")
            console.print()

            if _elute_one(session, video_path, None, console, default_output_dir):
                batch_success += 1
            else:
                batch_fail += 1

            _free_device_cache_safely()
            console.print()

        total_success += batch_success
        total_fail += batch_fail

        console.print(f"[{Colors.PRIMARY}]─── Batch {batch_num} Complete ───[/]")
        console.print(f"[{Colors.SUCCESS}]Successful: {batch_success}/{chunk_size}[/]")
        if batch_fail > 0:
            console.print(f"[{Colors.ERROR}]Failed: {batch_fail}[/]")
        console.print(
            f"[{Colors.TEXT_MUTED}]Overall progress: "
            f"{total_success + total_fail}/{total_videos} processed[/]"
        )
        console.print()

        if chunk_end < total_videos:
            remaining = total_videos - chunk_end
            remaining_batches = total_batches - batch_num
            console.print(
                f"[{Colors.PRIMARY}]Batch {batch_num} finished. "
                f"{remaining} videos remaining ({remaining_batches} more batches).[/]"
            )
            console.print(
                f"[{Colors.TEXT_SECONDARY}]Press Enter to start batch {batch_num + 1}, "
                f"or Ctrl+C to stop.[/]"
            )
            try:
                input()
                console.print()
            except KeyboardInterrupt:
                console.print()
                console.print(f"[{Colors.WARNING}]Stopped after batch {batch_num}.[/]")
                break

    console.print(f"[{Colors.PRIMARY}]═══ All Batches Complete ═══[/]")
    console.print(f"[{Colors.SUCCESS}]Total successful: {total_success}/{total_videos}[/]")
    if total_fail > 0:
        console.print(f"[{Colors.ERROR}]Total failed: {total_fail}[/]")
    console.print()

    return total_success, total_fail


def get_video_files_from_folder(folder_path: Path) -> List[Path]:
    """Get all video files from a folder, sorted numerically/alphabetically."""
    videos: List[Path] = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(folder_path.glob(f"*{ext}"))
        videos.extend(folder_path.glob(f"*{ext.upper()}"))

    def sort_key(p: Path):
        stem = p.stem
        try:
            return (0, int(stem), stem)
        except ValueError:
            return (1, 0, stem)

    return sorted(set(videos), key=sort_key)


def interactive_mode(
    console: Console,
    session,
    default_output_dir: Path,
) -> bool:
    """Interactive mode - prompt user for video file path."""
    prompt_text = Text()
    prompt_text.append(
        "Drag a video file here, or paste its path",
        style=f"bold {Colors.PRIMARY}",
    )

    console.print(
        Panel(
            prompt_text,
            border_style=Colors.PRIMARY,
            padding=(0, 2),
        )
    )
    console.print()

    while True:
        try:
            file_path = Prompt.ask(f"[{Colors.TEXT_SECONDARY}]Video path[/]", console=console)

            if not file_path:
                console.print(f"[{Colors.WARNING}]Please enter a video file path.[/]")
                continue

            file_path = _clean_dragged_path(file_path)
            input_path = Path(file_path).expanduser().resolve()

            if not input_path.exists():
                console.print(f"[{Colors.ERROR}]File not found: {input_path}[/]")
                console.print(f"[{Colors.TEXT_MUTED}]Please check the path and try again.[/]")
                console.print()
                continue

            if not validate_video_file(input_path):
                console.print(f"[{Colors.ERROR}]Not a supported video file: {input_path.name}[/]")
                console.print(
                    f"[{Colors.TEXT_MUTED}]Supported formats: mp4, mkv, avi, mov, webm, flv, wmv, m4v[/]"
                )
                console.print()
                continue

            console.print()
            console.print(
                f"[{Colors.SUCCESS}]✓[/] [{Colors.TEXT_SECONDARY}]Selected:[/] {input_path.name}"
            )
            console.print()

            return _elute_one(session, input_path, None, console, default_output_dir)

        except KeyboardInterrupt:
            console.print()
            console.print(f"[{Colors.WARNING}]Cancelled.[/]")
            return False


def cmd_info(console: Console) -> None:
    """Print system and model diagnostic information."""
    import torch

    # Device
    from .utils.device import configure_mps_settings, get_optimal_device

    configure_mps_settings()
    device = get_optimal_device()

    # FFmpeg version
    from .core.extractor import get_ffmpeg_version

    ffmpeg_ver = get_ffmpeg_version() or "not found"

    # Torch version
    torch_ver = torch.__version__

    # Available memory via shared helper — uses the correct OS page size
    # (16384 on Apple Silicon, 4096 on Intel) and counts speculative pages
    # as free, matching Activity Monitor.
    from .utils.device import get_memory_info

    mem = get_memory_info()
    if "error" in mem:
        memory_str = "unavailable"
    else:
        memory_str = f"{mem['free_gb']:.1f} GB free"

    # Bandit v2 language checkpoints
    models_dir = get_app_dir() / "models"
    bandit_v2_status = {}
    for key in CHECKPOINT_KEYS:
        ckpt = get_checkpoint_path(key, model="bandit-v2")
        if ckpt.exists():
            size_mb = ckpt.stat().st_size // (1024 * 1024)
            bandit_v2_status[key] = f"installed ({size_mb} MB)"
        else:
            bandit_v2_status[key] = "not downloaded"

    config_path = models_dir / "config_bandit_v2.yaml"

    console.print(
        info_panel(
            "System",
            {
                "Device": str(device),
                "PyTorch": torch_ver,
                "FFmpeg": ffmpeg_ver,
                "Free memory": memory_str,
            },
        )
    )
    console.print()
    console.print(
        info_panel(
            "Bandit v2",
            {
                "Directory": str(models_dir),
                "Config": "present" if config_path.exists() else "missing (using bundled fallback)",
                **{f"  {key}": status for key, status in bandit_v2_status.items()},
            },
        )
    )


def main():
    """Main entry point for Eluate CLI."""
    parser = argparse.ArgumentParser(
        description="Eluate - Remove background music from video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  eluate                                Interactive mode - prompts for file
  eluate video.mp4                      Process file, output to ~/Documents/ELUATE/
  eluate video.mp4 -o output.mp4        Custom output path
  eluate --checkpoint eng video.mp4     Use English-optimised model
  eluate --batch files.txt              Batch mode with list of file paths
  eluate --batch files.txt --batch-size 10
                                        Process 10 videos, pause, then continue
  eluate --folder /path/to/videos       Process all videos in a folder
  eluate --folder ./videos --batch-size 10
                                        Process folder in batches of 10
  eluate info                           Show device, model and FFmpeg status
  eluate setup                          Download the default model
  eluate --checkpoint eng setup         Download a specific language checkpoint
  eluate video.mp4 --device cpu         Force CPU inference (skip MPS/CUDA)
  eluate video.mp4 --force              Skip duration/disk-space prechecks
  eluate video.mp4 --audio-codec alac   Write Apple Lossless audio
  eluate video.mp4 --audio-bitrate 192k Lower AAC bitrate

Batch file format (one path per line):
  /path/to/video1.mp4
  /path/to/video2.mkv
  # Lines starting with # are ignored

Checkpoint options: multi (default), eng, deu, fra, spa, cmn, fao

Model: bandit-v2, CC-BY-SA 4.0, 48 kHz (share-alike;
       consult a lawyer before commercial use).

Local debug log (off by default):
  Set ELUATE_TELEMETRY=1 to write a local JSONL log of processing stages to
  ~/.eluate/telemetry.jsonl. Helpful when reporting bugs. Nothing is ever
  sent anywhere — the log stays on your machine.
        """,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"eluate {__version__}",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help='Video file to process, "info" to show system status, or "setup" to download the model',
    )
    parser.add_argument(
        "-o", "--output", help="Output file path (default: ~/Documents/ELUATE/<name>_eluted.mp4)"
    )
    parser.add_argument(
        "-b",
        "--batch",
        metavar="FILE",
        help="Process multiple videos from a file (one path per line)",
    )
    parser.add_argument("-f", "--folder", metavar="DIR", help="Process all videos in a folder")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Process videos in batches of N, pausing between batches",
    )
    parser.add_argument(
        "--checkpoint",
        default="multi",
        choices=CHECKPOINT_KEYS,
        metavar="KEY",
        help=f"Bandit v2 checkpoint to use. Options: {', '.join(CHECKPOINT_KEYS)} (default: multi)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip duration and disk-space prechecks (use with care)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Override inference device. Default: auto (cuda > mps > cpu).",
    )
    parser.add_argument(
        "--audio-codec",
        default="aac",
        metavar="CODEC",
        help="FFmpeg codec for the output audio track (default: aac). "
        'Use "alac" or "flac" for lossless; note MP4 container support varies.',
    )
    parser.add_argument(
        "--audio-bitrate",
        default="256k",
        metavar="RATE",
        help="Audio bitrate for lossy codecs (default: 256k). "
        "Ignored by lossless codecs like alac/flac.",
    )

    args = parser.parse_args()

    # Emit run.start into the local debug log if the user has opted in
    # (ELUATE_TELEMETRY=1). No-op otherwise.
    record_run_start()

    console = create_console()

    # "eluate info" subcommand — handle before the header/model-check
    if args.input == "info":
        console.clear()
        console.print(get_header_panel())
        console.print()
        cmd_info(console)
        sys.exit(0)

    # "eluate setup" downloads the selected checkpoint without processing
    # a video. Keep status checks read-only by leaving ``info`` separate.
    if args.input == "setup":
        console.clear()
        setup(args.checkpoint, console=console)
        sys.exit(0)

    console.clear()
    console.print(get_header_panel())
    console.print()

    # Run preflight at the CLI layer so first-run downloads happen with
    # Rich-rendered progress. The Session built below also runs preflight
    # internally (silent, ``download_if_missing=False``); since the
    # checkpoint is now present that call is a no-op.
    preflight_config = run_preflight(args.checkpoint, console)

    force = args.force
    device_override = args.device if args.device != "auto" else None
    audio_codec = args.audio_codec
    audio_bitrate = args.audio_bitrate

    if args.batch_size is not None and args.batch_size < 1:
        console.print(error_panel("Invalid batch size", "Batch size must be at least 1"))
        sys.exit(1)

    default_output_dir = get_output_dir()

    session = _build_session(
        force=force,
        device_override=device_override,
        audio_codec=audio_codec,
        audio_bitrate=audio_bitrate,
        checkpoint=args.checkpoint,
        preflight_config=preflight_config,
    )

    # Batch mode
    if args.batch:
        batch_file = Path(args.batch).expanduser().resolve()
        if not batch_file.exists():
            console.print(error_panel("File not found", f"Batch file does not exist: {batch_file}"))
            sys.exit(1)
        with open(batch_file) as f:
            raw = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        if not raw:
            console.print(
                error_panel("No files found", "The batch file is empty or contains no valid paths")
            )
            sys.exit(1)
        video_paths = [Path(p).expanduser().resolve() for p in raw]
        console.print(f"[{Colors.PRIMARY}]Batch processing {len(video_paths)} videos...[/]")
        console.print()
        with session:
            if args.batch_size:
                _, fail = _process_chunked(
                    video_paths, args.batch_size, console, session, default_output_dir
                )
            else:
                _, fail = _process_sequence(video_paths, console, session, default_output_dir)
        sys.exit(0 if fail == 0 else 1)

    # Folder mode
    if args.folder:
        folder_path = Path(args.folder).expanduser().resolve()
        if not folder_path.exists():
            console.print(
                error_panel("Folder not found", f"Directory does not exist: {folder_path}")
            )
            sys.exit(1)
        if not folder_path.is_dir():
            console.print(error_panel("Invalid path", f"Path is not a directory: {folder_path}"))
            sys.exit(1)
        video_files = get_video_files_from_folder(folder_path)
        if not video_files:
            console.print(error_panel("No videos found", f"No video files found in: {folder_path}"))
            sys.exit(1)
        console.print(f"[{Colors.PRIMARY}]Found {len(video_files)} videos in folder[/]")
        console.print(f"[{Colors.TEXT_MUTED}]{folder_path}[/]")
        console.print()
        with session:
            if args.batch_size:
                _, fail = _process_chunked(
                    video_files,
                    args.batch_size,
                    console,
                    session,
                    default_output_dir,
                    source_label=str(folder_path),
                )
            else:
                _, fail = _process_sequence(video_files, console, session, default_output_dir)
        sys.exit(0 if fail == 0 else 1)

    # Interactive mode - no input provided
    if not args.input:
        with session:
            ok = interactive_mode(console, session, default_output_dir)
        sys.exit(0 if ok else 1)

    # Direct mode - input file provided
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else None

    with session:
        ok = _elute_one(session, input_path, output_path, console, default_output_dir)
    sys.exit(0 if ok else 1)


def setup(checkpoint: str = "multi", console: Optional[Console] = None):
    """Download model and run setup."""
    if console is None:
        console = create_console()

    console.print(get_header_panel())
    console.print()

    console.print(f"[{Colors.PRIMARY}]Setting up Eluate...[/]")
    console.print()

    model_path = get_checkpoint_path(checkpoint)

    if model_path.exists():
        # Also verifies the declared digest for existing checkpoint files.
        ensure_checkpoint(checkpoint, console)
        console.print(f"[{Colors.SUCCESS}]Model already installed![/]")
        console.print(f"  {model_path}")
        return

    ensure_checkpoint(checkpoint, console)

    # Seed the user-install copy of the model config from the wheel-bundled
    # one. Not strictly required at runtime — get_config_path() will fall
    # back to the bundled resource — but it preserves the legacy on-disk
    # layout under ~/.eluate/models/ that some users may expect.
    from .utils.paths import get_app_dir, get_bundled_config_path

    bundled_config = get_bundled_config_path("bandit_v2.yaml")
    user_config = get_app_dir() / "models" / "config_bandit_v2.yaml"
    if bundled_config is not None and not user_config.exists():
        import shutil

        shutil.copy2(bundled_config, user_config)
        console.print(f"[{Colors.SUCCESS}]Config installed to {user_config}[/]")

    console.print()
    console.print(f"[{Colors.SUCCESS}]Setup complete![/]")
    console.print(f"[{Colors.TEXT_SECONDARY}]Run 'eluate video.mp4' to start processing videos.[/]")


if __name__ == "__main__":
    main()
