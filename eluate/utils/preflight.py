# SPDX-License-Identifier: MIT
"""
Startup validation: checkpoint, config, FFmpeg.

run_preflight() sequences all checks and returns a PreflightConfig on
success. Calls sys.exit() on any hard failure so callers don't need to
handle individual error cases.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from eluate.ui.components import error_panel
from eluate.ui.theme import Colors
from eluate.utils.download import DownloadError, download_file, sha256_of_file
from eluate.utils.paths import (
    DEFAULT_MODEL,
    get_checkpoint_path,
    get_checkpoint_sha256,
    get_checkpoint_url,
    get_config_path,
    get_model_profile,
)

logger = logging.getLogger("eluate.utils.preflight")


@dataclass
class PreflightConfig:
    """Resolved model resources, available once all startup checks pass."""

    model_path: Path
    config_path: Path
    arch: str


def _checkpoint_label(key: str, model: str = DEFAULT_MODEL) -> str:
    profile = get_model_profile(model)
    return key if profile["supports_language"] else model


def _setup_hint(key: str) -> str:
    if key == "multi":
        return "eluate setup"
    return f"eluate --checkpoint {key} setup"


def _checkpoint_integrity_error(path: Path, expected_sha: str, actual_sha: str, key: str) -> str:
    return (
        f"Checkpoint SHA256 mismatch at {path}.\n"
        f"Expected: {expected_sha}\n"
        f"Actual:   {actual_sha}\n\n"
        f"Delete the file and run `{_setup_hint(key)}` to download a fresh copy."
    )


def verify_checkpoint_integrity(
    path: Path,
    key: str,
    *,
    console: Optional[Console] = None,
    model: str = DEFAULT_MODEL,
) -> None:
    """Verify an existing checkpoint when a SHA256 is declared.

    Unknown checkpoint digests are still allowed, matching download-time
    behaviour, but declared digests are enforced before any checkpoint can
    reach ``torch.load(weights_only=False)``.
    """
    expected_sha = get_checkpoint_sha256(key, model=model)
    label = _checkpoint_label(key, model=model)
    if not expected_sha:
        logger.warning(
            "No SHA256 declared for checkpoint %r; integrity will not be verified",
            label,
        )
        if console is not None:
            console.print(
                f"[{Colors.TEXT_MUTED}]Note: no SHA256 declared for checkpoint "
                f"'{label}'; integrity is not verified.[/]"
            )
        return

    actual_sha = sha256_of_file(path)
    if actual_sha.lower() != expected_sha.lower():
        raise DownloadError(_checkpoint_integrity_error(path, expected_sha, actual_sha, key))


def ensure_checkpoint(
    key: str, console: Optional[Console] = None, model: str = DEFAULT_MODEL
) -> Path:
    """Return checkpoint path, downloading if missing. Exits on download failure.

    Downloads are atomic (written to ``<dest>.part`` then renamed) and,
    when an expected SHA256 is declared in ``paths.CHECKPOINT_SHA256``,
    integrity-checked before the rename. Transient network errors are
    retried with exponential backoff; partial downloads are resumed where
    the server supports byte ranges.
    """
    path = get_checkpoint_path(key, model=model)

    if console is None:
        console = Console(quiet=True)

    profile = get_model_profile(model)
    label = _checkpoint_label(key, model=model)
    if path.exists():
        try:
            verify_checkpoint_integrity(path, key, console=console, model=model)
        except DownloadError as e:
            logger.error("Checkpoint integrity check failed for %r: %s", label, e)
            console.print()
            console.print(error_panel("Checkpoint verification failed", str(e)))
            sys.exit(1)
        return path

    url = get_checkpoint_url(key, model=model)
    size_mb = profile.get("download_size_mb", 450)
    expected_sha = get_checkpoint_sha256(key, model=model)

    logger.info(
        "Checkpoint %r not found locally; downloading ~%dMB from %s",
        label,
        size_mb,
        url,
    )
    if not expected_sha:
        logger.warning(
            "No SHA256 declared for checkpoint %r; integrity will not be verified",
            label,
        )

    console.print(
        f"[{Colors.PRIMARY}]Checkpoint '{label}' not found locally. "
        f"Downloading (~{size_mb} MB)...[/]"
    )
    console.print(f"[{Colors.TEXT_MUTED}]{url}[/]")
    if not expected_sha:
        console.print(
            f"[{Colors.TEXT_MUTED}]Note: no SHA256 declared for this checkpoint; "
            f"integrity is not verified.[/]"
        )
    console.print()

    progress = Progress(
        SpinnerColumn(spinner_name="bouncingBall", style=Colors.PRIMARY),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(
            bar_width=40,
            style=Colors.BG_PANEL,
            complete_style=Colors.PRIMARY,
            finished_style=Colors.PRIMARY_LIGHT,
        ),
        TaskProgressColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=False,
        refresh_per_second=20,
    )

    try:
        with progress:
            task_id = progress.add_task("Downloading", total=None)

            def _report(downloaded: int, total: Optional[int]) -> None:
                if total is not None:
                    progress.update(task_id, total=total, completed=downloaded)
                else:
                    progress.update(task_id, completed=downloaded)

            download_file(
                url=url,
                dest=path,
                expected_sha256=expected_sha or None,
                retries=3,
                progress_callback=_report,
            )
    except DownloadError as e:
        logger.error("Download failed for checkpoint %r: %s", label, e)
        console.print()
        console.print(
            error_panel(
                "Download failed",
                f"{e}\n\nDownload manually from:\n  {url}\n\nSave to:\n  {path}",
            )
        )
        sys.exit(1)

    logger.info("Downloaded checkpoint %r to %s", label, path)
    console.print()
    console.print(f"[{Colors.SUCCESS}]Downloaded to {path}[/]")
    console.print()
    return path


def run_preflight(
    checkpoint_key: str,
    console: Optional[Console] = None,
    *,
    download_if_missing: bool = True,
) -> PreflightConfig:
    """Run all startup checks. Returns PreflightConfig on success.

    Checks (in order):
    1. Checkpoint present or downloaded
    2. Config file resolved
    3. FFmpeg available

    Calls sys.exit() on any hard failure.

    When ``download_if_missing=False`` (used by the Python API), a missing
    checkpoint raises ``eluate.ModelNotInstalledError`` instead of being
    downloaded. The CLI keeps the default ``True`` so first-run downloads
    continue to work.
    """
    if console is None:
        console = Console(quiet=True)

    profile = get_model_profile(DEFAULT_MODEL)
    arch = profile["arch"]

    # 1. Checkpoint (download if missing — only when allowed).
    if download_if_missing:
        model_path = ensure_checkpoint(checkpoint_key, console)
    else:
        model_path = get_checkpoint_path(checkpoint_key)
        if not model_path.exists():
            from eluate.api import ModelNotInstalledError

            raise ModelNotInstalledError(
                f"Checkpoint '{checkpoint_key}' is not installed at "
                f"{model_path}. Run `{_setup_hint(checkpoint_key)}` to download it."
            )
        try:
            verify_checkpoint_integrity(model_path, checkpoint_key)
        except DownloadError as exc:
            from eluate.api import EluateError

            raise EluateError(str(exc)) from exc

    # 2. Config.
    try:
        config_path = get_config_path()
    except FileNotFoundError as e:
        logger.error("Config not found: %s", e)
        console.print(error_panel("Config not found", str(e)))
        sys.exit(1)

    # 3. FFmpeg — checked last so the model is already ready if FFmpeg is present.
    from eluate.core.extractor import check_ffmpeg_available

    if not check_ffmpeg_available():
        logger.error("FFmpeg not found on PATH")
        if sys.platform == "darwin":
            install_hint = "Install via Homebrew:\n  brew install ffmpeg"
        elif sys.platform.startswith("linux"):
            install_hint = "Install via your package manager, e.g.:\n  apt-get install ffmpeg"
        else:
            install_hint = "Install FFmpeg and ensure it is on your PATH."
        console.print(
            error_panel(
                "FFmpeg not found",
                f"FFmpeg is required but not installed.\n\n{install_hint}",
            )
        )
        sys.exit(1)

    return PreflightConfig(model_path=model_path, config_path=config_path, arch=arch)
