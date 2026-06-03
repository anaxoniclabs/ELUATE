# SPDX-License-Identifier: MIT
"""
Shared FFmpeg subprocess helpers.

All FFmpeg invocations go through these helpers so timeouts, signal
handling, and error formatting behave consistently across extract and
compile stages.

Two entry points:

- ``run_ffmpeg_simple`` — fire-and-wait, no progress parsing. Wraps
  ``subprocess.run`` so it supports the same mocking patterns used by
  the test suite.
- ``run_ffmpeg_with_progress`` — parses FFmpeg's ``-progress`` stream
  and calls a progress callback. Uses ``Popen`` in a new session so the
  whole process group can be killed cleanly on timeout or Ctrl-C.

Both:

- Terminate the child cleanly on ``KeyboardInterrupt`` so Ctrl-C never
  leaves an orphan ``ffmpeg`` process.
- Enforce an optional timeout (wall-clock seconds). On timeout the
  process is ``terminate()``d then ``kill()``ed if still alive, and a
  ``FFmpegTimeout`` is raised.
- Surface the last lines of FFmpeg's stderr in the error string on
  non-zero exit, via ``FFmpegError``.
"""

import os
import re
import signal
import subprocess
import time
from typing import Callable, Optional


class FFmpegError(RuntimeError):
    """FFmpeg exited non-zero. Message includes stage and last stderr lines."""


class FFmpegTimeout(FFmpegError):
    """FFmpeg did not finish within the allotted time."""


def media_timeout(duration_seconds: Optional[float]) -> float:
    """Wall-clock timeout for an extract/compile/remux call.

    Healthy FFmpeg runs at many times realtime on modern hardware, so a
    ``4 * duration`` budget with a 60 s floor is extremely generous while
    still catching a genuinely stuck process. Unknown duration falls back
    to 30 minutes. Shared by the extract and compile stages so both use
    the same policy.
    """
    if duration_seconds and duration_seconds > 0:
        return max(60.0, 4.0 * duration_seconds)
    return 30 * 60.0


def _tail(text: Optional[str], lines: int = 20) -> str:
    if not text:
        return ""
    return "\n".join(text.strip().splitlines()[-lines:])


def _quiet_ffmpeg_cmd(cmd: list[str]) -> list[str]:
    """Insert silencing flags right after ``ffmpeg`` so stderr stays small.

    ``-loglevel error`` drops the per-frame info log; ``-nostats`` drops
    the periodic status line; ``-hide_banner`` skips the build banner.
    Real errors still reach stderr, which is what ``_tail`` reports on
    non-zero exit.

    Why it matters:
    ``run_ffmpeg_with_progress`` only reads stdout (the ``-progress``
    stream); stderr fills the OS pipe buffer (~64 KB on macOS) and
    deadlocks the encode on long inputs at default info loglevel.
    Silencing stderr at the source removes that failure mode without
    needing a stderr drain thread.

    No-op for non-ffmpeg commands (e.g. ffprobe), since ``-nostats`` is
    ffmpeg-specific.
    """
    if not cmd or cmd[0] != "ffmpeg":
        return list(cmd)
    return [cmd[0], "-hide_banner", "-loglevel", "error", "-nostats", *cmd[1:]]


def _describe_cmd(cmd: list[str]) -> str:
    """Short, loggable description of a command without leaking long paths."""
    return " ".join(cmd[:2]) + (" ..." if len(cmd) > 2 else "")


def run_ffmpeg_simple(
    cmd: list[str],
    *,
    timeout: Optional[float] = None,
    stage: str = "ffmpeg",
) -> subprocess.CompletedProcess:
    """
    Run an FFmpeg (or ffprobe) command and wait for it to finish.

    Uses ``subprocess.run`` under the hood so callers can mock it the
    same way as before. Adds consistent timeout and error handling.

    Args:
        cmd: Command list (must start with ``ffmpeg`` or ``ffprobe``).
        timeout: Optional wall-clock timeout in seconds.
        stage: Human-readable stage name used in error messages.

    Returns:
        ``CompletedProcess`` on success.

    Raises:
        FFmpegTimeout: If the process exceeds ``timeout``.
        FFmpegError:   If the process exits non-zero.
    """
    cmd = _quiet_ffmpeg_cmd(cmd)
    try:
        return subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise FFmpegTimeout(
            f"{stage}: FFmpeg exceeded timeout of {timeout:.0f}s ({_describe_cmd(cmd)})"
        ) from None
    except subprocess.CalledProcessError as exc:
        raise FFmpegError(
            f"{stage}: FFmpeg exited with code {exc.returncode}.\n"
            f"Command: {_describe_cmd(cmd)}\n"
            f"Last stderr lines:\n{_tail(exc.stderr)}"
        ) from None


def _start_group(cmd: list[str]) -> subprocess.Popen:
    # ``start_new_session=True`` puts FFmpeg in its own process group so
    # we can signal the whole group without touching Python's process tree.
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def _kill_group(process: subprocess.Popen, grace_seconds: float = 2.0) -> None:
    if process.poll() is not None:
        return
    try:
        pgid = os.getpgid(process.pid)
    except (ProcessLookupError, OSError):
        pgid = None

    def _signal_group(sig: int) -> None:
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
                return
            except (ProcessLookupError, OSError):
                pass
        # Fall back to signaling just the direct child.
        try:
            process.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass

    _signal_group(signal.SIGTERM)

    deadline = time.monotonic() + grace_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)

    if process.poll() is None:
        _signal_group(signal.SIGKILL)
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass


def run_ffmpeg_with_progress(
    cmd: list[str],
    duration: float,
    progress_callback: Callable[[float], None],
    *,
    timeout: Optional[float] = None,
    stage: str = "ffmpeg",
) -> None:
    """
    Run an FFmpeg command and emit progress updates via callback.

    Args:
        cmd: FFmpeg command list (must start with ``ffmpeg``).
        duration: Total media duration in seconds (for progress scaling).
        progress_callback: Called with progress fraction (0.0–1.0).
        timeout: Optional wall-clock timeout in seconds.
        stage: Human-readable stage name used in error messages.

    Raises:
        FFmpegTimeout: If the process exceeds ``timeout``.
        FFmpegError:   If the process exits non-zero.
    """
    # _quiet_ffmpeg_cmd returns a fresh list for ffmpeg, so no need to copy.
    cmd_with_progress = _quiet_ffmpeg_cmd(cmd)
    cmd_with_progress.insert(1, "-progress")
    cmd_with_progress.insert(2, "pipe:1")
    cmd_with_progress.insert(3, "-stats_period")
    cmd_with_progress.insert(4, "0.5")

    process = _start_group(cmd_with_progress)
    time_pattern = re.compile(r"out_time_ms=(\d+)")

    start = time.monotonic()

    try:
        while True:
            if timeout is not None and (time.monotonic() - start) > timeout:
                _kill_group(process)
                raise FFmpegTimeout(
                    f"{stage}: FFmpeg exceeded timeout of {timeout:.0f}s ({_describe_cmd(cmd)})"
                )

            line = process.stdout.readline() if process.stdout else ""
            if not line:
                break
            match = time_pattern.search(line)
            if match and duration > 0:
                time_seconds = int(match.group(1)) / 1_000_000
                progress_callback(min(time_seconds / duration, 1.0))

        remaining = None
        if timeout is not None:
            remaining = max(0.0, timeout - (time.monotonic() - start))
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            _kill_group(process)
            raise FFmpegTimeout(
                f"{stage}: FFmpeg exceeded timeout of {timeout:.0f}s ({_describe_cmd(cmd)})"
            ) from None
    except KeyboardInterrupt:
        _kill_group(process)
        raise

    if process.returncode != 0:
        stderr = process.stderr.read() if process.stderr else ""
        raise FFmpegError(
            f"{stage}: FFmpeg exited with code {process.returncode}.\n"
            f"Command: {_describe_cmd(cmd)}\n"
            f"Last stderr lines:\n{_tail(stderr)}"
        )

    progress_callback(1.0)
