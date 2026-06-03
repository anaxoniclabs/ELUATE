# SPDX-License-Identifier: MIT
"""
JSONL telemetry for processing stages and config events.

Writes one JSON object per line to ~/.eluate/telemetry.jsonl. Captures wall
time, process peak RSS, and MPS memory at stage boundaries so the open-source
release has a paper trail of runtime behavior across machines.

Notes on interpretation:

- ``peak_rss_mb`` is ``ru_maxrss`` at stage exit. ``ru_maxrss`` is
  process-lifetime-monotonic: once the process has hit some peak, that value
  sticks for the rest of the run. So ``peak_rss_mb`` is the overall-peak-so-far
  at the moment the stage finishes, not that stage's standalone peak.
  ``delta_peak_rss_mb`` (peak at exit minus peak at entry) is the amount this
  stage pushed the overall peak upward — which is the useful number for
  "where did the memory go" attribution.

- ``ru_maxrss`` units differ by OS: bytes on Darwin, kilobytes on Linux.
  ``_ru_maxrss_to_mb`` handles both via a ``sys.platform`` branch, so the
  reported MB value is consistent across platforms.

- The log is rotated when it exceeds ``MAX_LOG_BYTES`` (10 MB) so the file
  cannot grow without bound over long-term use. One backup is kept at
  ``telemetry.jsonl.1``; the previous backup (if any) is discarded.

- Enabling: telemetry is **off by default**. Set ``ELUATE_TELEMETRY=1`` in
  the environment to enable. The log is always local — nothing is sent
  anywhere — but it is opt-in so users aren't surprised by a file appearing
  under ``~/.eluate/``.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from eluate.utils.paths import get_app_dir


def _ru_maxrss_to_mb() -> float:
    """Return current ``ru_maxrss`` in MB, accounting for OS units."""
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return maxrss / (1024**2)
    # Linux and most others report kilobytes.
    return maxrss / 1024


def _mps_memory_mb() -> tuple[Optional[float], Optional[float]]:
    """Return (current, driver) MPS allocation in MB, or (None, None)."""
    # Import lazily so ``--version`` and ``info`` don't pay torch's import
    # cost when telemetry never fires.
    import torch

    if not torch.backends.mps.is_available():
        return None, None
    try:
        current = torch.mps.current_allocated_memory() / (1024**2)
        driver = torch.mps.driver_allocated_memory() / (1024**2)
        return current, driver
    except Exception:
        return None, None


@dataclass
class StageHandle:
    """Handle passed to the ``record_stage`` context body.

    Use ``.add(key, value)`` to attach extra metrics that get merged into
    the stage's ``extra`` field on exit.
    """

    extra: dict[str, Any] = field(default_factory=dict)

    def add(self, key: str, value: Any) -> None:
        self.extra[key] = value


MAX_LOG_BYTES = 10 * 1024 * 1024  # Rotate when telemetry log passes 10 MB.


class _Recorder:
    def __init__(self) -> None:
        self._enabled = os.environ.get("ELUATE_TELEMETRY", "0") == "1"
        self._log_path: Optional[Path] = None
        if self._enabled:
            try:
                self._log_path = get_app_dir() / "telemetry.jsonl"
            except Exception:
                # If we can't resolve the app dir (e.g. read-only home in
                # some sandbox), silently disable rather than crash the run.
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _maybe_rotate(self) -> None:
        if self._log_path is None:
            return
        try:
            size = self._log_path.stat().st_size
        except FileNotFoundError:
            return
        except OSError:
            return
        if size < MAX_LOG_BYTES:
            return
        backup = self._log_path.with_suffix(self._log_path.suffix + ".1")
        try:
            if backup.exists():
                backup.unlink()
            self._log_path.replace(backup)
        except OSError:
            # If rotation fails we give up quietly — the existing file
            # will keep growing, but the run must not break.
            pass

    def write(self, record: dict[str, Any]) -> None:
        if not self._enabled or self._log_path is None:
            return
        try:
            self._maybe_rotate()
            with self._log_path.open("a", encoding="utf-8") as f:
                json.dump(record, f, default=str)
                f.write("\n")
        except Exception:
            # Telemetry must never break a run.
            pass


_recorder: Optional[_Recorder] = None


def _get_recorder() -> _Recorder:
    global _recorder
    if _recorder is None:
        _recorder = _Recorder()
    return _recorder


@contextmanager
def record_stage(name: str, extra: Optional[dict[str, Any]] = None) -> Iterator[StageHandle]:
    """Record a processing stage to the telemetry log.

    Emits two records: ``stage.start`` on entry and ``stage.complete`` on
    exit (or ``stage.error`` if the block raised). The exit record carries
    wall time, process peak RSS, the stage's delta contribution to that
    peak, and MPS memory snapshot.

    The yielded handle accepts ``.add(key, value)`` to attach stage-specific
    metrics (e.g. input duration, output size) before exit.
    """
    recorder = _get_recorder()
    handle = StageHandle(extra=dict(extra or {}))

    if not recorder.enabled:
        yield handle
        return

    started_at = time.time()
    peak_rss_entry = _ru_maxrss_to_mb()

    recorder.write(
        {
            "ts": started_at,
            "event": "stage.start",
            "stage": name,
        }
    )

    error: Optional[BaseException] = None
    try:
        yield handle
    except BaseException as exc:
        error = exc
        raise
    finally:
        ended_at = time.time()
        peak_rss_exit = _ru_maxrss_to_mb()
        mps_current, mps_driver = _mps_memory_mb()

        record: dict[str, Any] = {
            "ts": ended_at,
            "event": "stage.complete" if error is None else "stage.error",
            "stage": name,
            "wall_seconds": round(ended_at - started_at, 3),
            "peak_rss_mb": round(peak_rss_exit, 1),
            "delta_peak_rss_mb": round(peak_rss_exit - peak_rss_entry, 1),
            "mps_current_mb": round(mps_current, 1) if mps_current is not None else None,
            "mps_driver_mb": round(mps_driver, 1) if mps_driver is not None else None,
        }
        if handle.extra:
            record["extra"] = handle.extra
        if error is not None:
            record["error_type"] = type(error).__name__
            record["error_message"] = str(error)[:500]
        recorder.write(record)


def record_event(event_type: str, payload: dict[str, Any]) -> None:
    """Record a free-form event (e.g. effective config) to the log."""
    recorder = _get_recorder()
    if not recorder.enabled:
        return
    recorder.write(
        {
            "ts": time.time(),
            "event": event_type,
            "payload": payload,
        }
    )


def record_run_start() -> None:
    """Emit a ``run.start`` record with platform/python info.

    Call once per invocation from the CLI; safe to call more than once
    (each call produces a record).
    """
    # Gate the torch import on the recorder being enabled — otherwise every
    # CLI invocation (including ``--version`` and ``info``) would pay torch's
    # import cost just to build a payload that gets discarded in record_event.
    if not _get_recorder().enabled:
        return
    import torch

    record_event(
        "run.start",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
        },
    )
