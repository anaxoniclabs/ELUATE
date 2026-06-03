# SPDX-License-Identifier: MIT
"""
Robust file downloads with atomic rename, optional SHA256 verification,
resume support, and exponential-backoff retries.

Used by Eluate to fetch model checkpoints from Zenodo on first run. All
downloads go through ``download_file()``; callers pass a URL, destination
path, and (optionally) an expected SHA256 hex digest. The function:

- Writes to ``<dest>.part`` so a partial download never leaves a broken
  file at ``<dest>``.
- Attempts HTTP Range-based resume when ``<dest>.part`` already exists,
  falling back to a clean restart if the server refuses partial content.
- Retries transient network errors (up to ``retries`` times) with
  exponential backoff.
- Verifies SHA256 if ``expected_sha256`` is provided. Rejects the file
  on mismatch and deletes the partial.
- Atomically renames ``<dest>.part`` to ``<dest>`` on success.

Expected SHA256 hashes for Eluate's built-in checkpoints are declared in
``eluate.utils.paths.CHECKPOINT_SHA256`` (populate as they become known;
an empty string means "verification skipped, file still written atomically").
"""

from __future__ import annotations

import hashlib
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional


class DownloadError(RuntimeError):
    """Download failed after retries, or integrity check mismatched."""


ProgressCallback = Callable[[int, Optional[int]], None]
"""Progress callback: ``(bytes_downloaded, total_bytes_or_None)``."""


def sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 hex digest for ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _open_url_with_range(
    url: str, start_byte: int, timeout: float
) -> tuple[Any, int, Optional[int]]:
    """Open the URL, requesting a byte range if ``start_byte > 0``.

    Returns ``(response, actual_start, total_size)``. ``actual_start`` is
    0 if the server ignored the Range header (full restart) and
    ``start_byte`` if it honored it.
    """
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "eluate-downloader/1"},
    )
    if start_byte > 0:
        req.add_header("Range", f"bytes={start_byte}-")

    response = urllib.request.urlopen(req, timeout=timeout)
    status = getattr(response, "status", None) or response.getcode()

    if start_byte > 0 and status == 206:
        # Partial content honored. Total size = start + Content-Length.
        length = response.headers.get("Content-Length")
        total = (start_byte + int(length)) if length is not None else None
        return response, start_byte, total

    # Either no range requested, or server returned 200 (ignored range).
    length = response.headers.get("Content-Length")
    total = int(length) if length is not None else None
    return response, 0, total


def download_file(
    url: str,
    dest: Path,
    expected_sha256: Optional[str] = None,
    retries: int = 3,
    backoff_base: float = 2.0,
    chunk_size: int = 1024 * 64,
    timeout: float = 30.0,
    progress_callback: Optional[ProgressCallback] = None,
) -> Path:
    """Download ``url`` to ``dest`` safely.

    Args:
        url: HTTP(S) URL to download.
        dest: Final path. A sibling ``<dest>.part`` is used during transfer.
        expected_sha256: If set, must match the downloaded file's hex digest.
        retries: Max retry attempts on transient network errors.
        backoff_base: Exponential backoff base in seconds (wait = base ** i).
        chunk_size: Read chunk size in bytes.
        timeout: Per-request network timeout in seconds.
        progress_callback: Optional callback invoked with ``(bytes, total)``
            roughly every ``chunk_size`` bytes. ``total`` may be ``None``.

    Returns:
        ``dest`` on success.

    Raises:
        DownloadError: On repeated failure or hash mismatch.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    last_error: Optional[BaseException] = None

    for attempt in range(retries):
        try:
            start_byte = part.stat().st_size if part.exists() else 0
            response, actual_start, total = _open_url_with_range(url, start_byte, timeout)
            mode = "ab" if actual_start > 0 else "wb"
            downloaded = actual_start

            try:
                with part.open(mode) as out:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback is not None:
                            progress_callback(downloaded, total)
            finally:
                response.close()

            # If we know the total size, verify we actually finished.
            if total is not None and downloaded < total:
                raise DownloadError(f"Download truncated: got {downloaded} of {total} bytes")

            # Verify integrity if a hash was declared.
            if expected_sha256:
                actual = sha256_of_file(part)
                if actual.lower() != expected_sha256.lower():
                    part.unlink(missing_ok=True)
                    raise DownloadError(
                        f"SHA256 mismatch. Expected {expected_sha256}, got {actual}"
                    )

            # Atomic rename — final file appears all-or-nothing.
            part.replace(dest)
            return dest

        except DownloadError:
            # Integrity failures are not retried — bad hash means either
            # the upstream changed or the URL served different content.
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(backoff_base**attempt)
                continue
            break

    raise DownloadError(f"Failed to download {url} after {retries} attempts: {last_error}")
