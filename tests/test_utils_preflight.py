# SPDX-License-Identifier: MIT
"""
Tests for startup preflight checks.
"""

import hashlib

import pytest

import eluate
from eluate.utils.preflight import ensure_checkpoint, run_preflight


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_ensure_checkpoint_verifies_existing_declared_digest(tmp_path, monkeypatch):
    ckpt = tmp_path / "checkpoint-multi.ckpt"
    data = b"known-good-checkpoint"
    ckpt.write_bytes(data)

    monkeypatch.setattr(
        "eluate.utils.preflight.get_checkpoint_path",
        lambda key, model="bandit-v2": ckpt,
    )
    monkeypatch.setattr(
        "eluate.utils.preflight.get_checkpoint_sha256",
        lambda key, model="bandit-v2": _sha(data),
    )

    assert ensure_checkpoint("multi") == ckpt


def test_ensure_checkpoint_rejects_existing_mismatched_digest(tmp_path, monkeypatch):
    ckpt = tmp_path / "checkpoint-multi.ckpt"
    ckpt.write_bytes(b"tampered")

    monkeypatch.setattr(
        "eluate.utils.preflight.get_checkpoint_path",
        lambda key, model="bandit-v2": ckpt,
    )
    monkeypatch.setattr(
        "eluate.utils.preflight.get_checkpoint_sha256",
        lambda key, model="bandit-v2": _sha(b"expected"),
    )

    with pytest.raises(SystemExit) as excinfo:
        ensure_checkpoint("multi")

    assert excinfo.value.code == 1


def test_api_preflight_rejects_existing_mismatched_digest_before_ffmpeg(tmp_path, monkeypatch):
    ckpt = tmp_path / "checkpoint-multi.ckpt"
    ckpt.write_bytes(b"tampered")
    config = tmp_path / "bandit_v2.yaml"
    config.write_text("dummy: true")

    monkeypatch.setattr(
        "eluate.utils.preflight.get_checkpoint_path",
        lambda key, model="bandit-v2": ckpt,
    )
    monkeypatch.setattr(
        "eluate.utils.preflight.get_checkpoint_sha256",
        lambda key, model="bandit-v2": _sha(b"expected"),
    )
    monkeypatch.setattr("eluate.utils.preflight.get_config_path", lambda: config)

    ffmpeg_checked = False

    def fake_check_ffmpeg_available():
        nonlocal ffmpeg_checked
        ffmpeg_checked = True
        return True

    monkeypatch.setattr(
        "eluate.core.extractor.check_ffmpeg_available",
        fake_check_ffmpeg_available,
    )

    with pytest.raises(eluate.EluateError, match="SHA256 mismatch"):
        run_preflight("multi", download_if_missing=False)

    assert ffmpeg_checked is False


def _install_valid_checkpoint(tmp_path, monkeypatch):
    """Plant an installed, integrity-passing checkpoint so preflight gets
    past the checkpoint step and on to config/FFmpeg."""
    ckpt = tmp_path / "checkpoint-multi.ckpt"
    ckpt.write_bytes(b"good")
    monkeypatch.setattr(
        "eluate.utils.preflight.get_checkpoint_path",
        lambda key, model="bandit-v2": ckpt,
    )
    # Empty digest → integrity check is skipped (matches undeclared-SHA path).
    monkeypatch.setattr(
        "eluate.utils.preflight.get_checkpoint_sha256",
        lambda key, model="bandit-v2": "",
    )


def test_library_mode_missing_ffmpeg_raises_not_systemexit(tmp_path, monkeypatch):
    """The headline contract bug: in library mode (download_if_missing=False)
    a missing FFmpeg must raise the typed FFmpegNotFoundError, never
    SystemExit — otherwise it is uncatchable through EluateError and would
    abort a notebook cell or server worker."""
    _install_valid_checkpoint(tmp_path, monkeypatch)
    config = tmp_path / "bandit_v2.yaml"
    config.write_text("dummy: true")
    monkeypatch.setattr("eluate.utils.preflight.get_config_path", lambda: config)
    monkeypatch.setattr(
        "eluate.core.extractor.check_ffmpeg_available",
        lambda: False,
    )

    with pytest.raises(eluate.FFmpegNotFoundError):
        run_preflight("multi", download_if_missing=False)

    # FFmpegNotFoundError is catchable through the documented hierarchy.
    assert issubclass(eluate.FFmpegNotFoundError, eluate.EluateError)


def test_library_mode_missing_config_raises_file_not_found(tmp_path, monkeypatch):
    """In library mode a missing config propagates as an unwrapped
    FileNotFoundError (per the API's stdlib-exceptions contract), not
    SystemExit."""
    _install_valid_checkpoint(tmp_path, monkeypatch)

    def _no_config():
        raise FileNotFoundError("config gone")

    monkeypatch.setattr("eluate.utils.preflight.get_config_path", _no_config)

    with pytest.raises(FileNotFoundError, match="config gone"):
        run_preflight("multi", download_if_missing=False)


def test_cli_mode_missing_ffmpeg_still_exits(tmp_path, monkeypatch):
    """CLI mode (default download_if_missing=True) keeps the sys.exit(1)
    behaviour so first-run UX and the Rich panel are unchanged."""
    _install_valid_checkpoint(tmp_path, monkeypatch)
    config = tmp_path / "bandit_v2.yaml"
    config.write_text("dummy: true")
    monkeypatch.setattr("eluate.utils.preflight.get_config_path", lambda: config)
    monkeypatch.setattr(
        "eluate.core.extractor.check_ffmpeg_available",
        lambda: False,
    )

    with pytest.raises(SystemExit) as excinfo:
        run_preflight("multi", download_if_missing=True)

    assert excinfo.value.code == 1
