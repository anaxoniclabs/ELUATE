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
