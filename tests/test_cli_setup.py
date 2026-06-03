# SPDX-License-Identifier: MIT
"""
Tests for the CLI setup command.
"""

import sys

import pytest

from eluate.utils.preflight import PreflightConfig


def test_cli_setup_downloads_selected_checkpoint(tmp_path, monkeypatch):
    from eluate import cli

    calls = []

    def fake_ensure_checkpoint(checkpoint, console):
        calls.append(checkpoint)
        return tmp_path / f"checkpoint-{checkpoint}.ckpt"

    monkeypatch.setattr("eluate.cli.ensure_checkpoint", fake_ensure_checkpoint)
    monkeypatch.setattr(
        "eluate.cli.get_checkpoint_path",
        lambda checkpoint: tmp_path / f"checkpoint-{checkpoint}.ckpt",
    )
    monkeypatch.setattr(
        "eluate.utils.paths.get_bundled_config_path",
        lambda filename: None,
    )
    monkeypatch.setattr("eluate.utils.paths.get_app_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "eluate.cli.run_preflight",
        lambda *args, **kwargs: pytest.fail("setup should not run video preflight"),
    )
    monkeypatch.setattr(sys, "argv", ["eluate", "--checkpoint", "eng", "setup"])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 0
    assert calls == ["eng"]


def test_cli_build_session_reuses_preflight_config(tmp_path, monkeypatch):
    from eluate import cli

    preflight = PreflightConfig(
        model_path=tmp_path / "checkpoint.ckpt",
        config_path=tmp_path / "config.yaml",
        arch="bandit_v2",
    )
    session = cli._build_session(
        force=False,
        device_override=None,
        audio_codec="aac",
        audio_bitrate="256k",
        checkpoint="multi",
        preflight_config=preflight,
    )

    monkeypatch.setattr(
        "eluate.api.run_preflight",
        lambda *args, **kwargs: pytest.fail("preflight should already be cached"),
    )

    pipeline = session._ensure_pipeline(tmp_path)

    assert pipeline.model_path == preflight.model_path
    assert pipeline.config_path == preflight.config_path
