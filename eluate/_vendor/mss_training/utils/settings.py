# SPDX-License-Identifier: MIT
"""
Slim adaptation of ``utils/settings.py`` from the upstream framework.

Eluate only needs config loading plus the ``bandit_v2`` branch of the
model dispatch. This file keeps the names and signatures of the two
functions eluate calls (``load_config``, ``get_model_from_config``)
identical to upstream so the call site in
``eluate/core/separator.py`` stays unchanged in spirit, while the
import target moves from ``utils.settings`` to
``eluate._vendor.mss_training.utils.settings``.

Other ``model_type`` values raise ``ValueError`` here. Eluate only
ships bandit_v2 weights — any caller asking for another architecture
is misconfigured, and a clear error is better than a missing-module
``ImportError`` from the wheel.
"""

from __future__ import annotations

from typing import Tuple, Union

import yaml
from ml_collections import ConfigDict
from omegaconf import OmegaConf
from torch import nn


def load_config(model_type: str, config_path: str) -> Union[ConfigDict, OmegaConf]:
    """Load a model configuration file.

    Mirrors upstream ``utils.settings.load_config``: returns an
    OmegaConf object for ``htdemucs`` and a YAML-parsed
    ``ConfigDict`` for everything else (including ``bandit_v2``).
    """
    try:
        with open(config_path, "r") as f:
            if model_type == "htdemucs":
                config = OmegaConf.load(config_path)
            else:
                config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")


def get_model_from_config(
    model_type: str, config_path: str
) -> Tuple[nn.Module, Union[ConfigDict, OmegaConf]]:
    """Instantiate a model from its config file.

    Eluate only ships the ``bandit_v2`` branch from the upstream
    dispatch. Other model types raise ``ValueError`` to surface
    misconfiguration explicitly rather than failing later with an
    ``ImportError`` for a module that was never bundled.
    """
    config = load_config(model_type, config_path)
    if "model_type" in config.training:
        model_type = config.training.model_type

    if model_type == "bandit_v2":
        from eluate._vendor.mss_training.models.bandit_v2.bandit import Bandit

        model = Bandit(**config.kwargs)
        return model, config

    raise ValueError(
        f"eluate bundles only the 'bandit_v2' model architecture; "
        f"requested model_type={model_type!r}. "
        "Other architectures from the upstream MSS-Training framework "
        "are not shipped in the eluate wheel."
    )
