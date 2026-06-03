# SPDX-License-Identifier: MIT
"""
Path management for Eluate.
"""

import tempfile
from importlib import resources
from pathlib import Path

# Zenodo record 12701995 — Bandit v2 language-specific checkpoints
CHECKPOINT_KEYS = ["multi", "eng", "deu", "fra", "spa", "cmn", "fao"]

CHECKPOINT_URLS: dict[str, str] = {
    key: f"https://zenodo.org/records/12701995/files/checkpoint-{key}.ckpt?download=1"
    for key in CHECKPOINT_KEYS
}


# Expected SHA256 digests for downloaded checkpoints.
#
# Looked up by ``_ensure_checkpoint`` in the CLI to verify model downloads.
# An empty string ("") or missing key skips verification for that checkpoint
# — the file is still written atomically via ``download_file`` so an
# interrupted download can't leave a half-written ``.ckpt`` at the final
# path, but integrity against upstream is not confirmed.
#
# To populate a new entry: after a known-good download, run
#     shasum -a 256 ~/.eluate/models/<filename>
# and paste the hex digest below.
CHECKPOINT_SHA256: dict[tuple[str, str], str] = {
    ("bandit-v2", "multi"): "abcfccf65446752a057f4a302c941479a54b7560ebf8d7bca039d2ea98e64cfc",
    # Remaining bandit-v2 language variants have not been verified yet.
    # They download without an integrity check until a digest is recorded.
    # ("bandit-v2", "eng"):   "",
    # ("bandit-v2", "deu"):   "",
    # ("bandit-v2", "fra"):   "",
    # ("bandit-v2", "spa"):   "",
    # ("bandit-v2", "cmn"):   "",
    # ("bandit-v2", "fao"):   "",
}


def get_checkpoint_sha256(key: str = "multi", model: str = "bandit-v2") -> str:
    """Return the expected SHA256 for a checkpoint, or ``""`` if unknown."""
    return CHECKPOINT_SHA256.get((model, key), "")


# Model profile registry. Eluate currently ships a single model
# (Bandit v2, CC-BY-SA 4.0, 48 kHz). The registry shape is preserved so
# additional models can be added in the future without churning callers.
MODEL_PROFILES: dict[str, dict] = {
    "bandit-v2": {
        "arch": "bandit_v2",
        "config": "bandit_v2.yaml",
        "checkpoint": "checkpoint-{lang}.ckpt",
        "sample_rate": 48000,
        "license": "CC-BY-SA 4.0",
        "noncommercial": False,
        "supports_language": True,
        "zenodo_record": "12701995",
        "download_size_mb": 450,
    },
}

DEFAULT_MODEL = "bandit-v2"


def get_model_profile(model: str = DEFAULT_MODEL) -> dict:
    """Return the profile dict for a model. Raises ValueError on unknown model."""
    if model not in MODEL_PROFILES:
        raise ValueError(f"Unknown model '{model}'. Valid options: {list(MODEL_PROFILES)}")
    return MODEL_PROFILES[model]


def get_output_dir() -> Path:
    """
    Get the output directory for processed videos.

    Default: ~/Documents/ELUATE/

    Returns:
        Path to output directory (created if needed)
    """
    documents = Path.home() / "Documents"
    eluate_dir = documents / "ELUATE"
    eluate_dir.mkdir(parents=True, exist_ok=True)
    return eluate_dir


def get_app_dir() -> Path:
    """
    Get the Eluate application data directory.

    Default: ~/.eluate/

    Returns:
        Path to app directory (created if needed)
    """
    app_dir = Path.home() / ".eluate"
    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir


def get_config_path(model: str = DEFAULT_MODEL) -> Path:
    """
    Resolve the config YAML for a given model.

    Prefers the user-installed copy at ~/.eluate/models/config_<name>.yaml
    (or the legacy ~/.eluate/models/config_bandit_v2.yaml for bandit-v2),
    so existing installs that ran scripts/install.sh keep working.
    Falls back to the bundled copy under eluate/configs/ shipped inside
    the wheel — that path is what makes `pip install eluate` work without
    a prior install-script run.

    Args:
        model: Model profile key (see MODEL_PROFILES)

    Returns:
        Path to existing config file

    Raises:
        FileNotFoundError: If neither location has the config
    """
    profile = get_model_profile(model)
    config_filename = profile["config"]

    models_dir = get_app_dir() / "models"

    # User-installed location. Historically we used "config_bandit_v2.yaml";
    # keep that exact filename for bandit-v2 to preserve existing installs.
    if model == "bandit-v2":
        user_config = models_dir / "config_bandit_v2.yaml"
    else:
        user_config = models_dir / config_filename

    if user_config.exists():
        return user_config

    bundled = get_bundled_config_path(config_filename)
    if bundled is not None:
        return bundled

    raise FileNotFoundError(
        f"Config '{config_filename}' not found for model '{model}'. "
        "Reinstall eluate or run ./scripts/install.sh from a clone."
    )


def get_bundled_config_path(filename: str) -> Path | None:
    """
    Return the on-disk path of a config YAML shipped inside the package.

    Wheels installed by pip are unpacked to site-packages, so the
    Traversable returned by importlib.resources resolves to a real
    filesystem Path. Returns None if the resource is missing — callers
    decide how to surface that.
    """
    resource = resources.files("eluate.configs").joinpath(filename)
    return Path(str(resource)) if resource.is_file() else None


def get_checkpoint_path(key: str = "multi", model: str = DEFAULT_MODEL) -> Path:
    """
    Get the path to a specific checkpoint.

    For bandit-v2, `key` is the language variant (multi, eng, deu, ...).
    Models that don't support language variants ignore `key`.

    Args:
        key: Language variant for bandit-v2. Ignored otherwise.
        model: Model profile key (see MODEL_PROFILES)

    Returns:
        Path to the checkpoint file under ~/.eluate/models/
    """
    profile = get_model_profile(model)
    models_dir = get_app_dir() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if profile["supports_language"]:
        if key not in CHECKPOINT_KEYS:
            raise ValueError(f"Unknown checkpoint key '{key}'. Valid options: {CHECKPOINT_KEYS}")
        return models_dir / profile["checkpoint"].format(lang=key)

    return models_dir / profile["checkpoint"]


def get_checkpoint_url(key: str = "multi", model: str = DEFAULT_MODEL) -> str:
    """
    Resolve the download URL for a checkpoint.

    Args:
        key: Language variant for bandit-v2. Ignored otherwise.
        model: Model profile key (see MODEL_PROFILES)

    Returns:
        Direct Zenodo download URL
    """
    profile = get_model_profile(model)
    record = profile["zenodo_record"]

    if profile["supports_language"]:
        if key not in CHECKPOINT_KEYS:
            raise ValueError(f"Unknown checkpoint key '{key}'. Valid options: {CHECKPOINT_KEYS}")
        return f"https://zenodo.org/records/{record}/files/checkpoint-{key}.ckpt?download=1"

    remote = profile["remote_filename"]
    return f"https://zenodo.org/records/{record}/files/{remote}?download=1"


def get_model_paths(checkpoint_key: str = "multi", model: str = DEFAULT_MODEL) -> tuple[Path, Path]:
    """
    Get paths to model checkpoint and config.

    Args:
        checkpoint_key: Language variant for bandit-v2 (default "multi")
        model: Model profile key (default "bandit-v2")

    Returns:
        Tuple of (checkpoint_path, config_path)
    """
    checkpoint = get_checkpoint_path(checkpoint_key, model=model)
    config = get_config_path(model=model)
    return checkpoint, config


def get_temp_dir() -> Path:
    """
    Get temp directory for processing.

    Uses system temp with eluate subdirectory.

    Returns:
        Path to temp directory (created if needed)
    """
    temp_base = Path(tempfile.gettempdir()) / "eluate"
    temp_base.mkdir(parents=True, exist_ok=True)
    return temp_base


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to the eluate project root
    """
    # eluate/utils/paths.py -> eluate/utils -> eluate -> project_root
    return Path(__file__).parent.parent.parent


def get_vendor_path() -> Path:
    """
    Get path to vendor directory containing submodules.

    Returns:
        Path to vendor/mss-training
    """
    return get_project_root() / "vendor" / "mss-training"


def sanitize_filename(name: str, max_length: int = 80) -> str:
    """
    Sanitize a string for use as a filename.

    Args:
        name: Original filename
        max_length: Maximum length of result

    Returns:
        Safe filename string
    """
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_")
    result = "".join(c for c in name if c in safe_chars)
    result = result.strip()[:max_length]
    while "  " in result:
        result = result.replace("  ", " ")
    return result or "untitled"


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists.

    Args:
        path: Directory path

    Returns:
        The same path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
