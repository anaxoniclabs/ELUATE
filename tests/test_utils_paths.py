# SPDX-License-Identifier: MIT
"""
Tests for eluate.utils.paths module.
"""

from pathlib import Path

import pytest

from eluate.utils.paths import (
    DEFAULT_MODEL,
    MODEL_PROFILES,
    ensure_dir,
    get_app_dir,
    get_bundled_config_path,
    get_checkpoint_path,
    get_checkpoint_url,
    get_config_path,
    get_model_paths,
    get_model_profile,
    get_output_dir,
    get_project_root,
    get_temp_dir,
    get_vendor_path,
    sanitize_filename,
)


class TestGetOutputDir:
    """Tests for get_output_dir function."""

    def test_returns_path(self):
        """Should return a Path object."""
        result = get_output_dir()
        assert isinstance(result, Path)

    def test_path_in_documents(self):
        """Should be in ~/Documents/ELUATE/."""
        result = get_output_dir()
        assert result.parent.name == "Documents"
        assert result.name == "ELUATE"

    def test_directory_exists(self):
        """Should create directory if needed."""
        result = get_output_dir()
        assert result.exists()
        assert result.is_dir()


class TestGetAppDir:
    """Tests for get_app_dir function."""

    def test_returns_path(self):
        """Should return a Path object."""
        result = get_app_dir()
        assert isinstance(result, Path)

    def test_path_is_hidden(self):
        """Should be ~/.eluate/."""
        result = get_app_dir()
        assert result.name == ".eluate"
        assert result.parent == Path.home()

    def test_directory_exists(self):
        """Should create directory if needed."""
        result = get_app_dir()
        assert result.exists()
        assert result.is_dir()


class TestGetModelPaths:
    """Tests for get_model_paths function."""

    def test_returns_tuple(self):
        """Should return tuple of two paths."""
        result = get_model_paths()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_paths_are_path_objects(self):
        """Should return Path objects."""
        checkpoint, config = get_model_paths()
        assert isinstance(checkpoint, Path)
        assert isinstance(config, Path)

    def test_checkpoint_path(self):
        """Checkpoint should be in models directory."""
        checkpoint, _ = get_model_paths()
        assert checkpoint.parent.name == "models"
        assert checkpoint.name == "checkpoint-multi.ckpt"

    def test_config_path(self):
        """Config should resolve to an existing YAML file."""
        _, config = get_model_paths()
        assert config.suffix == ".yaml"
        assert config.exists()


class TestGetTempDir:
    """Tests for get_temp_dir function."""

    def test_returns_path(self):
        """Should return a Path object."""
        result = get_temp_dir()
        assert isinstance(result, Path)

    def test_eluate_subdirectory(self):
        """Should be eluate subdirectory of system temp."""
        result = get_temp_dir()
        assert result.name == "eluate"

    def test_directory_exists(self):
        """Should create directory if needed."""
        result = get_temp_dir()
        assert result.exists()


class TestGetProjectRoot:
    """Tests for get_project_root function."""

    def test_returns_path(self):
        """Should return a Path object."""
        result = get_project_root()
        assert isinstance(result, Path)

    def test_contains_eluate_package(self):
        """Project root should contain eluate package."""
        result = get_project_root()
        assert (result / "eluate").exists()


class TestGetVendorPath:
    """Tests for get_vendor_path function."""

    def test_returns_path(self):
        """Should return a Path object."""
        result = get_vendor_path()
        assert isinstance(result, Path)

    def test_path_structure(self):
        """Should point to vendor/mss-training."""
        result = get_vendor_path()
        assert result.name == "mss-training"
        assert result.parent.name == "vendor"


class TestGetBundledConfigPath:
    """Tests for get_bundled_config_path function."""

    def test_resolves_bundled_yaml(self):
        """Should resolve the bandit_v2.yaml shipped under eluate/configs/."""
        result = get_bundled_config_path("bandit_v2.yaml")
        assert result is not None
        assert result.is_file()
        assert result.name == "bandit_v2.yaml"

    def test_missing_resource_returns_none(self):
        """Unknown filenames resolve to None rather than raising."""
        assert get_bundled_config_path("does-not-exist.yaml") is None


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_basic_filename(self):
        """Should preserve safe characters."""
        assert sanitize_filename("my_video") == "my_video"

    def test_removes_special_chars(self):
        """Should remove special characters."""
        assert sanitize_filename("file@#$%name") == "filename"

    def test_preserves_spaces(self):
        """Should preserve single spaces."""
        assert sanitize_filename("my video file") == "my video file"

    def test_collapses_multiple_spaces(self):
        """Should collapse multiple spaces to single."""
        assert sanitize_filename("my   video   file") == "my video file"

    def test_trims_whitespace(self):
        """Should trim leading/trailing whitespace."""
        assert sanitize_filename("  my video  ") == "my video"

    def test_max_length(self):
        """Should respect max_length parameter."""
        long_name = "a" * 100
        result = sanitize_filename(long_name, max_length=50)
        assert len(result) == 50

    def test_empty_becomes_untitled(self):
        """Empty result should become 'untitled'."""
        assert sanitize_filename("@#$%") == "untitled"

    def test_preserves_hyphens(self):
        """Should preserve hyphens."""
        assert sanitize_filename("my-video-file") == "my-video-file"

    def test_preserves_underscores(self):
        """Should preserve underscores."""
        assert sanitize_filename("my_video_file") == "my_video_file"

    def test_mixed_case(self):
        """Should preserve case."""
        assert sanitize_filename("MyVideoFile") == "MyVideoFile"


class TestModelProfiles:
    """Tests for MODEL_PROFILES registry and get_model_profile."""

    def test_default_is_bandit_v2(self):
        assert DEFAULT_MODEL == "bandit-v2"
        assert DEFAULT_MODEL in MODEL_PROFILES

    def test_bandit_v2_profile_shape(self):
        profile = MODEL_PROFILES["bandit-v2"]
        assert profile["arch"] == "bandit_v2"
        assert profile["sample_rate"] == 48000
        assert profile["noncommercial"] is False
        assert profile["supports_language"] is True

    def test_every_profile_has_required_keys(self):
        required = {
            "arch",
            "config",
            "checkpoint",
            "sample_rate",
            "license",
            "noncommercial",
            "supports_language",
            "zenodo_record",
        }
        for name, profile in MODEL_PROFILES.items():
            missing = required - set(profile)
            assert not missing, f"{name} missing keys: {missing}"

    def test_get_model_profile_returns_default(self):
        assert get_model_profile() == MODEL_PROFILES[DEFAULT_MODEL]

    def test_get_model_profile_unknown_raises(self):
        with pytest.raises(ValueError):
            get_model_profile("does-not-exist")


class TestGetConfigPath:
    """Tests for get_config_path function."""

    def test_default_model(self):
        result = get_config_path()
        assert result.exists()
        assert result.suffix == ".yaml"

    def test_bandit_v2_explicit(self):
        result = get_config_path("bandit-v2")
        assert result.exists()

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            get_config_path("unknown-model")


class TestGetCheckpointPath:
    """Tests for get_checkpoint_path function."""

    def test_default(self):
        result = get_checkpoint_path()
        assert result.name == "checkpoint-multi.ckpt"

    def test_language_variant(self):
        result = get_checkpoint_path("eng", model="bandit-v2")
        assert result.name == "checkpoint-eng.ckpt"

    def test_invalid_lang_key_raises(self):
        with pytest.raises(ValueError):
            get_checkpoint_path("xyz", model="bandit-v2")

    def test_in_models_dir(self):
        result = get_checkpoint_path(model="bandit-v2")
        assert result.parent.name == "models"


class TestGetCheckpointUrl:
    """Tests for get_checkpoint_url function."""

    def test_bandit_v2_default(self):
        url = get_checkpoint_url()
        assert "12701995" in url
        assert "checkpoint-multi.ckpt" in url

    def test_bandit_v2_language(self):
        url = get_checkpoint_url("eng", model="bandit-v2")
        assert "checkpoint-eng.ckpt" in url


class TestGetModelPathsWithModel:
    """Tests for get_model_paths with new model parameter."""

    def test_default_backwards_compat(self):
        checkpoint, config = get_model_paths()
        assert checkpoint.name == "checkpoint-multi.ckpt"
        assert config.exists()

    def test_language_key_backwards_compat(self):
        checkpoint, _ = get_model_paths("eng")
        assert checkpoint.name == "checkpoint-eng.ckpt"


class TestEnsureDir:
    """Tests for ensure_dir function."""

    def test_creates_directory(self, temp_dir):
        """Should create directory if it doesn't exist."""
        new_dir = temp_dir / "new_directory"
        assert not new_dir.exists()

        ensure_dir(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_returns_same_path(self, temp_dir):
        """Should return the same path for chaining."""
        new_dir = temp_dir / "test_dir"
        result = ensure_dir(new_dir)
        assert result == new_dir

    def test_handles_existing_directory(self, temp_dir):
        """Should not fail if directory already exists."""
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()

        result = ensure_dir(existing_dir)

        assert existing_dir.exists()
        assert result == existing_dir

    def test_creates_nested_directories(self, temp_dir):
        """Should create nested directories."""
        nested = temp_dir / "a" / "b" / "c"

        ensure_dir(nested)

        assert nested.exists()
        assert nested.is_dir()
