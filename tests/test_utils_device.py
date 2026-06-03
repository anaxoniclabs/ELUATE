# SPDX-License-Identifier: MIT
"""
Tests for eluate.utils.device module.
"""

from unittest.mock import MagicMock, patch

import torch

from eluate.utils.device import (
    clear_mps_cache,
    configure_mps_settings,
    get_device_info,
    get_memory_info,
    get_optimal_device,
    get_system_memory_gb,
)


class TestGetOptimalDevice:
    """Tests for get_optimal_device function."""

    def test_returns_torch_device(self, mock_mps_unavailable):
        """Should return a torch.device object."""
        result = get_optimal_device()
        assert isinstance(result, torch.device)

    def test_returns_mps_when_available(self, mock_mps_available):
        """Should return MPS device when available."""
        result = get_optimal_device()
        assert result.type == "mps"

    def test_returns_cpu_when_mps_unavailable(self, mock_mps_unavailable):
        """Should return CPU device when MPS unavailable."""
        result = get_optimal_device()
        assert result.type == "cpu"

    def test_returns_cpu_when_mps_not_built(self):
        """Should return CPU when MPS available but not built."""
        with (
            patch("torch.backends.mps.is_available", return_value=True),
            patch("torch.backends.mps.is_built", return_value=False),
        ):
            result = get_optimal_device()
            assert result.type == "cpu"


class TestConfigureMpsSettings:
    """Tests for configure_mps_settings function."""

    def test_sets_environment_variable(self):
        """Should set PYTORCH_ENABLE_MPS_FALLBACK."""
        import os

        configure_mps_settings()
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


class TestGetMemoryInfo:
    """Tests for get_memory_info function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        result = get_memory_info()
        assert isinstance(result, dict)

    def test_handles_vm_stat_success(self):
        """Should parse vm_stat output correctly."""
        mock_output = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               50000.
Pages active:                            100000.
Pages inactive:                           30000.
Pages wired down:                         20000."""

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=mock_output, returncode=0)
            result = get_memory_info()

            assert "free_gb" in result
            assert "active_gb" in result
            assert "inactive_gb" in result
            assert "wired_gb" in result

    def test_handles_error(self):
        """Should return error dict on failure."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("vm_stat failed")
            result = get_memory_info()
            assert "error" in result


class TestGetSystemMemoryGb:
    """Tests for get_system_memory_gb function."""

    def test_returns_float(self):
        """Should return a float."""
        result = get_system_memory_gb()
        assert isinstance(result, float)

    def test_parses_sysctl_output(self):
        """Should parse sysctl output correctly."""
        # 16 GB in bytes
        mock_bytes = str(16 * 1024 * 1024 * 1024)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=mock_bytes + "\n", returncode=0)
            result = get_system_memory_gb()
            assert result == 16.0

    def test_handles_error(self):
        """Should return 0.0 on error."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("sysctl failed")
            result = get_system_memory_gb()
            assert result == 0.0


class TestClearMpsCache:
    """Tests for clear_mps_cache function."""

    def test_clears_when_mps_available(self):
        """Should call torch.mps.empty_cache when MPS available."""
        with (
            patch("torch.backends.mps.is_available", return_value=True),
            patch("torch.mps.empty_cache") as mock_empty,
        ):
            clear_mps_cache()
            mock_empty.assert_called_once()

    def test_no_error_when_mps_unavailable(self, mock_mps_unavailable):
        """Should not error when MPS unavailable."""
        clear_mps_cache()  # Should not raise


class TestGetDeviceInfo:
    """Tests for get_device_info function."""

    def test_returns_dict(self, mock_mps_unavailable):
        """Should return a dictionary."""
        result = get_device_info()
        assert isinstance(result, dict)

    def test_contains_device_info(self, mock_mps_unavailable):
        """Should contain device information."""
        result = get_device_info()

        assert "device" in result
        assert "device_type" in result
        assert "mps_available" in result
        assert "mps_built" in result
        assert "pytorch_version" in result
        assert "total_memory_gb" in result

    def test_device_type_is_cpu_without_mps(self, mock_mps_unavailable):
        """Should report CPU device when MPS unavailable."""
        result = get_device_info()
        assert result["device_type"] == "cpu"

    def test_includes_pytorch_version(self, mock_mps_unavailable):
        """Should include PyTorch version."""
        result = get_device_info()
        assert result["pytorch_version"] == torch.__version__
