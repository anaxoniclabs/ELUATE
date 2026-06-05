# SPDX-License-Identifier: MIT
"""
Device detection and configuration for inference.

Priority order: explicit override > CUDA > MPS (Apple Silicon) > CPU.
"""

import os
import subprocess
import sys
from typing import Literal, Optional

import torch

DeviceType = Literal["cuda", "mps", "cpu"]


def get_optimal_device(override: Optional[str] = None) -> torch.device:
    """
    Detect and return the optimal device for inference.

    Priority: explicit override > CUDA (if available) > MPS (Apple Silicon)
    > CPU. A non-trivial ``override`` is returned as-is, validated against
    availability — "cuda" with no CUDA present raises ``ValueError``.

    Args:
        override: Optional device string ("cuda", "mps", "cpu", or "auto").
            ``None`` or "auto" selects the best-available device.

    Returns:
        torch.device
    """
    if override and override != "auto":
        choice = override.lower()
        if choice == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("Requested --device cuda but no CUDA device is available.")
            return torch.device("cuda")
        if choice == "mps":
            if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
                raise ValueError("Requested --device mps but MPS is not available.")
            return torch.device("mps")
        if choice == "cpu":
            return torch.device("cpu")
        raise ValueError(f"Unknown device override: {override!r}")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def clear_device_cache(device: Optional[torch.device] = None) -> None:
    """Free cached allocator memory on the active accelerator.

    Safe to call unconditionally; a no-op on CPU.
    """
    target = device or get_optimal_device()
    if target.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    elif target.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def configure_mps_settings():
    """
    Configure optimal MPS settings for audio processing.

    This should be called before initializing PyTorch models
    when running on Apple Silicon.
    """
    # Enable MPS fallback for unsupported operations
    # This allows operations not yet implemented in MPS to transparently
    # fall back to CPU kernels
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def configure_cuda_settings():
    """
    Configure optimal CUDA settings for inference.

    - TF32 tensor-core matmuls: a free speedup on Ampere+ GPUs (e.g. the
      A100 on Colab) for the band-split convs and mask-estimation MLPs,
      with negligible precision impact at inference.
    - cuDNN autotuning (``benchmark``): streaming inference feeds
      fixed-shape ``(batch, channels, chunk_size)`` chunks, so letting
      cuDNN pick the best LSTM/conv algorithms once and reuse them pays
      off across the run.

    Safe to call unconditionally; a no-op when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def _meminfo_linux() -> dict:
    """Parse ``/proc/meminfo`` for memory stats on Linux (e.g. Colab).

    ``MemAvailable`` is the kernel's own estimate of reclaimable memory and
    is the right analogue to the speculative-aware ``free_gb`` reported on
    macOS. Values in ``/proc/meminfo`` are in kibibytes.
    """
    stats: dict[str, int] = {}
    with open("/proc/meminfo", encoding="ascii") as fh:
        for line in fh:
            key, _, rest = line.partition(":")
            value = rest.strip().split()
            if value and value[0].isdigit():
                stats[key.strip()] = int(value[0]) * 1024  # kiB -> bytes

    available = stats.get("MemAvailable", stats.get("MemFree", 0))
    total = stats.get("MemTotal", 0)
    return {
        "free_gb": available / (1024**3),
        "total_gb": total / (1024**3),
        "raw_stats": stats,
    }


def get_memory_info() -> dict:
    """
    Get current memory usage information.
    On MPS, uses system memory (unified architecture).

    Returns:
        Dictionary with memory information
    """
    if sys.platform.startswith("linux"):
        try:
            return _meminfo_linux()
        except Exception as e:
            return {"error": str(e)}

    try:
        result = subprocess.run(["vm_stat"], capture_output=True, text=True)

        # Parse vm_stat output
        lines = result.stdout.strip().split("\n")
        stats: dict[str, int] = {}

        for line in lines[1:]:  # Skip header
            if ":" in line:
                parts = line.split(":")
                key = parts[0].strip()
                value = parts[1].strip().rstrip(".")
                try:
                    stats[key] = int(value)
                except ValueError:
                    # Non-integer lines (e.g. "Mach Virtual Memory Statistics:"
                    # banner) are skipped — downstream consumers want ints only.
                    continue

        # Page size is 16384 on Apple Silicon, 4096 on Intel Macs and most
        # Linux systems. Query it dynamically so we don't misreport memory.
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError):
            page_size = 16384
        free_pages = stats.get("Pages free", 0)
        speculative_pages = stats.get("Pages speculative", 0)
        active_pages = stats.get("Pages active", 0)
        inactive_pages = stats.get("Pages inactive", 0)
        wired_pages = stats.get("Pages wired down", 0)

        return {
            # Speculative pages are OS read-ahead cache that can be reclaimed
            # under pressure — match Activity Monitor by counting them free.
            "free_gb": ((free_pages + speculative_pages) * page_size) / (1024**3),
            "active_gb": (active_pages * page_size) / (1024**3),
            "inactive_gb": (inactive_pages * page_size) / (1024**3),
            "wired_gb": (wired_pages * page_size) / (1024**3),
            "page_size": page_size,
            "raw_stats": stats,
        }
    except Exception as e:
        return {"error": str(e)}


def get_system_memory_gb() -> float:
    """
    Get total system memory in GB.

    Returns:
        Total RAM in gigabytes
    """
    if sys.platform.startswith("linux"):
        try:
            return _meminfo_linux().get("total_gb", 0.0)
        except Exception:
            return 0.0

    try:
        result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        total_bytes = int(result.stdout.strip())
        return total_bytes / (1024**3)
    except Exception:
        return 0.0


def clear_mps_cache():
    """
    Clear the MPS memory cache.

    Call this periodically during long processing to prevent
    memory buildup on Apple Silicon.
    """
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_device_info() -> dict:
    """
    Get comprehensive device information.

    Returns:
        Dictionary with device details
    """
    device = get_optimal_device()

    info = {
        "device": str(device),
        "device_type": device.type,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "pytorch_version": torch.__version__,
        "total_memory_gb": get_system_memory_gb(),
    }

    # Add memory info
    mem_info = get_memory_info()
    if "error" not in mem_info:
        info["free_memory_gb"] = mem_info.get("free_gb", 0)

    return info
