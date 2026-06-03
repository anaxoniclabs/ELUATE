# SPDX-License-Identifier: MIT
"""
Eluate - Remove background music from videos.

Public API: ``elute``, ``Session``, ``Result``, ``EluateError``, ``main``,
``__version__``. Everything else under ``eluate.*`` is internal and may
change in any release.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("eluate")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+dev"


from .api import (
    DurationOutOfRange,
    EluateError,
    InsufficientDiskSpace,
    ModelNotInstalledError,
    Result,
    Session,
    elute,
)
from .cli import main

__all__ = [
    "DurationOutOfRange",
    "EluateError",
    "InsufficientDiskSpace",
    "ModelNotInstalledError",
    "Result",
    "Session",
    "__version__",
    "elute",
    "main",
]
