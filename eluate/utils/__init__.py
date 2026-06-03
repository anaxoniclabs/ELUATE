# SPDX-License-Identifier: MIT
"""
Eluate utility modules.

Provides device detection, path management, and validation helpers.
Import submodules directly (e.g. ``from eluate.utils.paths import
get_app_dir``) — this package does not re-export anything. Keeping the
package free of eager imports is what lets ``eluate --version`` start
without paying torch's import cost.
"""
