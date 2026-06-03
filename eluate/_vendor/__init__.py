# SPDX-License-Identifier: MIT
"""
Vendored third-party code shipped inside the eluate wheel.

Subpackages here are frozen copies of code eluate needs at runtime.
The originals live as git submodules under ``vendor/`` at the repo
root for upstream-tracking purposes; that directory is excluded from
the wheel. Only ``eluate._vendor.*`` is shipped to PyPI users.

Do not import ``eluate._vendor`` from anything other than eluate's own
internals. It is private and may change without notice.
"""
