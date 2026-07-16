"""Private DMD initial-validation implementation.

The package intentionally lives under ``DMD/`` so it can be separated from the
summer-school repository without moving tutorial code.
"""

from .config import load_config
from .paths import DMD_ROOT, REPO_ROOT, RUN_DIR

__all__ = ["DMD_ROOT", "REPO_ROOT", "RUN_DIR", "load_config"]
