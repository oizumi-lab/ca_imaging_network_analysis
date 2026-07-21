"""Paths and frozen settings for the initial investigation."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_DIR.parent
CONFIG_PATH = PROJECT_DIR / "configs" / "initial_analysis.json"
RESULTS_DIR = PROJECT_DIR / "results"
MATRIX_DIR = RESULTS_DIR / "01_matrices"
COMPARISON_DIR = RESULTS_DIR / "02_network_comparison"

with CONFIG_PATH.open(encoding="utf-8") as handle:
    CONFIG = json.load(handle)


def alpha_tag(alpha: float) -> str:
    """Filesystem-stable tag for a regularization value."""
    return f"{alpha:.3f}".replace(".", "p")

