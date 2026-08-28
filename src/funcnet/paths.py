"""Canonical project paths.

Resolved relative to this installed package, so they work on any machine and
from any working directory (no hardcoded absolute paths). With the editable
install (``poetry install``), ``__file__`` points at ``<root>/src/funcnet/paths.py``,
so the project root is three levels up.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PHYSIOLOGY_DIR = RAW_DIR / "eeg_emg"
REFERENCES_DIR = PROJECT_ROOT / "references"

# All notebook outputs (figures, csv, ...) accumulate here.
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"


def ensure_results() -> Path:
    """Create the results/figures directory if needed and return FIG_DIR."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR
