"""Paths resolved relative to this package, with no machine-specific strings."""

from __future__ import annotations

from pathlib import Path

DMD_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = DMD_ROOT.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
CONFIG_PATH = DMD_ROOT / "configs" / "initial_validation.json"
RUN_DIR = DMD_ROOT / "results" / "03_initial_validation"
FIGURE_DIR = RUN_DIR / "figures"
TABLE_DIR = RUN_DIR / "tables"
ARTIFACT_DIR = RUN_DIR / "artifacts"


def ensure_run_directories() -> None:
    """Create the generated-output tree used by every pipeline stage."""
    for path in (RUN_DIR, FIGURE_DIR, TABLE_DIR, ARTIFACT_DIR):
        path.mkdir(parents=True, exist_ok=True)
