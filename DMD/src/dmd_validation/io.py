"""Small helpers for reproducible, atomic tabular and JSON artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dmd-matplotlib")

import matplotlib
import numpy as np
import pandas as pd
import scipy

from .paths import DMD_ROOT


def write_json(path: str | Path, value: Any) -> None:
    """Write JSON through a sibling temporary file, then replace atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n")
    os.replace(temporary, path)


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.10g")
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def software_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """Capture versions plus content hashes, including untracked DMD source."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    implementation_files = sorted(
        path
        for directory in ("src", "scripts", "tests", "configs")
        for path in (DMD_ROOT / directory).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    file_hashes = {
        str(path.relative_to(DMD_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in implementation_files
    }
    tree_digest = hashlib.sha256()
    for relative_path, digest in file_hashes.items():
        tree_digest.update(relative_path.encode())
        tree_digest.update(b"\0")
        tree_digest.update(digest.encode())
        tree_digest.update(b"\n")
    try:
        git_status = subprocess.run(
            ["git", "status", "--short", "--", "DMD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_status = ["unavailable"]
    return {
        "config_path": config["_config_path"],
        "config_sha256": config["_config_sha256"],
        "git_revision": revision,
        "git_status_dmd": git_status,
        "implementation_tree_sha256": tree_digest.hexdigest(),
        "implementation_file_sha256": file_hashes,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }
