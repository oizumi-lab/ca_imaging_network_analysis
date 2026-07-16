"""Configuration loading and validation for the frozen pilot protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .paths import CONFIG_PATH


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the JSON protocol and attach a reproducibility checksum."""
    path = Path(path)
    raw = path.read_bytes()
    config = json.loads(raw)
    _validate(config)
    config["_config_path"] = str(path.resolve())
    config["_config_sha256"] = hashlib.sha256(raw).hexdigest()
    return config


def _validate(config: dict[str, Any]) -> None:
    required = {"recordings", "windows", "arms", "model", "resampling", "simulation", "gates"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Configuration is missing sections: {missing}")
    windows = config["windows"]
    positions = sorted(windows["development_positions"] + windows["evaluation_positions"])
    if positions != list(range(windows["per_label"])):
        raise ValueError("Development/evaluation positions must partition selected windows")
    if set(windows["development_positions"]).intersection(windows["evaluation_positions"]):
        raise ValueError("Development and evaluation windows overlap")
    if config["model"]["tracking_subspace_dimension"] >= config["model"]["tracking_minimum_rank"]:
        raise ValueError("Tracking subspace dimension must be strictly smaller than eligible model rank")
