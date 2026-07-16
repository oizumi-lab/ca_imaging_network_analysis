"""Immutable run directories and chronological stage-status records."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .io import write_json
from .paths import RUN_DIR, ensure_run_directories


STAGE_DIRECTORIES = {
    "data": "00_data",
    "preprocessing": "01_preprocessing",
    "precedent": "02_precedent",
    "simulation": "03_simulation",
    "forecast": "04_forecast",
    "stability": "05_stability",
    "nulls": "06_nulls",
    "decision": "07_decision",
}


@dataclass(frozen=True)
class RunContext:
    run_id: str
    root: Path

    def stage_dir(self, stage: str) -> Path:
        if stage not in STAGE_DIRECTORIES:
            raise KeyError(f"Unknown stage {stage!r}")
        path = self.root / STAGE_DIRECTORIES[stage]
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def status_path(self) -> Path:
        return self.root / "stage_status.json"


def _git_short_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def create_run(config: dict[str, Any]) -> RunContext:
    """Create a new immutable run and mark it as the active pilot run."""
    ensure_run_directories()
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    run_id = f"{timestamp}_{_git_short_revision()}_{config['_config_sha256'][:10]}"
    root = RUN_DIR / "runs" / run_id
    root.mkdir(parents=True, exist_ok=False)
    context = RunContext(run_id=run_id, root=root)
    write_json(
        context.status_path,
        {"run_id": run_id, "created_at": datetime.now().astimezone().isoformat(), "stages": {}},
    )
    write_json(RUN_DIR / "active_run.json", {"run_id": run_id, "root": str(root.resolve())})
    return context


def active_run() -> RunContext:
    path = RUN_DIR / "active_run.json"
    if not path.exists():
        raise FileNotFoundError("No active DMD pilot run; execute the data stage first")
    values = json.loads(path.read_text())
    context = RunContext(run_id=str(values["run_id"]), root=Path(values["root"]))
    if not context.root.exists():
        raise FileNotFoundError(f"Active run directory is missing: {context.root}")
    return context


def stage_state(context: RunContext, stage: str) -> str | None:
    values = json.loads(context.status_path.read_text())
    record = values.get("stages", {}).get(stage)
    return None if record is None else str(record.get("state"))


def start_stage(context: RunContext, stage: str) -> None:
    values = json.loads(context.status_path.read_text())
    previous = values.setdefault("stages", {}).get(stage)
    if previous and previous.get("state") == "completed":
        raise RuntimeError(f"Stage {stage!r} is already complete in immutable run {context.run_id}")
    values["stages"][stage] = {
        "state": "running",
        "started_at": datetime.now().astimezone().isoformat(),
    }
    write_json(context.status_path, values)


def complete_stage(context: RunContext, stage: str, summary: dict[str, Any]) -> None:
    values = json.loads(context.status_path.read_text())
    record = values.setdefault("stages", {}).setdefault(stage, {})
    record.update(
        {
            "state": "completed",
            "completed_at": datetime.now().astimezone().isoformat(),
            "summary": summary,
        }
    )
    write_json(context.status_path, values)


def fail_stage(context: RunContext, stage: str, message: str) -> None:
    values = json.loads(context.status_path.read_text())
    record = values.setdefault("stages", {}).setdefault(stage, {})
    record.update(
        {
            "state": "failed",
            "failed_at": datetime.now().astimezone().isoformat(),
            "message": message,
        }
    )
    write_json(context.status_path, values)
