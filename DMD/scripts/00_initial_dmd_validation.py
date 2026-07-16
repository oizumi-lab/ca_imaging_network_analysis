#!/usr/bin/env python3
"""Run one reviewable stage of the private initial DMD validation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dmd-matplotlib")

DMD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DMD_ROOT / "src"))

from dmd_validation.pipeline import run_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("data", "preprocessing", "precedent", "simulation", "forecast", "stability", "nulls", "decision"),
        help="Run exactly one stage and save all intermediate evidence.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DMD_ROOT / "configs" / "initial_validation.json",
        help="Frozen JSON protocol.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = run_stage(args.stage, args.config)
    print(f"completed stage={args.stage} run_id={context.run_id}")
    print(f"artifacts={context.stage_dir(args.stage)}")


if __name__ == "__main__":
    main()
