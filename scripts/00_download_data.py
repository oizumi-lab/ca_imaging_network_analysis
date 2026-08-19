"""Download the data used by the CSHA modularity hands-on.

The default course download is one complete version-3 sleep recording and its
synchronized EEG/EMG file:

    poetry run python scripts/00_download_data.py

This downloads ``mouse02_sleep`` (about 1.09 GB) plus its physiology file (about
0.36 GB). It is the recording used by scripts 01--07.

For the full-cohort paper analyses in scripts 08--10, download every calcium and
physiology recording:

    poetry run python scripts/00_download_data.py --all

Dataset version 3: RIKEN neurodata 20260708-001 (CC-BY 4.0)
  https://neurodata.riken.jp/id/20260708-001

Files are streamed from the public RIKEN API. Already-complete files are
skipped after checking their expected byte size, so the script is safe to rerun.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# Add the repository root so ``src.funcnet`` is importable from any cwd.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.funcnet.paths import PHYSIOLOGY_DIR, RAW_DIR

API = "https://neurodata.riken.jp/api/v3/files/{id}/download/"
SAMPLE_RECORDING = "mouse02_sleep.mat"
SAMPLE_PHYSIOLOGY = "mouse02_sleep_physiological_data.mat"

CALCIUM_FILES: dict[str, tuple[str, int]] = {
    "mouse01_sleep.mat": ("1f786027-c4b2-4ac9-8449-24c0cfb434db", 658_474_258),
    "mouse02_sleep.mat": ("b266f5ee-cc1d-4a47-9d41-29655a7c3967", 1_093_789_677),
    "mouse03_sleep.mat": ("3191b169-1d10-4807-9284-edfd96f6c44e", 1_197_235_306),
    "mouse03_ane.mat": ("1616a6b6-edec-4637-bea9-bca9c0a95848", 1_123_058_079),
    "mouse04_day1_sleep.mat": ("8d84d7ac-b4c8-4208-9e5c-c3a4605d3e31", 1_625_754_526),
    "mouse04_day2_sleep.mat": ("105e2c10-8a9a-4cd2-9c76-411cf673ec28", 2_254_368_168),
    "mouse05_sleep.mat": ("ed06e6f2-7485-493e-8d32-d01e0d13aec9", 1_263_665_343),
    "mouse05_ane.mat": ("5b5f67db-2bd0-43ae-8b4f-eb50138e6057", 1_079_340_853),
    "mouse06_ane.mat": ("e1dd0560-3486-45e6-a086-e2c1bfb2bb6e", 563_929_318),
    "mouse07_ane.mat": ("ffdc68e6-fbb4-43f5-81e2-a8271a50819a", 363_189_208),
}

PHYSIOLOGY_FILES: dict[str, tuple[str, int]] = {
    "README_EEG_EMG.md": ("62efc349-d2ac-4aec-89d5-d747af61a20c", 2_969),
    "mouse01_sleep_physiological_data.mat": ("ba79f124-a00e-4f71-ae60-729ee9b8dbcc", 180_001_112),
    "mouse02_sleep_physiological_data.mat": ("b60cfc5d-c6f5-499f-bb8b-e93564e07360", 360_001_112),
    "mouse03_sleep_physiological_data.mat": ("1b58e659-c2d9-406d-94dd-41d25c856f78", 360_001_112),
    "mouse04_day1_sleep_physiological_data.mat": ("6cf82a55-0e08-4a07-8716-9aed78b50430", 360_001_112),
    "mouse04_day2_sleep_physiological_data.mat": ("25761808-4bbd-471f-b249-4f48d0cb33d4", 432_001_112),
    "mouse05_sleep_physiological_data.mat": ("3c6a3b7b-3dfb-4631-a670-5edbf8ebf81c", 360_001_112),
    "mouse03_ane_physiological_data.mat": ("0bf7785e-612a-41cd-9f52-33261659be30", 362_601_312),
    "mouse05_ane_physiological_data.mat": ("48f04ef2-dcd8-48fc-8d16-26f079cfb314", 546_001_312),
    "mouse06_ane_physiological_data_awake.mat": ("e80affb3-91a2-4a7c-bafc-5cfe352b3fca", 179_201_296),
    "mouse06_ane_physiological_data_ane.mat": ("b5548e9c-76a4-4b7a-8a11-b58fc48a332f", 299_601_296),
    "mouse07_ane_physiological_data.mat": ("4a57ad37-a7e9-4948-acc9-708926eb1b4f", 619_201_496),
}

def download(file_id: str, destination: Path, expected_size: int | None = None) -> None:
    """Stream one file and replace the destination only after validation."""
    if destination.exists() and (
        (expected_size is None and destination.stat().st_size > 0)
        or destination.stat().st_size == expected_size
    ):
        print(f"  skip  {destination.name} (already complete)")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    with requests.get(API.format(id=file_id), stream=True, timeout=60) as response:
        response.raise_for_status()
        declared_size = int(response.headers.get("content-length", 0))
        total = expected_size or declared_size
        received = 0
        with partial.open("wb") as stream, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"  {destination.name}",
            leave=True,
        ) as progress:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                stream.write(chunk)
                received += len(chunk)
                progress.update(len(chunk))

    required_size = expected_size or declared_size
    if required_size and received != required_size:
        raise OSError(
            f"Incomplete download for {destination.name}: received {received:,} "
            f"bytes, expected {required_size:,}."
        )
    partial.replace(destination)


def download_group(
    files: dict[str, tuple[str, int]], destination: Path, label: str
) -> None:
    total_gb = sum(size for _, size in files.values()) / 1e9
    print(f"{label} -> {destination}  (~{total_gb:.2f} GB)")
    for name, (file_id, size) in files.items():
        download(file_id, destination / name, expected_size=size)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--all",
    action="store_true",
    help="download all calcium and EEG/EMG recordings for scripts 08--10",
)
args = parser.parse_args()

calcium = CALCIUM_FILES if args.all else {SAMPLE_RECORDING: CALCIUM_FILES[SAMPLE_RECORDING]}
physiology = (
    PHYSIOLOGY_FILES
    if args.all
    else {SAMPLE_PHYSIOLOGY: PHYSIOLOGY_FILES[SAMPLE_PHYSIOLOGY]}
)
download_group(calcium, RAW_DIR, "Processed calcium recordings")
download_group(physiology, PHYSIOLOGY_DIR, "Synchronized EEG/EMG recordings")

print("\nDone.")
