"""Download the RIKEN calcium-imaging and physiology datasets.

Processed calcium data: RIKEN neurodata 20260409-001, v2.0 (CC-BY 4.0)
  https://neurodata.riken.jp/id/20260409-001
Mirror:  Zenodo  https://doi.org/10.5281/zenodo.17667863

EEG/EMG extension: RIKEN neurodata 20260708-001, v3.0 (CC-BY 4.0)
  https://neurodata.riken.jp/id/20260708-001

Usage
-----
    poetry run python scripts/download_data.py            # processed calcium (~11 GB)
    poetry run python scripts/download_data.py --example  # just example_data.mat
    poetry run python scripts/download_data.py --eeg-emg  # physiology only (~4.1 GB)
    poetry run python scripts/download_data.py --refs     # only the reference docs

Files are streamed from the public RIKEN API. Already-complete files (matching
the expected byte size) are skipped, so the script is safe to re-run.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# add the repo root (parent of scripts/) to the path so `src.funcnet` is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.funcnet.paths import PHYSIOLOGY_DIR, RAW_DIR, REFERENCES_DIR as REF_DIR

API = "https://neurodata.riken.jp/api/v3/files/{id}/download/"

# name -> (file-id, expected size in bytes). example_data first (small, for dev).
DATA_FILES: dict[str, tuple[str, int]] = {
    "example_data.mat":       ("7fe2101f-4375-438f-b3c0-c4405256d7de", 83_735_492),
    "mouse01_sleep.mat":      ("1f786027-c4b2-4ac9-8449-24c0cfb434db", 658_474_258),
    "mouse02_sleep.mat":      ("b266f5ee-cc1d-4a47-9d41-29655a7c3967", 1_093_789_677),
    "mouse03_sleep.mat":      ("3191b169-1d10-4807-9284-edfd96f6c44e", 1_197_235_306),
    "mouse03_ane.mat":        ("1616a6b6-edec-4637-bea9-bca9c0a95848", 1_123_058_079),
    "mouse04_day1_sleep.mat": ("8d84d7ac-b4c8-4208-9e5c-c3a4605d3e31", 1_625_754_526),
    "mouse04_day2_sleep.mat": ("105e2c10-8a9a-4cd2-9c76-411cf673ec28", 2_254_368_168),
    "mouse05_sleep.mat":      ("ed06e6f2-7485-493e-8d32-d01e0d13aec9", 1_263_665_343),
    "mouse05_ane.mat":        ("5b5f67db-2bd0-43ae-8b4f-eb50138e6057", 1_079_340_853),
    "mouse06_ane.mat":        ("e1dd0560-3486-45e6-a086-e2c1bfb2bb6e", 563_929_318),
    "mouse07_ane.mat":        ("ffdc68e6-fbb4-43f5-81e2-a8271a50819a", 363_189_208),
}

# Reference docs / original MATLAB example (small) -> references/
REF_FILES: dict[str, tuple[str, str]] = {
    "dataset_Readme.md":          ("f8939569-4991-41c2-9c2e-4c17b6de8274", "README"),
    "dataset_Figure_guide.md":    ("c746bf67-145b-4f90-ad6e-6335a9e60361", "figure guide"),
    "example_network_analysis.m": ("6a72f9a9-f41b-407a-ba7d-14138caef44e", "original example"),
}

# v3.0 EEG/EMG files -> data/raw/eeg_emg/.  Keeping classic MATLAB files in a
# subdirectory prevents dataio.list_recordings() from treating them as calcium
# recordings.  mouse06 has separate chronological awake/anesthesia files.
PHYSIOLOGY_FILES: dict[str, tuple[str, int]] = {
    "README_EEG_EMG.md": (
        "62efc349-d2ac-4aec-89d5-d747af61a20c",
        2_969,
    ),
    "mouse01_sleep_physiological_data.mat": (
        "ba79f124-a00e-4f71-ae60-729ee9b8dbcc",
        180_001_112,
    ),
    "mouse02_sleep_physiological_data.mat": (
        "b60cfc5d-c6f5-499f-bb8b-e93564e07360",
        360_001_112,
    ),
    "mouse03_sleep_physiological_data.mat": (
        "1b58e659-c2d9-406d-94dd-41d25c856f78",
        360_001_112,
    ),
    "mouse04_day1_sleep_physiological_data.mat": (
        "6cf82a55-0e08-4a07-8716-9aed78b50430",
        360_001_112,
    ),
    "mouse04_day2_sleep_physiological_data.mat": (
        "25761808-4bbd-471f-b249-4f48d0cb33d4",
        432_001_112,
    ),
    "mouse05_sleep_physiological_data.mat": (
        "3c6a3b7b-3dfb-4631-a670-5edbf8ebf81c",
        360_001_112,
    ),
    "mouse03_ane_physiological_data.mat": (
        "0bf7785e-612a-41cd-9f52-33261659be30",
        362_601_312,
    ),
    "mouse05_ane_physiological_data.mat": (
        "48f04ef2-dcd8-48fc-8d16-26f079cfb314",
        546_001_312,
    ),
    "mouse06_ane_physiological_data_awake.mat": (
        "e80affb3-91a2-4a7c-bafc-5cfe352b3fca",
        179_201_296,
    ),
    "mouse06_ane_physiological_data_ane.mat": (
        "b5548e9c-76a4-4b7a-8a11-b58fc48a332f",
        299_601_296,
    ),
    "mouse07_ane_physiological_data.mat": (
        "4a57ad37-a7e9-4948-acc9-708926eb1b4f",
        619_201_496,
    ),
}


def download(file_id: str, dest: Path, expected_size: int | None = None) -> None:
    """Stream and size-check one file, replacing ``dest`` only when complete."""
    if dest.exists() and expected_size is not None and dest.stat().st_size == expected_size:
        print(f"  skip  {dest.name} (already complete)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(f"{dest.name}.part")
    url = API.format(id=file_id)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        declared_size = int(r.headers.get("content-length", 0))
        total = expected_size or declared_size
        received = 0
        with partial.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"  {dest.name}", leave=True
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                received += len(chunk)
                bar.update(len(chunk))
    required_size = expected_size or declared_size
    if required_size and received != required_size:
        raise OSError(
            f"Incomplete download for {dest.name}: received {received:,} bytes, "
            f"expected {required_size:,}. The final destination was not replaced."
        )
    partial.replace(dest)


parser = argparse.ArgumentParser(description=__doc__)
selection = parser.add_mutually_exclusive_group()
selection.add_argument(
    "--example",
    action="store_true",
    help="only example_data.mat",
)
selection.add_argument(
    "--eeg-emg",
    action="store_true",
    help="only the v3.0 EEG/EMG files",
)
selection.add_argument("--refs", action="store_true", help="only reference docs")
args = parser.parse_args()

if args.eeg_emg:
    total_gb = sum(size for _, size in PHYSIOLOGY_FILES.values()) / 1e9
    print(f"EEG/EMG files -> {PHYSIOLOGY_DIR}  (~{total_gb:.1f} GB)")
    for name, (file_id, size) in PHYSIOLOGY_FILES.items():
        download(file_id, PHYSIOLOGY_DIR / name, expected_size=size)
elif args.refs:
    print(f"Reference docs -> {REF_DIR}")
    for name, (fid, _desc) in REF_FILES.items():
        download(fid, REF_DIR / name)
else:
    print(f"Reference docs -> {REF_DIR}")
    for name, (fid, _desc) in REF_FILES.items():
        download(fid, REF_DIR / name)

    items = (
        {"example_data.mat": DATA_FILES["example_data.mat"]}
        if args.example
        else DATA_FILES
    )
    total_gb = sum(sz for _, sz in items.values()) / 1e9
    print(f"\nData files -> {RAW_DIR}  (~{total_gb:.1f} GB)")
    for name, (fid, size) in items.items():
        download(fid, RAW_DIR / name, expected_size=size)

print("\nDone.")
