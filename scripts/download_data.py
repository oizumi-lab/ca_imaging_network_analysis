"""Download the RIKEN v2.0 calcium-imaging dataset into ``data/raw/``.

Dataset: RIKEN neurodata 20260409-001, v2.0 (CC-BY 4.0)
  https://neurodata.riken.jp/id/20260409-001
Mirror:  Zenodo  https://doi.org/10.5281/zenodo.17667863

Usage
-----
    poetry run python scripts/download_data.py            # everything (~11 GB)
    poetry run python scripts/download_data.py --example  # just example_data.mat
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
from src.funcnet.paths import RAW_DIR, REFERENCES_DIR as REF_DIR

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


def download(file_id: str, dest: Path, expected_size: int | None = None) -> None:
    """Stream one file to ``dest`` with a progress bar; skip if already complete."""
    if dest.exists() and expected_size is not None and dest.stat().st_size == expected_size:
        print(f"  skip  {dest.name} (already complete)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = API.format(id=file_id)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or expected_size or 0
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"  {dest.name}", leave=True
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--example", action="store_true", help="only example_data.mat")
parser.add_argument("--refs", action="store_true", help="only reference docs")
args = parser.parse_args()

if args.refs:
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
