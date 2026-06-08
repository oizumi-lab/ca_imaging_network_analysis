# CLAUDE.md — ca_imaging_network_analysis

## What this project is

Hands-on teaching materials for a **neural-data-analysis course in China**.
Lecture topic: **state-dependent functional networks** — comparing the functional
network of the cortex across brain states (**awake, sleep, anesthesia**).

The first deliverable is a **modularity** hands-on that reproduces, then teaches,
the analysis in:

> Kiyooka & Oomoto et al. (2026). *Single-cell resolution functional networks
> during unconsciousness are segregated into spatially intermixed modules.*
> **Cell Reports.** https://doi.org/10.1016/j.celrep.2025.116902

- Reference MATLAB code: https://github.com/oizumi-lab/mouse_network_2P
- Dataset (RIKEN 20260409-001, **v2.0**, CC-BY 4.0): https://neurodata.riken.jp/id/20260409-001
- Mirror: Zenodo https://doi.org/10.5281/zenodo.17667863

## Repository layout

```
.claude/        project rules & shared settings
src/funcnet/    installable package: dataio.py (loader), network.py (analysis),
                paths.py (project paths). Editable-installed by `poetry install`.
data/raw/       the 11 downloaded .mat recordings  (gitignored, ~11 GB)
scripts/        Python entry-point scripts (import from funcnet)
  download_data.py            fetch the dataset into data/raw/
  00_inspect_data.py          explore one recording
  01_reproduce_example.py     faithful port of the dataset's example_network_analysis.m
  10_functional_connectivity.py
  20_modularity.py
  30_state_comparison.py      the lecture result (modularity awake vs sleep/ane)
references/     paper/dataset README, Figure_guide, original MATLAB example
documents/      written walkthrough + reproduction report
results/        generated outputs — results/figures/ etc. (gitignored)
logs/           run logs (downloads, long jobs)
```

## Conventions

- **Hands-on scripts are interactive `# %%` cell scripts** (VS Code / Spyder),
  **not** `.ipynb`, and have **no `main()`** — flat top-level cells. Reusable
  logic goes in the `src/funcnet/` package (normal functions).
- **Imports:** `funcnet` is an editable-installed package (src layout), so scripts
  just do `from funcnet import dataio, network as net` — no `sys.path` hacks, works
  from any directory and any machine. Re-run `poetry install` after pulling.
- **Outputs go to `results/`** (e.g. `from funcnet.paths import FIG_DIR`). Don't
  write generated files into `scripts/`.
- English for all materials.
- `data/`, `.venv/`, and `results/` are gitignored. Don't commit data or the venv.

## Environment

- **pyenv** local: Python **3.12.13** (`.python-version`).
- **Poetry** with in-project venv → `./.venv` (`poetry.toml`).

```bash
poetry install                         # create .venv, install deps
poetry run python scripts/download_data.py            # get data (~11 GB)
poetry run python scripts/download_data.py --example  # just the 84 MB sample
poetry run python scripts/01_reproduce_example.py     # validate the port
```

Key deps: numpy, scipy, pandas, matplotlib, **h5py**, **pymatreader** (reads the
v7.3/HDF5 `.mat` files), **bctpy** (`community_louvain` etc.), networkx, tqdm,
requests.

## Critical data-format note (v1 → v2)

The MATLAB repo targets dataset **v1.0**; we use **v2.0**, which renamed
variables AND stores `.mat` files as **MATLAB v7.3 (HDF5)**. Consequences:

- `scipy.io.loadmat` **cannot** read these files — use `pymatreader` (already
  wired into `src/funcnet/dataio.py`).
- Variable renames and 1-based→0-based frame indices are handled by the loader.
  Full mapping: see `.claude/rules/dataset-v2-format.md`.
- `ROIs.atlas` (region labels) is a MATLAB *string-class* object that neither
  pymatreader nor h5py decode; it is exposed as `None`. Not needed for the
  single-cell modularity hands-on.

## Status

- [x] Project scaffold, env, data download
- [x] v2.0 loader + network/modularity library
- [x] Faithful reproduction of `example_network_analysis.m` (validated)
- [x] Modularity hands-on scripts (00/01/10/20/30)
- [ ] Future analyses (coarse-graining / mesoscale, per-neuron Qi, module stability)
