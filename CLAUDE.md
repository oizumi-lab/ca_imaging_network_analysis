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
src/funcnet/    the package: dataio.py (loader), network.py (analysis),
                smallworld.py (path length / clustering / SWP),
                coarsegrain.py (spatial parcels + distance–module profile), paths.py
data/raw/       the 11 downloaded .mat recordings  (gitignored, ~11 GB)
scripts/        Python entry-point scripts (import from src.funcnet)
  download_data.py            fetch the dataset into data/raw/
  00_inspect_data.py          explore one recording
  01_reproduce_example.py     faithful port of the dataset's example_network_analysis.m
  10_functional_connectivity.py
  20_modularity.py
  30_state_comparison.py      the lecture result (modularity awake vs sleep/ane;
                              per-mouse scatter reproducing the talk slide)
  40_small_world.py           path length, clustering, small-world-ness / SWP
  50_coarse_grain_modularity.py  mesoscale modularity across spatial scales (paper Fig. 7 B–F)
  60_module_spatial_distribution.py  where modules sit: single-cell (intermixed) vs mesoscale
                              (localized) — paper Fig. 5A–C/G/H + Fig. 7F/G/H
verification/   library cross-checks & method controls (not part of the teaching sequence)
  50_verify_modularity.py     cross-check modularity vs NetworkX / bctpy / igraph / python-louvain
  51_verify_smallworld.py     cross-check clustering / path length / SWP vs NetworkX / bctpy
  shuffle_null_control.py     do awake-vs-unconscious Q/C/L differences survive circular-shift shuffling?
                              (raw-vs-null per state; L/Q genuine, C confounded)
  smallworld_shuffle_corrected.py  same shuffle control for the small-world measures (C, L, SWP at K=1%)
  shuffle_investigation.py    shared per-recording cache of real+shuffle Q/C/L and marginals (→results/cache/)
  why_QL_robust_C_confounded.py    OQ1: why the shuffle confounds C (~56%) but not Q (~18%) or L (~4%)
  state_difference_cause.py   OQ2: the C confound is driven by per-neuron kurtosis (burstiness), not coupling
  sparsity_clustering_mechanism.py  reframe: kurtosis is a proxy for sparsity (~1/event-count); coincidence-clique
                              mechanism (why sparsity→high C), confirmed on the real shuffle graph
  clustering_confound_mechanism.py  independent-signal proof: sparsity/coincidence (not amplitude) drives C
  burstiness_raster.py        raster/burstiness visualisation (awake vs unconscious)
references/     paper/dataset README, Figure_guide, original MATLAB example
documents/      written walkthrough + reproduction report
results/        generated outputs — results/figures/ etc. (gitignored)
logs/           run logs (downloads, long jobs)
```

## Conventions

- **Hands-on scripts are interactive `# %%` cell scripts** (VS Code / Spyder),
  **not** `.ipynb`, and have **no `main()`** — flat top-level cells. Reusable
  logic goes in the `src/funcnet/` package (normal functions).
- **Imports:** the package is **not** installed into the venv. Scripts add the
  repo root to the path (anchored to the file, so it works from any working
  directory) and import it by its explicit location, so it is obvious where the
  code lives:
  ```python
  import os, sys
  # add the repo root (parent of scripts/) so `src.funcnet` is importable
  sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
  from src.funcnet import dataio, network as net
  from src.funcnet.paths import FIG_DIR
  ```
- **Outputs go to `results/`** (e.g. `from src.funcnet.paths import FIG_DIR`).
  Don't write generated files into `scripts/`.
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
- [x] Small-world hands-on (40): path length, clustering, SWP (ports SWP/ from oizumi-lab/mouse_network)
- [x] Library cross-checks (verification/50/51): custom measures validated against NetworkX / bctpy / igraph / python-louvain
- [x] Mesoscale coarse-graining (script 50): paper Fig. 7 B–F (modularity vs spatial scale)
- [x] Spatial distribution of modules (script 60): Fig. 5 A–C/G/H + Fig. 7 F/G/H (intermixed vs localized)
- [x] Shuffle-null confound investigation (verification/): L/Q genuine, C/SWP confounded; C confound driven
      by per-neuron kurtosis, not coupling (OQ1/OQ2 resolved + adversarially verified; documents/04)
- [ ] Future analyses (per-neuron Qi, module stability, distance–activity-correlation Fig. 7 I–L)
