# CLAUDE.md — ca_imaging_network_analysis

## Purpose

Hands-on materials for the CSHA 2026 Neural Data Science course. The teaching
topic is state-dependent functional-network modularity in wide-field two-photon
calcium imaging, based on Kiyooka & Oomoto et al. (Cell Reports, 2026).

## Teaching workflow

- Scripts 00--06 use the complete `mouse02_sleep` calcium recording and its
  synchronized EEG/EMG recording.
- Scripts 07--09 require all calcium recordings and reproduce all-mice
  modularity and spatial-scale comparisons.

Run the scripts in numeric order. Scripts 01--09 are interactive `# %%` files
for VS Code or Spyder. Reusable logic belongs under `src/funcnet/`; settings,
narrative cells, and figure composition remain in `scripts/`.

## Repository layout

```text
scripts/          00--09 course and research-extension workflow
  supplemental/   optional unnumbered analysis scripts
src/funcnet/      data I/O, physiology, time windows, networks, coarse-graining,
                  statistics, plotting, and project paths
tests/            data-free regression tests
documents/        written hands-on guide and slide-deck builder
references/       deposited dataset documentation and source links
data/raw/         downloaded recordings (gitignored)
results/          generated figures, CSVs, and movies (gitignored)
```

The earlier broad research repository is preserved in branch
`archive/full-analysis-2026` and tag `full-analysis-2026-08-18`.

## Data

Default course download:

```bash
poetry run python scripts/00_download_data.py
```

All data for scripts 07--09:

```bash
poetry run python scripts/00_download_data.py --all
```

The course uses dataset version 3. MATLAB v7.3 files must be read with
`pymatreader`/`h5py`, not `scipy.io.loadmat`. The loader also converts MATLAB
indices for direct NumPy use.

## Core conventions

- English for all course materials.
- Outputs go under `results/`; scripts never write generated files beside code.
- Compare states using identical neuron rows and equal graph density.
- Louvain analyses repeat stochastic optimization and report the max-Q partition.
- Windows are nested within recordings; biological inference is summarized by mouse.
- Keep course defaults tractable and expose `PAPER_MODE` for long full-neuron runs.

## Environment and checks

Python 3.12 with Poetry and an in-project `.venv`.

```bash
poetry install
poetry run python -m unittest discover -s tests -v
poetry run ruff check src/funcnet scripts tests
```
