# CSHA neural data science — functional-network modularity hands-on

Teaching materials for the **CSHA 2026 Neural Data Science for Large-scale
Recordings** course. The tutorial uses wide-field two-photon calcium imaging to
ask how functional-network modularity changes between wakefulness and NREM
sleep, and how the result depends on spatial scale.

The analysis follows:

> Kiyooka & Oomoto et al. (2026). *Single-cell resolution functional networks
> during unconsciousness are segregated into spatially intermixed modules.*
> Cell Reports. https://doi.org/10.1016/j.celrep.2025.116902

- Dataset version 3: https://neurodata.riken.jp/id/20260708-001
- Reference MATLAB analysis: https://github.com/oizumi-lab/mouse_network_2P

## Two-track workflow

### Track A — recommended hands-on: one complete recording

The default downloads the complete `mouse02_sleep` calcium recording (~1.09 GB)
and its frame-trigger-synchronized
EEG/EMG file (~0.36 GB). Run scripts 00--07 in order.

| Script | Outcome |
|---|---|
| `00_download_data.py` | Download `mouse02_sleep` and its synchronized EEG/EMG recording |
| `01_inspect_data.py` | Plot 100 raw traces, atlas-colored spatial coverage, all-neuron raster, EEG spectrogram, EMG, and state labels |
| `02_functional_connectivity.py` | Convert matched activity to equal-density graphs and report each state's correlation threshold |
| `03_modularity.py` | Apply repeated Louvain optimization and visualize functional modules |
| `04_sample_state_comparison.py` | Compare Awake and NREM modularity across windows and graph densities |
| `05_sample_coarse_grain_modularity.py` | Rebuild modular networks after spatial coarse-graining |
| `06_sample_module_spatial_distribution.py` | Compare intermixed single-cell modules with localized parcel modules |
| `07_multiscale_module_movie.py` | Animate module geography from single cells to 40-neuron parcels |

### Track B — optional research extension: all mice

Download every calcium and EEG/EMG recording, then run scripts 08--10:

```bash
poetry run python scripts/00_download_data.py --all
```

| Script | Paper-scale result |
|---|---|
| `08_all_mice_modularity.py` | Awake versus NREM/anesthesia modularity across all recordings and biological mice |
| `09_all_mice_coarse_grain_modularity.py` | State-dependent modularity across spatial scales (paper Fig. 7B--F) |
| `10_all_mice_module_spatial_distribution.py` | Module maps and same-module probability versus distance (paper Fig. 5/7) |

The all-mice scripts default to teaching-sized settings so they can be explored
interactively. Set `PAPER_MODE = True` in scripts 08--10 for all selected
neurons, 200 Louvain runs, and the expanded paper-style analysis. Those runs are
computationally intensive.

## Setup

Requires Python 3.12 and Poetry.

```bash
poetry install
poetry run python scripts/00_download_data.py
```

Open scripts 01--10 in VS Code with the Python extension or in Spyder and run
the `# %%` cells sequentially. Generated figures, tables, and movies are written
under `results/` and are not committed.

## Core analysis

```text
smoothed deconvolved activity
        ↓
Pearson correlation between every neuron pair
        ↓
retain the strongest |r| values at a fixed graph density K
        ↓
repeat Louvain community detection
        ↓
max-Q partition and modularity Q
```

Fixed density keeps the compared graphs at the same edge count. Louvain is
stochastic, so the scripts repeat it and retain the highest-Q partition, matching
the published workflow. Sleep windows contain 1,500 frames; anesthesia windows
contain 2,900 frames.

## Reusable package

General functions live in `src/funcnet/`:

| Module | Responsibility |
|---|---|
| `dataio.py` | Load version-3 MATLAB/HDF5 recordings, states, and active-neuron rows |
| `physiology.py` | Load and frame-trigger-align version-3 EEG/EMG; prepare display and scoring features |
| `timeseries.py` | Stable-state windows, acquisition segments, and temporal helpers |
| `network.py` | Correlation, density thresholding, Louvain modularity, and consensus partitions |
| `coarsegrain.py` | Spatial parcels, signal averaging, and module-distance summaries |
| `visualization.py` | State, activity, physiology, and module-map plotting helpers |
| `statistics.py` | Mouse-level confidence summaries |
| `paths.py` | Portable input/output paths |

## Tests

The regression tests use synthetic, data-free fixtures:

```bash
poetry run python -m unittest discover -s tests -v
poetry run ruff check src/funcnet scripts tests
```

## Course slides and archive

`documents/CSHA_082426.pptx` is an untracked 91-MB source talk used only as a
content/design reference. The hands-on deck will be regenerated after the
numbered scripts are finalized.

The broader pre-course repository—including DMD, graphical lasso, Rastermap,
small-world, and verification studies—is preserved in the Git branch
`archive/full-analysis-2026` and tag `full-analysis-2026-08-18`.

## License and attribution

The datasets are CC BY 4.0. Cite the dataset release and the Cell Reports paper
when reusing the materials or derived results.
