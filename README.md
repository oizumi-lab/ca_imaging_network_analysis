# ca_imaging_network_analysis

Hands-on materials for **state-dependent functional network analysis** of
large-scale two-photon calcium imaging in mouse cortex, comparing functional
networks across **wakefulness, sleep, and anesthesia**.

Built for a neural-data-analysis course. The first module reproduces and teaches
the **modularity** analysis from:

> Kiyooka & Oomoto et al. (2026). *Single-cell resolution functional networks
> during unconsciousness are segregated into spatially intermixed modules.*
> **Cell Reports.** https://doi.org/10.1016/j.celrep.2025.116902

- **Processed calcium dataset** (RIKEN 20260409-001, v2.0, CC-BY 4.0): https://neurodata.riken.jp/id/20260409-001
- **EEG/EMG extension** (RIKEN 20260708-001, v3.0, CC-BY 4.0): https://neurodata.riken.jp/id/20260708-001
- **Reference MATLAB code**: https://github.com/oizumi-lab/mouse_network_2P
- **Rastermap method**: Stringer et al. (2025), Nature Neuroscience,
  https://doi.org/10.1038/s41593-024-01783-4
- **Official Rastermap code**: https://github.com/MouseLand/rastermap

## Quick start

Requires [pyenv](https://github.com/pyenv/pyenv) (Python 3.12.13) and
[Poetry](https://python-poetry.org/).

```bash
poetry install                                     # build ./.venv, install deps
poetry run python scripts/download_data.py --example   # 84 MB sample (fast)
# or:  poetry run python scripts/download_data.py       # full dataset (~11 GB)
poetry run python scripts/download_data.py --eeg-emg   # EEG/EMG extension (~4.1 GB)

poetry run python scripts/01_reproduce_example.py  # validate against the reference
```

Open the `scripts/*.py` files as **interactive `# %%` cell scripts** in VS Code
(Python extension) or Spyder and run them cell by cell.

`00_inspect_data.py` requires the full download: it builds the ten-session
inventory, compares complete raw ΔF/F views containing 100–1,000 random neurons,
plots every neuron alongside frame-trigger-synchronized EEG, EMG, and brain
states, and maps all 7,843 neurons in `mouse01_sleep` by their row-aligned
cortical atlas labels. The EEG/EMG panel additionally requires `--eeg-emg`.
The 84 MB sample remains useful for the later quick validation exercises.

## The hands-on, in order

| Script | What you learn |
|---|---|
| `00_inspect_data.py` | Survey raw states; align all-neuron activity, EEG, and EMG; compare complete ΔF/F traces; and map neurons by cortical region |
| `01_reproduce_example.py` | Reproduce + validate the dataset's official example pipeline |
| `02_visualization_activity.py` | ΔF/F, all-neuron reference rasters, and an active-neuron official Rastermap view over complete timelines |
| `03_verify_rastermap.py` | Introduce the paper-style Rastermap workflow: active-neuron selection, normalization, PCA, sorting, and the final activity map |
| `04_validate_rastermap.py` | Check the essential validity questions: activity selection, repeatability, held-out transfer, and state dependence |
| `09_verify_state_labels.py` | Reconstruct the paper's 4-s relative-delta, normalized delta/theta, and EMG-RMS sleep rule with blocked validation; inspect anesthesia physiology without inventing an unpublished classifier |
| `10_functional_connectivity.py` | Build correlation-based functional networks per state |
| `20_modularity.py` | Density thresholding, Louvain modularity, resolution, robustness |
| `30_state_comparison.py` | **The result:** modularity is higher during sleep/anesthesia |
| `40_small_world.py` | Path length, clustering coefficient, small-world-ness / SWP |
| `50_coarse_grain_modularity.py` | Rebuild modular networks across spatial scales |
| `60_module_spatial_distribution.py` | Compare intermixed single-cell and localized mesoscale modules |

The earlier, more exhaustive Rastermap validation workflows are preserved for
reference in [`scripts/arxiv/`](scripts/arxiv/):
[`04_validate_rastermap_robustness.py`](scripts/arxiv/04_validate_rastermap_robustness.py),
[`05_validate_rastermap_across_recordings.py`](scripts/arxiv/05_validate_rastermap_across_recordings.py),
[`06_validate_rastermap_resampling.py`](scripts/arxiv/06_validate_rastermap_resampling.py),
[`07_validate_rastermap_state_specific.py`](scripts/arxiv/07_validate_rastermap_state_specific.py),
and
[`08_validate_rastermap_active_sensitivity.py`](scripts/arxiv/08_validate_rastermap_active_sensitivity.py).
They retain the full diagnostic analyses; the active `04` tutorial extracts the
parts needed to interpret Rastermap without repeating every sensitivity screen.

## Reusable package organization

Tutorials keep their settings, narrative orchestration, and one-off figure
composition in `scripts/`. General functions live in broad, discoverable
categories under `src/funcnet/`:

| Module | Put reusable functions here when they concern… |
|---|---|
| `dataio.py` | v2.0 loading, state lookup, and reproducible neuron-row selection |
| `physiology.py` | v3.0 EEG/EMG loading, frame-trigger synchronization, display panels, and reusable 4-s state-scoring features and rules |
| `timeseries.py` | frame windows, contiguous bouts, acquisition breaks, temporal shuffles, smoothing |
| `visualization.py` | display binning, state timelines, activity plots, cortical-area annotations, and spatial module maps |
| `rastermap_tools.py` | active-row selection, official Rastermap fitting, tie-aware order metrics, original-ROI mapping, and cache validation |
| `network.py` | correlation networks, density thresholding, Louvain modularity, consensus partitions |
| `smallworld.py` | clustering, path length, null networks, and small-world propensity |
| `coarsegrain.py` | spatial parcels and spatial module measures |
| `statistics.py` | general statistical summaries whose pooling assumptions are explicit |
| `paths.py` | project input/output paths |

Scripts import these modules explicitly, so readers can immediately locate an
implementation and future tutorials can extend the appropriate category. The
path is anchored to the file, so imports work from any working directory:

```python
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.funcnet import dataio, network as net, timeseries as ts
```

Generated figures/CSVs go to `results/`.
Rastermap fits are cached under `results/cache/rastermap/`; the cache is reused
only when the source-file signature, recording dimensions, package version, and
all method settings match.

`02_visualization_activity.py` retains every neuron in its two reference spike
rasters. One orders neurons by whole-session activity; the other groups the same
rows by brain atlas region. Its separate Rastermap view keeps every neuron in an
explicit dataset-active population (the network-analysis-window-specific
`nonzero_ROI` mask, intersected only with finite/nonconstant trace validity),
caches the official fit, and displays only selected rows. `nonzero_ROI` is the
dataset's activity filter from the publication's complete analysis windows
(1,500 frames for sleep and 2,900 frames for anesthesia).
The Rastermap paper reports a general 0.1--0.25-Hz minimum-rate practice but
does not publish a Figure 3 calcium-specific cutoff or selection formula;
therefore, the optional positive-bin rules are explicitly uncalibrated
support-sensitivity criteria rather than reproductions of that cutoff.
`03_verify_rastermap.py` is the compact, paper-style introduction to one
official fit and its principal intermediate representations. The combined
`04_validate_rastermap.py` then retains the conclusions needed for responsible
interpretation: use the documented `nonzero_ROI` population as the primary
activity definition, distinguish reproducible coarse population structure from
a potentially unstable fine neuron order, evaluate held-out time blocks, and
compare states with temporal controls. Activity thresholds based on positive
bins or event onsets remain uncalibrated sensitivity analyses rather than
physiological firing-rate cutoffs. Cohort summaries remain descriptive. The
larger split×seed, state-specific, and nested-selection screens that established
these choices remain available under `scripts/arxiv/`.

The reusable helpers have small, data-free regression tests (the large tutorial
recordings are not needed):

```bash
poetry run python -m unittest discover -s tests -v
poetry run ruff check src/funcnet scripts tests
```

## Notes

- The `.mat` files are MATLAB **v7.3 (HDF5)** — read with `pymatreader`, not
  `scipy.io.loadmat`. The loader also fixes v1→v2 variable renames and 1-based
  frame indices. See `.claude/rules/dataset-v2-format.md`.
- `data/` and `.venv/` are gitignored.

## License

Code: see repository. Dataset: CC-BY 4.0 (cite the paper and Zenodo DOI
10.5281/zenodo.17667863).
