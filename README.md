# ca_imaging_network_analysis

Hands-on materials for **state-dependent functional network analysis** of
large-scale two-photon calcium imaging in mouse cortex, comparing functional
networks across **wakefulness, sleep, and anesthesia**.

Built for a neural-data-analysis course. The first module reproduces and teaches
the **modularity** analysis from:

> Kiyooka & Oomoto et al. (2026). *Single-cell resolution functional networks
> during unconsciousness are segregated into spatially intermixed modules.*
> **Cell Reports.** https://doi.org/10.1016/j.celrep.2025.116902

- **Dataset** (RIKEN 20260409-001, v2.0, CC-BY 4.0): https://neurodata.riken.jp/id/20260409-001
- **Reference MATLAB code**: https://github.com/oizumi-lab/mouse_network_2P

## Quick start

Requires [pyenv](https://github.com/pyenv/pyenv) (Python 3.12.13) and
[Poetry](https://python-poetry.org/).

```bash
poetry install                                     # build ./.venv, install deps
poetry run python scripts/download_data.py --example   # 84 MB sample (fast)
# or:  poetry run python scripts/download_data.py       # full dataset (~11 GB)

poetry run python scripts/01_reproduce_example.py  # validate against the reference
```

Open the `scripts/*.py` files as **interactive `# %%` cell scripts** in VS Code
(Python extension) or Spyder and run them cell by cell.

## The hands-on, in order

| Script | What you learn |
|---|---|
| `00_inspect_data.py` | What's in a recording: traces, brain states, neuron positions |
| `01_reproduce_example.py` | Reproduce + validate the dataset's official example pipeline |
| `02_visualization_activity.py` | Full timelines for all sleep/anesthesia sessions: sampled/all-neuron ΔF/F views and spike rasters |
| `10_functional_connectivity.py` | Build correlation-based functional networks per state |
| `20_modularity.py` | Density thresholding, Louvain modularity, resolution, robustness |
| `30_state_comparison.py` | **The result:** modularity is higher during sleep/anesthesia |
| `40_small_world.py` | Path length, clustering coefficient, small-world-ness / SWP |
| `50_coarse_grain_modularity.py` | Rebuild modular networks across spatial scales |
| `60_module_spatial_distribution.py` | Compare intermixed single-cell and localized mesoscale modules |

## Reusable package organization

Tutorials keep their settings, narrative orchestration, and one-off figure
composition in `scripts/`. General functions live in broad, discoverable
categories under `src/funcnet/`:

| Module | Put reusable functions here when they concern… |
|---|---|
| `dataio.py` | v2.0 loading, state lookup, and reproducible neuron-row selection |
| `timeseries.py` | frame windows, contiguous bouts, acquisition breaks, temporal shuffles, smoothing |
| `visualization.py` | display binning, state timelines, activity plots, and spatial module maps |
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
