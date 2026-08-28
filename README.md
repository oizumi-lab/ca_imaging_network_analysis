# Functional-network analysis of calcium-imaging data

This repository is a step-by-step tutorial on functional-network analysis of
large-scale two-photon calcium-imaging recordings. It follows the data from raw
neural and physiological signals through functional connectivity, community
detection, comparisons between brain states, and changes across spatial scales.

No prior network-analysis experience is assumed. The numbered Python files are
executable tutorials: open them in order, read the explanatory cells, and run
each code cell before continuing.

The analysis is based on:

> Kiyooka & Oomoto et al. (2026). *Single-cell resolution functional networks
> during unconsciousness are segregated into spatially intermixed modules.*
> Cell Reports. https://doi.org/10.1016/j.celrep.2025.116902

- Version-3 dataset: https://neurodata.riken.jp/id/20260708-001
- Reference MATLAB analysis: https://github.com/oizumi-lab/mouse_network_2P

## What this tutorial covers

The numbered workflow answers one connected set of questions:

```text
What was recorded?
        ↓
Which neurons have similar activity over time?
        ↓
How do we turn those similarities into comparable graphs?
        ↓
Do the graphs separate into functional modules?
        ↓
Does modularity change between Awake and NREM sleep or anesthesia?
        ↓
Does the result depend on spatial scale?
        ↓
Does the result persist across mice?
```

The explanations distinguish neural activity, correlation, graph edges,
modules, and modularity Q. They also explain why graph density is matched
between conditions and why the biological mouse—not an individual time
window—is the replicate for cohort-level comparisons.

## Setup

The repository uses Python 3.12 and Poetry.

```bash
poetry install
poetry run python scripts/00_download_data.py
```

The default download retrieves the complete `mouse02_sleep` calcium recording
(about 1.09 GB) and its frame-trigger-synchronized EEG/EMG recording (about
0.36 GB). Files are checked by expected size, so an interrupted download can be
resumed safely.

Open scripts `01`–`06` in VS Code with the Python extension or in Spyder. Run
the `# %%` cells sequentially. If you change a script's `RECORDING` setting,
make sure that recording has also been downloaded. Figures and tables are
written under `results/`, which is not committed to Git.

## Start with one complete recording

Run scripts `00`–`06` in order:

| Script | Question answered | Main output |
|---|---|---|
| `00_download_data.py` | Which files are needed for the example? | Calcium and synchronized EEG/EMG data |
| `01_inspect_data.py` | What was recorded, where were neurons sampled, and do the state labels agree with the physiology? | Raw ΔF/F traces, atlas-colored neuron map, raster, EEG spectrogram, EMG, and state strip |
| `02_functional_connectivity.py` | Which neuron pairs have similar activity, and how can states be compared with the same edge count? | Correlation matrices and equal-density graphs |
| `03_modularity.py` | Does each graph contain groups with more internal edges than expected? | Repeated-Louvain partitions, modularity Q, and module maps |
| `04_sample_state_comparison.py` | Does the Awake–NREM difference recur across time windows and density choices? | Within-recording state comparison |
| `05_sample_coarse_grain_modularity.py` | What happens when nearby neural signals are averaged into parcels? | Modularity and state contrast across parcel sizes |
| `06_sample_module_spatial_distribution.py` | Are functional modules spatially intermixed or localized? | Cortical module maps and same-module probability versus distance |

The scripts include responsive defaults where the full calculation would be
slow. Comments next to `MAX_NEURONS`, window limits, and Louvain repeat counts
explain how those settings affect runtime and interpretation.

## Extend the analysis to all mice

To reproduce the cohort-level results, download every version-3 calcium and
physiology recording:

```bash
poetry run python scripts/00_download_data.py --all
```

Then run scripts `07`–`09` in order:

| Script | Purpose |
|---|---|
| `07_all_mice_modularity.py` | Compare Awake with NREM or anesthesia across recordings and biological mice |
| `08_all_mice_coarse_grain_modularity.py` | Test the state contrast across spatial scales and reproduce the paper-style Figure 7 analysis |
| `09_all_mice_module_spatial_distribution.py` | Compare module maps and same-module probability versus cortical distance across mice |

With `PAPER_MODE = False`, these scripts run a smaller preview using limited
neurons, windows, and Louvain repetitions. Set `PAPER_MODE = True` for all
selected neurons, 200 Louvain runs, and the expanded analysis. Full runs require
substantially more time and memory and can take hours or days.

Mouse 4 has two sleep recording days. The cohort scripts combine those days
within the same biological mouse before calculating cohort summaries, so one
mouse does not receive twice the weight.

## Repository layout

```text
scripts/                 numbered tutorial and cohort workflows
scripts/supplemental/    optional, unnumbered visual extensions
src/funcnet/             reusable data, network, statistics, and plotting code
tests/                   synthetic regression tests; no download required
documents/               written guide and reproducible tutorial-slide builder
references/              dataset, paper, and source-code links
data/                    downloaded recordings; ignored by Git
results/                 generated figures, tables, and movies; ignored by Git
```

Run the data-free checks with:

```bash
poetry run python -m unittest discover -s tests -v
poetry run ruff check src/funcnet scripts tests
```
