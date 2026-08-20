# CSHA Neural Data Science: functional-network modularity hands-on

This repository contains the hands-on tutorial for the **CSHA 2026 Neural Data
Science for Large-scale Recordings** course. It follows one complete
two-photon calcium-imaging recording from raw signals to a functional-network
modularity result, then shows how to test the result across mice and spatial
scales.

No prior network-analysis experience is assumed. The numbered Python scripts
are written as executable tutorials: open them in order, read the markdown
cells, and run each code cell before continuing.

The analysis is based on:

> Kiyooka & Oomoto et al. (2026). *Single-cell resolution functional networks
> during unconsciousness are segregated into spatially intermixed modules.*
> Cell Reports. https://doi.org/10.1016/j.celrep.2025.116902

- Version-3 dataset: https://neurodata.riken.jp/id/20260708-001
- Reference MATLAB analysis: https://github.com/oizumi-lab/mouse_network_2P

## What you will learn

The tutorial answers one connected set of questions:

```text
What was recorded?
        ↓
Which neurons have similar activity over time?
        ↓
How do we turn those similarities into comparable graphs?
        ↓
Do the graphs separate into functional modules?
        ↓
Does modularity change between Awake and NREM sleep?
        ↓
Does the result persist across spatial scales and biological mice?
```

By the end, you should be able to distinguish neural activity, correlation,
graph edges, modules, and modularity Q; explain why graph density is matched
between conditions; and identify the biological mouse—not a time window—as the
replicate for a cohort-level comparison.

## Setup

The repository uses Python 3.12 and Poetry.

```bash
poetry install
poetry run python scripts/00_download_data.py
```

The default download retrieves the complete `mouse02_sleep` calcium recording
(about 1.09 GB) and its frame-trigger-synchronized EEG/EMG recording (about
0.36 GB). Files are checked by expected size, so an interrupted download can be
run again safely.

Open scripts `01`–`06` in VS Code with the Python extension or in Spyder. Run
the `# %%` cells sequentially. Figures and tables are written under `results/`;
that directory is not committed to Git.

## Recommended course path: one complete recording

Run scripts `00`–`06` in order. Each step uses `mouse02_sleep`, allowing every
network quantity to be traced back to the signals inspected at the beginning.

| Script | Question answered | Main output |
|---|---|---|
| `00_download_data.py` | Which files are needed for the course example? | Calcium and synchronized EEG/EMG data |
| `01_inspect_data.py` | What was recorded, where were neurons sampled, and do the state labels agree with the physiology? | Raw ΔF/F traces, atlas-colored neuron map, raster, EEG spectrogram, EMG, and state strip |
| `02_functional_connectivity.py` | Which neuron pairs have similar activity, and how can states be compared with the same edge count? | Correlation matrices and equal-density graphs |
| `03_modularity.py` | Does each graph contain groups with more internal edges than expected? | Repeated-Louvain partitions, modularity Q, and module maps |
| `04_sample_state_comparison.py` | Does the Awake–NREM difference recur across time windows and density choices? | Within-recording state comparison |
| `05_sample_coarse_grain_modularity.py` | What happens when nearby neural signals are averaged into parcels? | Modularity and state contrast across parcel sizes |
| `06_sample_module_spatial_distribution.py` | Are functional modules spatially intermixed or localized? | Cortical module maps and same-module probability versus distance |

Each analysis script ends with a practice prompt. The repository does not
include solution cells. Add your work in a new cell and use the neighboring code
as a pattern without replacing the supplied workflow.

## Optional supplemental analysis

[scripts/supplemental/multiscale_module_movie.py](scripts/supplemental/multiscale_module_movie.py)
extends scripts `05`–`06` by rebuilding the network at seven spatial scales and
rendering the resulting module maps as a movie. Supplemental scripts use
descriptive filenames without numeric prefixes because they are not required
steps in the main sequence.

## Research extension: all mice

Attendees who want to reproduce the cohort-level paper results or begin a
course project can download every version-3 calcium and physiology recording:

```bash
poetry run python scripts/00_download_data.py --all
```

Then run scripts `07`–`09` in order:

| Script | Purpose |
|---|---|
| `07_all_mice_modularity.py` | Compare Awake with NREM or anesthesia across recordings and biological mice |
| `08_all_mice_coarse_grain_modularity.py` | Test the state contrast across spatial scales and reproduce the paper-style Fig. 7 analysis |
| `09_all_mice_module_spatial_distribution.py` | Compare module maps and same-module probability versus cortical distance across mice |

The default settings limit the selected neurons, windows, and Louvain
repetitions so the workflow can be inspected interactively. Set
`PAPER_MODE = True` in scripts `07`–`09` for all selected neurons, 200 Louvain
runs, and the expanded analysis. These runs require substantially more time and
memory.

Mouse 4 has two sleep recording days. The all-mice scripts combine those days
within the same biological mouse before cohort summaries so that one mouse does
not receive twice the weight.

## Core network workflow

```text
smoothed deconvolved activity
        ↓  Pearson correlation
one value for every neuron pair
        ↓  retain the strongest |r| values at density K
binary graph with a matched edge count in every state
        ↓  repeat Louvain community detection
max-Q module partition and modularity Q
```

Here, a functional edge represents statistical co-activity; it does not imply a
direct synapse or causal influence. Matching density controls the number of
edges, but it does not remove every possible difference in activity statistics.
Louvain is stochastic, so the scripts repeat the optimization and retain the
highest-Q partition, following the published workflow.

## Repository layout

```text
scripts/                 numbered course and all-mice workflows
scripts/supplemental/    optional, unnumbered extensions
src/funcnet/             reusable loading, network, spatial, and plotting code
tests/                   synthetic regression tests that need no downloaded data
documents/               tutorial guide and reproducible slide-deck builder
data/                    downloaded recordings; ignored by Git
results/                 generated figures, tables, and movies; ignored by Git
```

The reusable package is organized as follows:

| Module | Responsibility |
|---|---|
| `dataio.py` | Load version-3 MATLAB/HDF5 recordings, states, and active-neuron rows |
| `physiology.py` | Load and frame-trigger-align EEG/EMG and prepare display features |
| `timeseries.py` | Construct stable-state windows and acquisition segments |
| `network.py` | Calculate correlation, fixed-density graphs, and Louvain modularity |
| `coarsegrain.py` | Build spatial parcels and summarize module distance dependence |
| `visualization.py` | Plot states, activity, cortical regions, and module maps |
| `statistics.py` | Calculate mouse-level confidence summaries |
| `paths.py` | Define portable input and output locations |

## Tutorial slides

After the analysis figures have been generated, rebuild the hands-on deck with:

```bash
poetry run python documents/build_handson_slides.py
```

This creates `documents/CSHA_handson_tutorial.pptx`, which summarizes the
script-by-script flow and the transition from the one-recording tutorial to the
all-mice extension. The generated deck and the large source talk
`documents/CSHA_082426.pptx` are intentionally ignored by Git; the small builder
script is the reproducible source kept in the repository.

## Tests

The tests use synthetic, data-free fixtures:

```bash
poetry run python -m unittest discover -s tests -v
poetry run ruff check src/funcnet scripts tests documents/build_handson_slides.py
```

## Archive and attribution

The broader pre-course code—including DMD, graphical lasso, Rastermap,
small-world, and verification studies—is preserved in branch
`archive/full-analysis-2026` and tag `full-analysis-2026-08-18`. The course-facing
main branch remains focused on the modularity tutorial while Git retains the
full development history.

The datasets are released under CC BY 4.0. Cite the version-3 dataset release
and the Cell Reports paper when reusing the data, scripts, or derived results.
