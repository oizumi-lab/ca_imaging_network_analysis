# Guide: state-dependent functional-network modularity

This guide summarizes the numbered `# %%` scripts. The first part uses one
complete sleep recording so every transformation can be traced from the
original calcium data, synchronized EEG/EMG, and deposited state labels to the
final network measurements.

## Learning goals

After scripts 00–06, you should be able to:

1. distinguish neural activity, functional connectivity, and a thresholded graph;
2. explain why graph density is matched across brain states;
3. interpret Louvain modules and modularity Q;
4. handle stochastic community detection with repeated optimization;
5. compare Awake and NREM networks without treating time windows as mice; and
6. explain how spatial coarse-graining changes module geography.

## Part 1: one complete recording

### 00 — download the example data

```bash
poetry install
poetry run python scripts/00_download_data.py
```

The default download contains:

- `mouse02_sleep.mat`: complete version-3 calcium recording (~1.09 GB);
- `mouse02_sleep_physiological_data.mat`: synchronized EEG/EMG (~0.36 GB).

### 01 — inspect neural activity and state physiology

Run `01_inspect_data.py` cell by cell. The script reports dimensions and the
active-neuron count, then makes three figures:

- full-session raw ΔF/F traces from a reproducible neuron sample;
- all neuron positions colored by layer-collapsed Allen cortical area; and
- an aligned event raster, EEG spectrogram, EMG envelope, and state strip.

The physiology panels help assess whether the deposited state labels are
plausible. They are not a newly fitted state classifier.

### 02 — construct a functional network

`02_functional_connectivity.py` takes matched Awake and NREM windows. For each
state it computes Pearson correlation between every selected neuron pair.
Correlation describes statistical co-activity; it does not establish anatomical
connectivity, a synapse, or a causal relationship.

The dense matrices are converted to graphs by retaining the strongest 5% of
absolute correlations. Matching graph density gives the two graphs the same
number of edges even when Awake and NREM require different numerical
correlation thresholds. This makes their modularity values more directly
comparable.

### 03 — estimate modular structure

`03_modularity.py` introduces modularity Q and Louvain community detection. The
main practical points are:

- module labels are arbitrary identifiers;
- Louvain is stochastic, so repeated runs are used;
- the max-Q run supplies the displayed partition;
- isolated nodes require the publication's giant-component initialization; and
- graph density K and resolution γ are analysis choices that should be reported.

The spatial maps demonstrate that functionally defined modules can be
intermixed across cortex rather than forming contiguous anatomical regions.

### 04 — compare states within one recording

`04_sample_state_comparison.py` repeats the pipeline across several complete
Awake and NREM windows and several graph densities. Window-to-window variation
is useful for assessing the method, but those windows share one animal and are
not independent biological replicates.

### 05 — rebuild networks after spatial coarse-graining

`05_sample_coarse_grain_modularity.py` groups nearby neurons into parcels of
increasing size. Parcel signals are averaged first; correlation, graph
thresholding, and modularity are then recomputed from scratch at every scale.
Merging an already constructed single-cell graph would answer a different
question.

### 06 — compare module geography across scales

`06_sample_module_spatial_distribution.py` contrasts single-cell and
40-neuron-parcel maps. It also plots the probability that two nodes share a
module as a function of their cortical distance.

### Supplemental visualization

`scripts/supplemental/multiscale_module_movie.py` computes partitions at seven
scales, aligns the otherwise arbitrary module colors across neighboring scales,
saves a static overview, and renders an MP4 or GIF movie. It is optional and is
kept outside the numbered workflow.

## Part 2: population-level analysis

Download every recording before running the cohort scripts:

```bash
poetry run python scripts/00_download_data.py --all
```

Then run:

1. `07_all_mice_modularity.py` — all sleep and anesthesia recordings;
2. `08_all_mice_coarse_grain_modularity.py` — modularity across spatial scales;
3. `09_all_mice_module_spatial_distribution.py` — module geography and distance profiles.

Mouse 4 has two sleep recording days. The scripts average those days within the
same biological mouse before cohort summaries, preventing that mouse from
receiving twice the inferential weight.

Keep `PAPER_MODE = False` for a responsive preview. Set `PAPER_MODE = True` for
the full-neuron, 200-Louvain-run workflow only when the required compute time
and memory are available.

## Interpretation boundaries

- Fixed graph density controls edge count, not every activity-statistics confound.
- Functional edges do not imply direct synaptic connections.
- A high Q describes segregation relative to the modularity null model.
- Window-level points quantify within-recording variability, not population n.
- Confidence intervals that include zero do not establish equivalence.
- Coarse-grained calcium parcels are a controlled spatial-scale analysis, not a
  literal model of EEG or fMRI measurements.
