# Guide: state-dependent functional-network modularity

This guide summarizes the numbered Jupyter notebooks. The first part uses one
complete sleep recording so every transformation can be traced from the
original calcium data, synchronized EEG/EMG, and deposited state labels to the
final network measurements.

## Learning goals

After notebooks 00–06, you should be able to:

1. distinguish neural activity, functional connectivity, and a thresholded graph;
2. explain why graph density is matched across brain states;
3. interpret Louvain modules and modularity Q;
4. handle stochastic community detection with repeated optimization;
5. compare Awake and NREM networks without treating time windows as mice; and
6. explain how spatial coarse-graining changes module geography.

## Part 1: one complete recording

### 00 — download the complete dataset

```bash
poetry install
```

Open `scripts/00_download_data.ipynb` with the Poetry environment selected as
the Jupyter kernel, then run all cells. The notebook downloads:

- all 10 processed calcium recordings (~11.22 GB); and
- all synchronized EEG/EMG files plus their README (~4.06 GB).

The complete download is about 15.3 GB. Complete files are skipped on reruns,
so an interrupted download can be resumed safely.

### 01 — inspect neural activity and state physiology

Run `01_inspect_data.ipynb` cell by cell. The notebook reports dimensions and the
active-neuron count, then makes four figures:

- full-session raw ΔF/F traces from a reproducible neuron sample;
- all neuron positions colored by layer-collapsed Allen cortical area;
- population activity above a deconvolved-event raster grouped and
  activity-ranked within those cortical areas; and
- an aligned event raster, EEG spectrogram, EMG envelope, and state strip.

The physiology panels help assess whether the deposited state labels are
plausible. They are not a newly fitted state classifier.

### 02 — construct a functional network

`02_functional_connectivity.ipynb` takes matched Awake and NREM windows. For each
state it computes Pearson correlation between every selected neuron pair.
Correlation describes statistical co-activity; it does not establish anatomical
connectivity, a synapse, or a causal relationship.

The dense matrices are converted to graphs by retaining the strongest 5% of
absolute correlations. Matching graph density gives the two graphs the same
number of edges even when Awake and NREM require different numerical
correlation thresholds. This makes their modularity values more directly
comparable.

### 03 — estimate modular structure

`03_modularity.ipynb` introduces modularity Q and Louvain community detection. The
main practical points are:

- module labels are arbitrary identifiers;
- Louvain is stochastic, so repeated runs are used;
- the max-Q run supplies the displayed partition;
- isolated nodes require the publication's giant-component initialization; and
- graph density K and resolution γ are analysis choices that should be reported.

The spatial maps demonstrate that functionally defined modules can be
intermixed across cortex rather than forming contiguous anatomical regions.

### 04 — compare states within one recording

`04_sample_state_comparison.ipynb` repeats the pipeline across several complete
Awake and NREM windows and several graph densities. Window-to-window variation
is useful for assessing the method, but those windows share one animal and are
not independent biological replicates.

### 05 — rebuild networks after spatial coarse-graining

`05_sample_coarse_grain_modularity.ipynb` groups nearby neurons into parcels of
increasing size. Parcel signals are averaged first; correlation, graph
thresholding, and modularity are then recomputed from scratch at every scale.
Merging an already constructed single-cell graph would answer a different
question.

### 06 — compare module geography across scales

`06_sample_module_spatial_distribution.ipynb` contrasts single-cell and
40-neuron-parcel maps. It also plots the probability that two nodes share a
module as a function of their cortical distance.

### Supplemental visualization

`scripts/supplemental/multiscale_module_movie.ipynb` computes partitions at seven
scales, aligns the otherwise arbitrary module colors across neighboring scales,
saves a static overview, and renders an MP4 or GIF movie. It is optional and is
kept outside the numbered workflow.

## Part 2: population-level analysis

Notebook 00 has already downloaded the complete dataset. Continue with:

1. `07_all_mice_modularity.ipynb` — all sleep and anesthesia recordings;
2. `08_all_mice_coarse_grain_modularity.ipynb` — modularity across spatial scales;
3. `09_all_mice_module_spatial_distribution.ipynb` — module geography and distance profiles.

Mouse 4 has two sleep recording days. The notebooks average those days within the
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
