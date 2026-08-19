# Hands-on guide: state-dependent functional-network modularity

This guide accompanies the numbered `# %%` scripts. The recommended course path
uses one complete sleep recording so students see the original calcium data,
synchronized EEG/EMG, state labels, and every network-analysis transformation.

## Learning goals

By the end of scripts 00--07, students should be able to:

1. distinguish neural activity, functional connectivity, and a thresholded graph;
2. explain why graph density must be matched across brain states;
3. compute and interpret Louvain modularity Q;
4. handle stochastic community detection with repeated optimization;
5. compare Awake and NREM networks without treating time windows as mice;
6. explain how spatial coarse-graining changes module geography; and
7. modify one setting and interpret a short practice analysis in every script.

## Track A: one complete recording

### 00 — download the course data

```bash
poetry install
poetry run python scripts/00_download_data.py
```

The default download contains:

- `mouse02_sleep.mat`: complete version-3 calcium recording (~1.09 GB);
- `mouse02_sleep_physiological_data.mat`: synchronized EEG/EMG (~0.36 GB).

### 01 — inspect neural activity and state physiology

Run `01_inspect_data.py` cell by cell. Confirm the dimensions and active-neuron
count. The script first makes two standalone figures:

- full-session raw ΔF/F traces from 100 reproducibly selected neurons;
- all neuron positions colored by layer-collapsed Allen cortical area.

The third figure aligns:

- the all-neuron deconvolved-event raster;
- the EEG spectrogram;
- the EMG RMS envelope;
- the deposited Wake / quiet-Wake / NREM / REM state labels.

The state panel is an inspection of physiological support for the deposited
labels. It is not a newly fitted state classifier.

### 02 — construct a functional network

`02_functional_connectivity.py` takes matched 1,500-frame Awake and NREM windows.
For each state it computes Pearson correlation between every selected neuron pair.
The correlation is statistical co-activity, not anatomical connectivity.

The dense matrices are converted to graphs by retaining the strongest 5% of
absolute correlations. Matching K guarantees the two graphs have the same edge
count, even though Awake and NREM require different numerical correlation
thresholds. This is the input needed for the modularity comparison in script 03.
The unsolved exercise asks attendees to change K and check how the threshold and
edge count respond.

### 03 — estimate modular structure

`03_modularity.py` introduces modularity Q and Louvain community detection. The
important practical points are:

- module labels are arbitrary identifiers;
- Louvain is stochastic, so a single run is insufficient;
- repeated runs provide a max-Q partition;
- isolated nodes require the paper's giant-component initialization;
- density K and resolution γ are analysis choices that should be checked.

The spatial maps demonstrate that functional modules can be intermixed across
the cortex rather than forming contiguous anatomical regions.

### 04 — compare states within the example mouse

`04_sample_state_comparison.py` repeats the pipeline across several complete
Awake and NREM windows and across graph densities. Window-to-window variation
is useful for method exploration, but the windows share one animal and are not
independent biological replicates.

### 05 — rebuild networks after spatial coarse-graining

`05_sample_coarse_grain_modularity.py` groups nearby neurons into parcels of
1, 2, 5, 10, 20, or 40 neurons. Parcel signals are averaged first, and then the
correlation graph and modularity are recomputed from scratch at every scale.

### 06 — compare module geography across scales

`06_sample_module_spatial_distribution.py` contrasts single-cell and 40-neuron
parcel maps. It also plots the probability that two nodes share a module versus
their cortical distance.

### 07 — animate the multiscale transition

`07_multiscale_module_movie.py` computes partitions at seven scales, aligns the
otherwise arbitrary module colors across neighboring scales, saves a static
overview, and renders an MP4 movie.

### Practice analyses

Each core script ends with an unsolved exercise prompt. Solution code is
deliberately omitted: early exercises can be written by adapting nearby cells,
while later extensions are suitable for carefully checked AI assistance.

1. **Easy:** compare average activity between Awake and NREM;
2. **Easy:** change graph density and inspect edge counts and thresholds;
3. **Easy–intermediate:** change resolution and inspect the module count;
4. **Intermediate:** calculate state contrast across densities;
5. **Intermediate:** find where the spatial-scale contrast is smallest;
6. **Intermediate–advanced:** construct a module-localization contrast; and
7. **Advanced, AI-assisted:** extend the multiscale comparison to both states.

## Track B: reproduce population-level paper results

Students using the dataset for projects should first download every recording:

```bash
poetry run python scripts/00_download_data.py --all
```

Then run:

1. `08_all_mice_modularity.py` — all sleep and anesthesia recordings;
2. `09_all_mice_coarse_grain_modularity.py` — modularity across spatial scales;
3. `10_all_mice_module_spatial_distribution.py` — module geography and distance profiles.

Mouse 4 has two sleep recording days. The scripts average those days within the
same biological mouse before cohort summaries, preventing that mouse from
receiving twice the inferential weight.

For an exploratory run, retain the default teaching-sized settings. For the
full-neuron, 200-Louvain-run workflow, set `PAPER_MODE = True`. Expect long run
times and substantial memory use.

## Interpretation boundaries

- Fixed graph density controls edge count, not every activity-statistics confound.
- Functional edges do not imply direct synaptic connections.
- A high Q describes segregation relative to the modularity null model.
- Window-level points quantify within-recording variability, not population n.
- Confidence intervals that include zero do not establish equivalence.
- Coarse-grained calcium parcels are a controlled spatial-scale analysis, not a
  literal model of EEG or fMRI measurements.
