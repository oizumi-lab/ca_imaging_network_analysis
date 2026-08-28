# %% [markdown]
# # 01 · Inspect one complete calcium-imaging and EEG/EMG recording
#
# ## Where this script fits
# The tutorial follows one recording, ``mouse02_sleep``, from the measured
# signals to a network result:
#
# ```text
# inspect the recording → build a network → find modules → compare brain states
# ```
#
# Before calculating anything, we need to know what was measured. A result can
# be misleading if, for example, a recording contains too little Awake data, the
# neural traces contain long gaps, or the state labels disagree with the EEG and
# EMG. This script therefore answers five basic questions:
#
# 1. How many neurons and time points were recorded?
# 2. What does activity from individual neurons look like?
# 3. Which cortical areas were sampled?
# 4. How is population activity distributed across those cortical areas over time?
# 5. Do the EEG and EMG patterns make the deposited sleep-state labels plausible?
#
# Scripts 01--06 use this same recording. The optional supplemental movie also
# uses it. This lets you trace every later network quantity back to the data
# inspected here before moving to the all-mice analyses in scripts 07--09.
#
# ## How to read and modify this file
#
# Lines beginning with ``# %%`` divide this Python file into notebook-like
# cells. Editors such as VS Code and Spyder can run one cell at a time. A
# ``# %% [markdown]`` cell is explanatory text; a plain ``# %%`` cell is code.
# Running from top to bottom is important because later cells reuse variables
# created earlier.
#
# A few Python conventions used throughout the tutorial:
#
# - ``UPPER_CASE`` names are settings intended to be easy to find and edit.
# - ``rec`` is the loaded recording object; attributes such as ``rec.dFF`` hold
#   NumPy arrays.
# - Neural arrays have shape ``(neurons, frames)``. Row 10 is one neuron and
#   column 100 is one time point.
# - ``rows`` contains selected neuron indices and ``frames`` contains selected
#   time indices. They are indices, not measured values.
# - ``fig`` is a complete Matplotlib figure and ``ax`` is one plotting panel.
# - A function call such as ``dataio.load_recording(RECORDING)`` sends the value
#   in parentheses to a reusable function and receives its returned result.
#
# For a first pass, change only the settings below, then rerun the whole file.

# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

from src.funcnet import (
    dataio,
    physiology as physio,
    timeseries as ts,
    visualization as viz,
)
from src.funcnet.paths import FIG_DIR

FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Settings you may safely edit
#
# - ``RECORDING`` selects the dataset. Use a name printed by
#   ``dataio.list_recordings()``.
# - ``TRACE_NEURONS`` controls how many raw traces are drawn, not how many
#   neurons are retained in later network analyses.
# - ``TRACE_SEED`` makes the random trace selection repeatable. A different
#   integer displays a different reproducible sample.
# - ``DISPLAY_BIN_SECONDS`` changes only the raster display resolution.
# - ``POPULATION_SMOOTH_SECONDS`` controls smoothing of the active-neuron
#   percentage above the brain-area-grouped raster.
# - ``EEG_MAX_FREQUENCY_HZ`` and ``EEG_WINDOW_SECONDS`` control the EEG figure.
#   They do not modify the calcium-network calculations in later scripts.

# %%
RECORDING = "mouse01_sleep"  # dataset name, without a file extension
TRACE_NEURONS = 100          # number of randomly selected raw traces to plot
TRACE_SEED = 7               # integer seed for a reproducible random selection
DISPLAY_BIN_SECONDS = 1.0    # temporal width of one raster-display column
POPULATION_SMOOTH_SECONDS = 5.0  # smoothing width for population activity
EEG_MAX_FREQUENCY_HZ = 25.0  # highest EEG frequency shown
EEG_WINDOW_SECONDS = 4.0     # time span used for each spectrogram estimate

# %% [markdown]
# ## Load and summarize the complete recording
#
# ``state`` contains the deposited behavioral labels. ``used_frame`` supplies
# stable state epochs used in the network analysis, while ``nonzero_ROI`` marks
# neurons that passed the publication's activity criterion.
#
# The recording contains several representations of neural activity. Raw
# ``dFF`` is useful for inspecting fluorescence traces. ``spike_deconv`` marks
# inferred activity events for the raster display, and ``spike_smoothed`` is the
# continuous activity estimate used later to calculate correlations. Keeping
# these roles separate prevents a display choice from being mistaken for an
# analysis choice.

# %%
# ``load_recording`` reads the files and packages related arrays and metadata in
# one object. ``select_neuron_rows`` returns the indices that passed the
# publication's activity criterion.
rec = dataio.load_recording(RECORDING)
active_rows = dataio.select_neuron_rows(rec)
print(rec)
print(f"Sampling rate       : {rec.fs:.2f} Hz")
print(f"Duration            : {rec.n_frames / rec.fs / 60:.1f} min")
print(f"All neurons         : {rec.n_neurons:,}")
print(f"Analysis neurons    : {active_rows.size:,}")
print("Stable analysis frames:")
for label in rec.state_labels:
    frames = dataio.state_frames(rec, label)
    print(f"  {label:<12}: {frames.size:>6,} ({frames.size / rec.fs / 60:5.1f} min)")

# %% [markdown]
# ## Figure 1 — full-session raw traces from 100 random neurons
#
# This figure retains the complete raw 7.65-Hz ΔF/F sequence. The selected
# neurons are median-centered and vertically offset only for display: there is
# no smoothing, temporal binning, clipping, or per-neuron amplitude scaling.
# The random seed makes the selected rows exactly reproducible.
#
# Programming note: square brackets select array entries. For example,
# ``rec.dFF[trace_rows]`` keeps the requested neuron rows and every frame.
# NumPy operations with ``axis=1`` work separately across time for each neuron.

# %%
duration_min = rec.n_frames / rec.fs / 60
boundary_minutes = [
    (int(boundary) + 1) / rec.fs / 60
    for boundary in np.asarray(rec.boundary_ind).ravel()
    if 0 <= int(boundary) < rec.n_frames - 1
]
timeline_view = {
    "n_frames": rec.n_frames,
    "fs": rec.fs,
    "duration_min": duration_min,
    "time_limits_min": (0.0, duration_min),
    "state": rec.state,
    "codes": dict(dataio.state_codes(rec)),
    "boundary_minutes": boundary_minutes,
}

# Create a local random-number generator rather than changing global randomness.
rng = np.random.default_rng(TRACE_SEED)
trace_rows = np.sort(rng.choice(rec.n_neurons, size=TRACE_NEURONS, replace=False))
traces = rec.dFF[trace_rows]
# ``keepdims=True`` keeps a (neurons, 1) result so NumPy can subtract one median
# from every time point in the corresponding neuron row (broadcasting).
centered = traces - np.nanmedian(traces, axis=1, keepdims=True)
q01, q99 = np.nanpercentile(centered, (1, 99))
spacing = 1.15 * max(float(q99 - q01), 1e-6)
offsets = -np.arange(TRACE_NEURONS) * spacing
time_min = np.arange(rec.n_frames) / rec.fs / 60
# Each pair is ``(start, stop)`` for one continuous acquisition segment. Plotting
# segments separately avoids drawing a false line across an acquisition gap.
acquisition_segments = ts.acquisition_segments(rec.n_frames, rec.boundary_ind)

fig = plt.figure(figsize=(15, 10), constrained_layout=True)
grid = fig.add_gridspec(2, 1, height_ratios=(9, 0.48))
trace_ax = fig.add_subplot(grid[0])
state_ax = fig.add_subplot(grid[1], sharex=trace_ax)
viz.shade_states(trace_ax, timeline_view)
# ``zip`` walks through one trace and its matching vertical offset together.
for trace, offset in zip(centered, offsets):
    for start, stop in acquisition_segments:
        trace_ax.plot(
            time_min[start:stop],
            trace[start:stop] + offset,
            color="0.08",
            lw=0.35,
            rasterized=True,
            zorder=2,
        )

tick_rows = np.unique(np.linspace(0, TRACE_NEURONS - 1, 11).astype(int))
trace_ax.set_yticks(offsets[tick_rows])
trace_ax.set_yticklabels(trace_rows[tick_rows], fontsize=7)
trace_ax.set_ylim(offsets[-1] - spacing, spacing)
trace_ax.set_xlim(0, duration_min)
trace_ax.set_ylabel("neuron row (0-based)\n(raw ΔF/F, offset)")
trace_ax.tick_params(axis="x", labelbottom=False)
trace_ax.set_title(
    f"{rec.name} · Sleep recording: wakefulness and sleep stages\n"
    f"random {TRACE_NEURONS} of {rec.n_neurons:,} neurons · full recorded sequence"
)
viz.mark_acquisition_boundaries(trace_ax, timeline_view)

target = 0.6 * spacing
power = 10.0 ** np.floor(np.log10(target))
scale = next(
    (multiplier * power for multiplier in (5.0, 2.0, 1.0) if multiplier * power <= target),
    power,
)
y0 = offsets[2]
transform = blended_transform_factory(trace_ax.transAxes, trace_ax.transData)
trace_ax.plot(
    [0.985, 0.985],
    [y0, y0 + scale],
    color="black",
    lw=1.5,
    transform=transform,
    clip_on=False,
    zorder=4,
)
trace_ax.text(
    0.975,
    y0 + scale / 2,
    f"{scale:g} ΔF/F",
    ha="right",
    va="center",
    fontsize=8,
    transform=transform,
    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1},
    zorder=4,
)
viz.plot_state_strip(state_ax, timeline_view)
fig.suptitle(
    f"Full-session raw ΔF/F example ({TRACE_NEURONS} random neurons)",
    fontsize=15,
)
trace_path = FIG_DIR / "01_raw_dff_traces.png"
fig.savefig(trace_path, dpi=160, bbox_inches="tight")
plt.show()
print("saved ->", trace_path)

# %% [markdown]
# ## Figure 2 — spatial distribution by cortical area
#
# Every recorded neuron is colored by its row-aligned Allen-atlas annotation.
# The terminal layer ``2/3`` suffix is collapsed for display. Motor areas use
# greens, somatosensory areas warm colors, retrosplenial areas blues, and visual
# areas purples. The map includes all neurons, not only the network-analysis
# subset. This check matters because a network observed in only one small patch
# of cortex should not be interpreted as a whole-cortex network.
#
# Programming note: expressions such as ``display_regions == region`` produce a
# Boolean mask (one True/False value per neuron). Using that mask inside square
# brackets selects only neurons belonging to the requested cortical region.

# %%
if rec.atlas is None:
    raise RuntimeError("This recording did not provide row-aligned atlas labels")
atlas = np.asarray(rec.atlas, dtype=str)
if atlas.shape != (rec.n_neurons,):
    raise ValueError("Atlas labels are not aligned one-to-one with neuron rows")

display_regions = viz.cortical_region_labels(atlas)
present_regions = set(display_regions.tolist())
region_order = tuple(
    region for region in viz.CORTICAL_REGION_COLORS if region in present_regions
)
print("Cortical-area counts (layer suffix collapsed):")
for region in region_order:
    print(f"  {region:<10} {np.count_nonzero(display_regions == region):5,d}")

coords = rec.centroid_um
fig, ax = plt.subplots(figsize=(10.5, 6.2))
background_regions = tuple(
    region
    for region in ("Unassigned", "Other", "Unknown")
    if region in region_order
)
named_regions = tuple(
    region
    for region in region_order
    if region not in {"Unassigned", "Other", "Unknown"}
)
# The leading ``*`` unpacks both tuples into one iteration sequence.
for region in (*background_regions, *named_regions):
    selected = display_regions == region
    ax.scatter(
        coords[selected, 0],
        coords[selected, 1],
        s=5,
        color=viz.CORTICAL_REGION_COLORS[region],
        alpha=0.88,
        linewidths=0,
        rasterized=True,
        zorder=1 if region in {"Unassigned", "Other", "Unknown"} else 2,
    )

ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(
    f"{rec.name} · all {rec.n_neurons:,} neurons by cortical area\n"
    "Allen atlas layer 2/3 labels"
)
legend = ax.legend(
    handles=viz.cortical_region_legend_handles(region_order, unabridged=True),
    loc="center left",
    bbox_to_anchor=(1.01, 0.5),
    frameon=False,
    handletextpad=0.35,
    borderaxespad=0,
)
for text, region in zip(legend.get_texts(), region_order):
    text.set_color(viz.CORTICAL_REGION_COLORS[region])
for spine in ax.spines.values():
    spine.set_color("0.35")
    spine.set_linewidth(0.9)
fig.tight_layout()
spatial_path = FIG_DIR / "01_spatial_distribution_by_area.png"
fig.savefig(spatial_path, dpi=160, bbox_inches="tight")
plt.show()
print("saved ->", spatial_path)

# %% [markdown]
# ## Figure 3 — population activity and a brain-area-grouped raster
#
# The top panel summarizes population activity as the percentage of neurons with
# a positive deconvolution sample at each imaging frame. A short moving average
# makes the full-session trend visible without changing any later network input.
# Smoothing is performed separately within each continuous acquisition segment,
# so a microscope break cannot blend activity from two disconnected periods.
#
# The raster uses the same positive deconvolution events, combined into short
# display bins. Neurons are grouped by their layer-collapsed atlas region and
# ranked from more to less active within each region. The colored strip and
# horizontal separators mark region boundaries; they do not color the activity
# events themselves. Faint background colors and the bottom strip show the
# deposited brain states.
#
# This is a descriptive view. An apparent regional difference can reflect the
# number of sampled neurons, event rate, or visual density and should be tested
# quantitatively before it is interpreted as a biological effect.

# %%
raster, active_counts, bin_frames, activity_order, bin_centers = (
    viz.binned_spike_raster(
        rec.spike_deconv,
        rec.fs,
        DISPLAY_BIN_SECONDS,
        rec.boundary_ind,
        rec.state,
    )
)
bin_centers_min = bin_centers / rec.fs / 60

# ``raster`` is already activity-ranked. Group positions from that ordering by
# atlas region to retain the within-region activity ranking.
activity_ranked_regions = display_regions[activity_order]
grouped_positions = np.concatenate(
    [
        np.flatnonzero(activity_ranked_regions == region)
        for region in region_order
    ]
)
grouped_raster = raster[grouped_positions]
grouped_neuron_rows, grouped_occupied_bins = np.nonzero(grouped_raster)

region_limits = []
region_start = 0
for region in region_order:
    region_stop = region_start + np.count_nonzero(display_regions == region)
    region_limits.append((region, region_start, region_stop))
    region_start = region_stop

population_percent = 100 * active_counts / rec.n_neurons
population_smooth_frames = max(
    1,
    int(round(POPULATION_SMOOTH_SECONDS * rec.fs)),
)
population_percent_smoothed = ts.segmented_moving_average(
    population_percent,
    population_smooth_frames,
    rec.boundary_ind,
)
population_mean = float(np.mean(population_percent))

state_labels = dict(viz.DEFAULT_STATE_LABELS)
state_labels.update(
    {"awake": "Wake", "quiet_awake": "Quiet awake", "nrem": "NREM", "rem": "REM"}
)
state_short_labels = dict(viz.DEFAULT_STATE_SHORT_LABELS)
state_short_labels.update(
    {"awake": "W", "quiet_awake": "Q", "nrem": "N", "rem": "R"}
)

fig = plt.figure(figsize=(15.5, 10), constrained_layout=True)
grid = fig.add_gridspec(
    4,
    2,
    width_ratios=(0.022, 1),
    height_ratios=(1.15, 5.3, 0.42, 0.85),
    hspace=0.06,
    wspace=0.025,
)
population_ax = fig.add_subplot(grid[0, 1])
region_ax = fig.add_subplot(grid[1, 0])
grouped_raster_ax = fig.add_subplot(grid[1, 1], sharex=population_ax)
grouped_state_ax = fig.add_subplot(grid[2, 1], sharex=population_ax)
region_legend_ax = fig.add_subplot(grid[3, 1])

viz.shade_states(population_ax, timeline_view, alpha=0.09)
for start, stop in acquisition_segments:
    population_ax.plot(
        time_min[start:stop],
        population_percent_smoothed[start:stop],
        color="0.12",
        lw=0.9,
        zorder=2,
    )
population_ax.axhline(
    population_mean,
    color="0.4",
    lw=0.8,
    ls=(0, (3, 2)),
    zorder=1,
)
population_ax.set_xlim(0, duration_min)
population_ax.set_ylim(bottom=0)
population_ax.set_ylabel("active neurons\n(%)")
population_ax.tick_params(axis="x", labelbottom=False)
population_ax.text(
    0.995,
    0.92,
    f"{POPULATION_SMOOTH_SECONDS:g}-s moving average · dashed = session mean",
    transform=population_ax.transAxes,
    ha="right",
    va="top",
    fontsize=7,
    color="0.25",
)
viz.mark_acquisition_boundaries(population_ax, timeline_view)

viz.shade_states(grouped_raster_ax, timeline_view, alpha=0.075)
grouped_raster_ax.scatter(
    bin_centers_min[grouped_occupied_bins],
    grouped_neuron_rows,
    s=0.08,
    color="black",
    alpha=0.30,
    marker=".",
    linewidths=0,
    rasterized=True,
    zorder=2,
)
for _region, _start, stop in region_limits[:-1]:
    grouped_raster_ax.axhline(stop - 0.5, color="white", lw=0.6, zorder=3)
grouped_raster_ax.set_xlim(0, duration_min)
grouped_raster_ax.set_ylim(rec.n_neurons - 0.5, -0.5)
grouped_raster_ax.set_yticks((0, rec.n_neurons - 1))
grouped_raster_ax.set_yticklabels(("1", f"{rec.n_neurons:,}"))
grouped_raster_ax.set_ylabel(
    f"all {rec.n_neurons:,} neurons\n(grouped by atlas region)"
)
grouped_raster_ax.tick_params(axis="x", labelbottom=False)
grouped_raster_ax.text(
    0.995,
    0.02,
    f"activity-ranked within each region · display bins ≤ {bin_frames / rec.fs:.2f} s",
    transform=grouped_raster_ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=7,
    color="0.25",
    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1},
    zorder=4,
)
viz.mark_acquisition_boundaries(grouped_raster_ax, timeline_view)

for region, start, stop in region_limits:
    region_ax.axhspan(
        start - 0.5,
        stop - 0.5,
        color=viz.CORTICAL_REGION_COLORS[region],
        lw=0,
    )
region_ax.set_xlim(0, 1)
region_ax.set_ylim(rec.n_neurons - 0.5, -0.5)
region_ax.set_xticks([])
region_ax.set_yticks([])
region_ax.set_ylabel("atlas\nregion", fontsize=8)
for spine in region_ax.spines.values():
    spine.set_visible(False)

viz.plot_state_strip(
    grouped_state_ax,
    timeline_view,
    state_labels=state_labels,
    short_labels=state_short_labels,
)
grouped_state_ax.set_xlabel("recorded time (min)")

region_legend_ax.set_axis_off()
region_legend_ax.legend(
    handles=viz.cortical_region_legend_handles(region_order),
    loc="center",
    ncol=min(6, len(region_order)),
    frameon=False,
    title="Atlas region (activity-ranked within each region)",
    fontsize=8,
    title_fontsize=8,
    handletextpad=0.3,
    columnspacing=1.2,
)

fig.suptitle(
    f"{rec.name}: brain-area-grouped deconvolved activity across the full recording",
    fontsize=15,
)
grouped_raster_path = FIG_DIR / "01_brain_area_grouped_raster.png"
fig.savefig(grouped_raster_path, dpi=160, bbox_inches="tight")
plt.show()
print("saved ->", grouped_raster_path)

# %% [markdown]
# ## Figure 4 — raster, EEG spectrogram, EMG, and deposited state labels
#
# Frame-trigger synchronization aligns the separate 5-kHz physiology recording
# to the calcium frames. The EEG panel shows 0.5--25-Hz spectral power. The EMG
# panel shows a 20--200-Hz, 250-ms RMS envelope. These display filters do not
# alter the signals used for network analysis.
#
# The state strip lets you examine why high delta power and low EMG support
# NREM labels, whereas awake periods typically have lower relative delta and
# stronger muscle activity. The labels are supplied by the dataset; this script
# visualizes their physiological basis rather than training a new classifier.
#
# The helper functions below return processed values for plotting. Their inputs
# state the important choices explicitly: signal arrays, sampling rates, time
# bins, and alignment information. If you change a display setting, trace that
# setting into the matching function call before interpreting the output.

# %%
neuron_rows, occupied_bins = np.nonzero(raster)

physiology = physio.load_physiology(rec.name)
alignment = physio.align_frame_triggers(
    physiology,
    n_frames=rec.n_frames,
    boundary_ind=rec.boundary_ind,
    imaging_fs=rec.fs,
)
eeg_panels, eeg_limits = physio.prepare_eeg_spectrogram(
    physiology,
    alignment,
    imaging_fs=rec.fs,
    max_frequency_hz=EEG_MAX_FREQUENCY_HZ,
    window_seconds=EEG_WINDOW_SECONDS,
)
emg_panels, emg_limit = physio.prepare_emg_envelope(
    physiology,
    alignment,
    imaging_fs=rec.fs,
)

fig = plt.figure(figsize=(13.5, 10), constrained_layout=True)
grid = fig.add_gridspec(
    4,
    2,
    width_ratios=(1, 0.025),
    height_ratios=(4.5, 1.6, 0.9, 0.36),
    hspace=0.06,
    wspace=0.04,
)
raster_ax = fig.add_subplot(grid[0, 0])
eeg_ax = fig.add_subplot(grid[1, 0], sharex=raster_ax)
emg_ax = fig.add_subplot(grid[2, 0], sharex=raster_ax)
state_ax = fig.add_subplot(grid[3, 0], sharex=raster_ax)
colorbar_ax = fig.add_subplot(grid[1, 1])

raster_ax.scatter(
    bin_centers_min[occupied_bins],
    neuron_rows,
    s=0.08,
    color="black",
    marker=".",
    linewidths=0,
    rasterized=True,
)
raster_ax.set_xlim(0, duration_min)
raster_ax.set_ylim(rec.n_neurons - 0.5, -0.5)
raster_ax.set_ylabel(f"all {rec.n_neurons:,} neurons\n(activity-ranked)")
raster_ax.set_title(
    f"{rec.name} · positive deconvolution samples in "
    f"~{bin_frames / rec.fs:.1f}-s display bins"
)
raster_ax.tick_params(axis="x", labelbottom=False)
viz.mark_acquisition_boundaries(raster_ax, timeline_view)

spectrogram_image = None
for panel in eeg_panels:
    spectrogram_image = eeg_ax.pcolormesh(
        panel.time_min,
        panel.frequency_hz,
        panel.power_db_hz,
        shading="auto",
        cmap="turbo",
        vmin=eeg_limits[0],
        vmax=eeg_limits[1],
        rasterized=True,
    )
if spectrogram_image is None:
    raise RuntimeError("No aligned EEG spectrogram was prepared")
eeg_ax.set_ylim(0, EEG_MAX_FREQUENCY_HZ)
eeg_ax.set_ylabel("EEG\nfrequency (Hz)")
eeg_ax.tick_params(axis="x", labelbottom=False)
viz.mark_acquisition_boundaries(eeg_ax, timeline_view)
colorbar = fig.colorbar(spectrogram_image, cax=colorbar_ax)
colorbar.set_label("EEG power\n(dB mV²/Hz)", fontsize=8)
colorbar.ax.tick_params(labelsize=7)

for panel in emg_panels:
    emg_ax.fill_between(
        panel.time_min,
        -panel.amplitude,
        panel.amplitude,
        color="0.08",
        linewidth=0,
        rasterized=True,
    )
emg_ax.axhline(0, color="0.35", lw=0.5)
emg_ax.set_ylim(-emg_limit, emg_limit)
emg_ax.set_ylabel(f"EMG RMS\n({physiology.emg_unit})")
emg_ax.tick_params(axis="x", labelbottom=False)
emg_ax.text(
    0.995,
    0.92,
    "20–200 Hz · 250-ms RMS",
    transform=emg_ax.transAxes,
    ha="right",
    va="top",
    fontsize=7,
    color="0.25",
)
viz.mark_acquisition_boundaries(emg_ax, timeline_view)

viz.plot_state_strip(
    state_ax,
    timeline_view,
    state_labels=state_labels,
    short_labels=state_short_labels,
)

physiology_path = FIG_DIR / "01_eeg_emg_state_classification.png"
fig.savefig(physiology_path, dpi=160, bbox_inches="tight")
plt.show()
print("saved ->", physiology_path)

# %% [markdown]
# ## Takeaway
#
# We have one complete, spatially resolved neural population plus independently
# recorded physiology and state labels. Grouping the event raster by atlas region
# makes regional sampling explicit, while the population trace shows how widely
# activity rises and falls across the full session. Scripts 02--06 use the same
# recording so every later analysis step can be traced back to this dataset.
