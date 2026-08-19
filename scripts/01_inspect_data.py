# %% [markdown]
# # 01 · Inspect one complete calcium-imaging and EEG/EMG recording
#
# ## Where this script fits
# The hands-on follows one recording, ``mouse02_sleep``, from the measured
# signals to a network result:
#
# ```text
# inspect the recording → build a network → find modules → compare brain states
# ```
#
# Before calculating anything, we need to know what was measured. A result can
# be misleading if, for example, a recording contains too little Awake data, the
# neural traces contain long gaps, or the state labels disagree with the EEG and
# EMG. This script therefore answers four basic questions:
#
# 1. How many neurons and time points were recorded?
# 2. What does activity from individual neurons look like?
# 3. Which cortical areas were sampled?
# 4. Do the EEG and EMG patterns make the deposited sleep-state labels plausible?
#
# Scripts 01--07 use this same recording so you can trace every later network
# quantity back to the data inspected here.

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
# ## Settings

# %%
RECORDING = "mouse02_sleep"
TRACE_NEURONS = 100
TRACE_SEED = 7
DISPLAY_BIN_SECONDS = 1.0
EEG_MAX_FREQUENCY_HZ = 25.0
EEG_WINDOW_SECONDS = 4.0

# %% [markdown]
# ## Load and summarize the complete recording
#
# ``state`` contains the deposited behavioral labels. ``used_frame`` supplies
# stable state epochs used in the network analysis, while ``nonzero_ROI`` marks
# neurons that passed the publication's activity criterion.

# %%
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

rng = np.random.default_rng(TRACE_SEED)
trace_rows = np.sort(rng.choice(rec.n_neurons, size=TRACE_NEURONS, replace=False))
traces = rec.dFF[trace_rows]
centered = traces - np.nanmedian(traces, axis=1, keepdims=True)
q01, q99 = np.nanpercentile(centered, (1, 99))
spacing = 1.15 * max(float(q99 - q01), 1e-6)
offsets = -np.arange(TRACE_NEURONS) * spacing
time_min = np.arange(rec.n_frames) / rec.fs / 60
acquisition_segments = ts.acquisition_segments(rec.n_frames, rec.boundary_ind)

fig = plt.figure(figsize=(15, 10), constrained_layout=True)
grid = fig.add_gridspec(2, 1, height_ratios=(9, 0.48))
trace_ax = fig.add_subplot(grid[0])
state_ax = fig.add_subplot(grid[1], sharex=trace_ax)
viz.shade_states(trace_ax, timeline_view)
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
# subset.

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
# ## Figure 3 — raster, EEG spectrogram, EMG, and deposited state labels
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

# %%
raster, _, bin_frames, _, bin_centers = viz.binned_spike_raster(
    rec.spike_deconv,
    rec.fs,
    DISPLAY_BIN_SECONDS,
    rec.boundary_ind,
    rec.state,
)
neuron_rows, occupied_bins = np.nonzero(raster)
bin_centers_min = bin_centers / rec.fs / 60

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

state_labels = dict(viz.DEFAULT_STATE_LABELS)
state_labels.update({"awake": "Wake", "quiet_awake": "Quiet awake", "nrem": "NREM", "rem": "REM"})
state_short_labels = dict(viz.DEFAULT_STATE_SHORT_LABELS)
state_short_labels.update({"awake": "W", "quiet_awake": "Q", "nrem": "N", "rem": "R"})
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
# recorded physiology and state labels. Scripts 02--07 use the same recording so
# every analysis step can be traced back to this concrete dataset.

# %% [markdown]
# ## Exercise 1 — compare basic activity between states (easy)
#
# Before constructing a network, ask a simpler question: is the *average amount*
# of deconvolved activity different between Awake and NREM?
#
# Write a short new cell that uses the first 1,500 stable frames from each state
# and reports:
#
# 1. the mean ``spike_smoothed`` value; and
# 2. the fraction of ``spike_smoothed`` samples greater than zero.
#
# Display the two states in a small table or bar plot. This is a descriptive
# activity comparison, not a network result.
#
# **Where to look for help:** the state-summary loop near the beginning shows how
# to obtain frames with ``dataio.state_frames``. The plotting sections show the
# basic Matplotlib pattern. Try writing this exercise without AI.
