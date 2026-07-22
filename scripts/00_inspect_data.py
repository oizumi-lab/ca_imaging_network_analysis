# %% [markdown]
# # 00 · Inspecting the dataset
#
# This is an interactive ``# %%`` script (run it cell-by-cell in VS Code or
# Spyder). It first summarizes every full recording, then opens one complete
# example session and shows every neuron. The aim is to understand the raw
# material before doing any network analysis.
#
# Dataset: *Single-cell calcium imaging across wakefulness, sleep, and anesthesia*
# (processed calcium: RIKEN 20260409-001 v2.0; synchronized EEG/EMG extension:
# RIKEN 20260708-001 v3.0; Kiyooka & Oomoto et al., Cell Reports 2026).

# %%
import os
import sys

# add the repo root (parent of scripts/) to the path so `src.funcnet` is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.funcnet import dataio, physiology as physio, timeseries as ts, visualization as viz
from src.funcnet.paths import FIG_DIR, RAW_DIR, RESULTS_DIR

FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Settings
#
# This inspection script intentionally keeps its few user choices together:
#
# - `DETAILED_RECORDING` chooses the complete session used by the activity and
#   spatial examples after the ten-session inventory.
# - `COMPARISON_NEURON_COUNT` controls the row-aligned raw ΔF/F versus smoothed
#   deconvolved-signal comparison. Ten neurons keeps both panels readable.
# - `COMPARISON_TIME_RANGE_MIN` sets its recorded-time start and end in minutes.
#   The default shows minutes 0–5; use `None` to restore the complete session.
# - `DFF_NEURON_COUNT` controls the separate full-session raw ΔF/F example.
#   Its default is 100 neurons; change this one scalar to show another count.
# - `TRACE_RANDOM_SEED` makes both random selections exactly reproducible without
#   favoring low-numbered ROI rows.
# - `EEG_MAX_FREQUENCY_HZ` and `EEG_WINDOW_SECONDS` control only the synchronized
#   inspection spectrogram. EEG and EMG are prepared independently inside every
#   acquisition segment, so filtering never crosses a microscope break.

# %%
DETAILED_RECORDING = "mouse01_sleep"
# DETAILED_RECORDING = "mouse03_ane"
COMPARISON_NEURON_COUNT = 10
COMPARISON_TIME_RANGE_MIN = (0.0, 5.0)  # (start_min, end_min), or None
DFF_NEURON_COUNT = 100
TRACE_RANDOM_SEED = 7
EEG_MAX_FREQUENCY_HZ = 25.0
EEG_WINDOW_SECONDS = 4.0

# %% [markdown]
# ## Start with all full recordings
#
# Before choosing one example, ask: **how large is the complete dataset, and how
# are the recorded frame labels distributed in each session?** The inventory
# below uses the raw ``state`` vector from each of the ten original recording
# files. It does not apply ``used_frame`` or another analysis-specific selection.
# ``example_data.mat`` is deliberately excluded because it is a 1,000-neuron
# subset of ``mouse01_sleep``, not an independent experiment.
#
# Loading every activity matrix just to learn its dimensions would be wasteful:
# the complete download is about 11 GB, and each file contains three large
# floating-point matrices. MATLAB v7.3 files are HDF5 containers, so ``h5py``
# can read dataset shapes and small metadata without loading the activity values.

# %%
def _decode_matlab_text(dataset: h5py.Dataset) -> str:
    """Decode a MATLAB character array stored as UTF-16 integer codes."""
    codes = np.asarray(dataset).ravel(order="F")
    return "".join(chr(int(code)) for code in codes if int(code)).strip()


def inspect_recording_metadata(recording_name: str) -> dict[str, object]:
    """Read session dimensions and raw state counts without activity arrays."""
    path = RAW_DIR / f"{recording_name}.mat"
    with h5py.File(path, "r") as mat:
        data_info = _decode_matlab_text(mat["data_info"])
        if data_info not in {"sleep", "ane"}:
            raise ValueError(f"Unexpected data_info={data_info!r} in {path.name}")

        # Semantic one-dimensional variables give unambiguous N and T. Direct
        # HDF5 activity shapes are usually (T, N), whereas dataio.load_recording
        # presents them to Python as the more convenient (N, T).
        n_rois = int(mat["nonzero_ROI"].size)
        state = np.asarray(mat["state"]).ravel(order="F").astype(float, copy=False)
        n_frames = int(state.size)
        valid_activity_shapes = {(n_frames, n_rois), (n_rois, n_frames)}
        for signal in ("dFF", "spike_deconv", "spike_smoothed"):
            if mat[signal].shape not in valid_activity_shapes:
                raise ValueError(
                    f"{path.name}:{signal} has shape {mat[signal].shape}; "
                    f"expected N={n_rois}, T={n_frames}"
                )
        if mat["ROIs/Centroid"].size != 2 * n_rois:
            raise ValueError(f"{path.name}: ROI centroids do not match ROI count")

        code_map = (
            dataio.SLEEP_STATE_CODES if data_info == "sleep" else dataio.ANE_STATE_CODES
        )
        unknown_codes = np.setdiff1d(np.unique(state), list(code_map))
        if unknown_codes.size:
            raise ValueError(
                f"{path.name}: unrecognized state codes {unknown_codes.tolist()}"
            )
        state_counts = {
            label: int(np.count_nonzero(np.isclose(state, code)))
            for code, label in code_map.items()
        }

    session = (
        recording_name.replace("day1", "day 1")
        .replace("day2", "day 2")
        .replace("_ane", "_anesthesia")
        .replace("_", " ")
    )
    row = {
        "recording": recording_name,
        "session": session,
        "animal": recording_name.split("_")[0],
        "paradigm": "Sleep" if data_info == "sleep" else "Anesthesia",
        "rois": n_rois,
        "frames": n_frames,
        "minutes": n_frames / dataio.FS_HZ / 60,
    }
    for label in ("awake", "quiet_awake", "nrem", "rem", "anesthesia"):
        count = state_counts.get(label, 0)
        row[f"{label}_frames"] = count
        row[f"{label}_pct"] = 100 * count / n_frames
    return row


# This tutorial targets the ten independent sessions in dataset v2.0. Listing
# them explicitly makes a partial download fail loudly instead of being mistaken
# for the complete cohort. ``example_data`` is intentionally absent.
EXPECTED_RECORDINGS = (
    "mouse01_sleep",
    "mouse02_sleep",
    "mouse03_sleep",
    "mouse04_day1_sleep",
    "mouse04_day2_sleep",
    "mouse05_sleep",
    "mouse03_ane",
    "mouse05_ane",
    "mouse06_ane",
    "mouse07_ane",
)
available_recordings = set(dataio.list_recordings())
missing_recordings = [
    name for name in EXPECTED_RECORDINGS if name not in available_recordings
]
if missing_recordings:
    raise FileNotFoundError(
        "The complete ten-session inventory requires the full dataset. "
        f"Missing: {', '.join(missing_recordings)}. Run "
        "`poetry run python scripts/download_data.py`, then rerun this cell."
    )
recording_names = list(EXPECTED_RECORDINGS)

inventory = pd.DataFrame(inspect_recording_metadata(name) for name in recording_names)
inventory["_paradigm_order"] = inventory["paradigm"].map(
    {"Sleep": 0, "Anesthesia": 1}
)
inventory = (
    inventory.sort_values(["_paradigm_order", "recording"])
    .drop(columns="_paradigm_order")
    .reset_index(drop=True)
)

# Basic consistency checks turn missing or double-counted labels into errors.
state_labels = ("awake", "quiet_awake", "nrem", "rem", "anesthesia")
state_frame_columns = [f"{label}_frames" for label in state_labels]
state_pct_columns = [f"{label}_pct" for label in state_labels]
if not inventory[state_frame_columns].sum(axis=1).eq(inventory["frames"]).all():
    raise ValueError("Raw state counts do not sum to the recording frame count")
if not np.allclose(inventory[state_pct_columns].sum(axis=1), 100):
    raise ValueError("Raw state proportions do not sum to 100 percent")

# Keep numeric columns numeric in the CSV; make a formatted copy only for print.
csv_path = RESULTS_DIR / "00_dataset_inventory.csv"
inventory.to_csv(csv_path, index=False, float_format="%.3f")

display_table = pd.DataFrame(
    {
        "Session": inventory["session"],
        "ROIs": inventory["rois"].map(lambda value: f"{value:,}"),
        "Frames": inventory["frames"].map(lambda value: f"{value:,}"),
        "Minutes": inventory["minutes"].map(lambda value: f"{value:.1f}"),
        "Awake": inventory["awake_pct"].map(lambda value: f"{value:.1f}%"),
        "Quiet awake": [
            f"{value:.1f}%" if paradigm == "Sleep" else "—"
            for value, paradigm in zip(
                inventory["quiet_awake_pct"], inventory["paradigm"]
            )
        ],
        "NREM": [
            f"{value:.1f}%" if paradigm == "Sleep" else "—"
            for value, paradigm in zip(inventory["nrem_pct"], inventory["paradigm"])
        ],
        "REM": [
            f"{value:.1f}%" if paradigm == "Sleep" else "—"
            for value, paradigm in zip(inventory["rem_pct"], inventory["paradigm"])
        ],
        "Anesthesia": [
            f"{value:.1f}%" if paradigm == "Anesthesia" else "—"
            for value, paradigm in zip(
                inventory["anesthesia_pct"], inventory["paradigm"]
            )
        ],
    }
)
print(display_table.to_string(index=False))
print("\nsaved numeric inventory ->", csv_path)

cohorts = inventory.groupby("paradigm", sort=False).agg(
    sessions=("recording", "size"),
    unique_mice=("animal", "nunique"),
)
print("\nExperimental units (sessions are not always independent mice):")
print(cohorts.to_string())

# %% [markdown]
# ### Dataset at a glance
#
# The panels show spatial scale (number of recorded ROIs), recording duration in
# minutes, and the proportion of every raw label in ``state``. Sleep files can
# contain awake, quiet awake, NREM, and REM; anesthesia files contain awake and
# anesthesia. Every recorded frame contributes exactly once. No paper-specific
# frame-selection mask is applied.

# %%
y = np.arange(len(inventory))
roi_colors = inventory["paradigm"].map(
    {"Sleep": "#2a9d8f", "Anesthesia": "#7b2cbf"}
)
state_display = (
    ("awake", "Awake"),
    ("quiet_awake", "Quiet awake"),
    ("nrem", "NREM"),
    ("rem", "REM"),
    ("anesthesia", "Anesthesia"),
)

fig, (ax_rois, ax_duration, ax_state) = plt.subplots(
    1,
    3,
    figsize=(16.5, 6.8),
    sharey=True,
    gridspec_kw={"width_ratios": [0.9, 0.9, 1.4]},
)

roi_bars = ax_rois.barh(y, inventory["rois"] / 1000, color=roi_colors)
ax_rois.bar_label(
    roi_bars,
    labels=[f"{value / 1000:.1f}k" for value in inventory["rois"]],
    padding=3,
    fontsize=8,
)
ax_rois.set_yticks(y, inventory["session"])
ax_rois.invert_yaxis()
ax_rois.set_xlabel("recorded ROIs (thousands)")
ax_rois.set_title("Population size")
ax_rois.set_xlim(0, inventory["rois"].max() / 1000 * 1.18)
ax_rois.grid(axis="x", alpha=0.2)
ax_rois.legend(
    handles=[
        Patch(color="#2a9d8f", label="sleep session"),
        Patch(color="#7b2cbf", label="anesthesia session"),
    ],
    loc="lower right",
    frameon=False,
)

duration_bars = ax_duration.barh(y, inventory["minutes"], color=roi_colors)
ax_duration.bar_label(
    duration_bars,
    labels=[f"{value:.1f}" for value in inventory["minutes"]],
    padding=3,
    fontsize=8,
)
ax_duration.set_xlabel("duration (min)")
ax_duration.set_title("Recording duration")
ax_duration.set_xlim(0, inventory["minutes"].max() * 1.18)
ax_duration.grid(axis="x", alpha=0.2)

left = np.zeros(len(inventory))
for label, display_name in state_display:
    values = inventory[f"{label}_pct"].to_numpy()
    ax_state.barh(
        y,
        values,
        left=left,
        color=viz.DEFAULT_STATE_COLORS[label],
        label=display_name,
    )
    for row, (start, width) in enumerate(zip(left, values)):
        if width >= 5:
            text_color = "black" if label in {"quiet_awake", "anesthesia"} else "white"
            ax_state.text(
                start + width / 2,
                row,
                f"{width:.0f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
            )
    left += values
ax_state.set_xlabel("proportion of recorded frames (%)")
ax_state.set_title("Raw brain-state labels")
ax_state.set_xlim(0, 100)
ax_state.grid(axis="x", alpha=0.2)
ax_state.legend(
    handles=[
        Patch(color=viz.DEFAULT_STATE_COLORS[label], label=display_name)
        for label, display_name in state_display
    ],
    loc="center left",
    bbox_to_anchor=(1.01, 0.5),
    frameon=False,
)

fig.suptitle(f"Complete calcium-imaging dataset: {len(inventory)} recording sessions")
fig.tight_layout()
summary_figure_path = FIG_DIR / "00_dataset_inventory.png"
fig.savefig(summary_figure_path, dpi=160, bbox_inches="tight")
plt.show()
print("saved ->", summary_figure_path)

# %% [markdown]
# ### Which cortical areas occur in each recording?
#
# Atlas coverage varies with the imaging field. This lightweight inventory reads
# only the row-aligned atlas strings and activity mask from each HDF5 file. It
# reports exact layer-collapsed areas for all recorded ROIs and for the optional
# publication ``nonzero_ROI`` subset. Every current recording contains at least
# one visual area; the less common visual and somatosensory subdivisions occur
# only in particular sessions. Source label ``root`` is the top of the Allen
# ontology rather than a cortical area, so figures call it ``Unassigned`` while
# retaining its count. Colors are organized by family: motor greens,
# somatosensory warm colors, retrosplenial blues, and visual purples.

# %%
def inspect_recording_atlas(recording_name: str) -> list[dict[str, object]]:
    """Return exact displayed-area counts without loading activity matrices."""
    path = RAW_DIR / f"{recording_name}.mat"
    atlas_labels = dataio.load_atlas_labels(path)
    if atlas_labels is None:
        raise ValueError(f"{path.name}: no row-aligned atlas labels were found")

    with h5py.File(path, "r") as mat:
        active_mask = (
            np.asarray(mat["nonzero_ROI"])
            .ravel(order="F")
            .astype(bool, copy=False)
        )
    source_regions = np.asarray(
        [str(label).removesuffix("2/3") for label in atlas_labels],
        dtype=object,
    )
    display_regions = viz.brain_region_labels(atlas_labels)
    if active_mask.shape != display_regions.shape:
        raise ValueError(f"{path.name}: atlas and nonzero_ROI rows do not align")

    paradigm = "Sleep" if recording_name.endswith("sleep") else "Anesthesia"
    rows = []
    for region in viz.BRAIN_REGION_COLORS:
        selected = display_regions == region
        if not np.any(selected):
            continue
        source_labels = ", ".join(sorted(set(source_regions[selected].tolist())))
        recorded = int(np.count_nonzero(selected))
        analysis = int(np.count_nonzero(selected & active_mask))
        rows.append(
            {
                "recording": recording_name,
                "paradigm": paradigm,
                "family": viz.BRAIN_REGION_FAMILIES[region],
                "display_region": region,
                "source_labels": source_labels,
                "recorded_rois": recorded,
                "nonzero_roi_rois": analysis,
                "excluded_rois": recorded - analysis,
            }
        )
    return rows


atlas_inventory = pd.DataFrame(
    row
    for recording_name in recording_names
    for row in inspect_recording_atlas(recording_name)
)
atlas_inventory_path = RESULTS_DIR / "00_atlas_region_inventory.csv"
atlas_inventory.to_csv(atlas_inventory_path, index=False)

atlas_count_table = atlas_inventory.pivot(
    index="recording",
    columns="display_region",
    values="recorded_rois",
).fillna(0).astype(int)
ordered_present_regions = [
    region for region in viz.BRAIN_REGION_COLORS if region in atlas_count_table
]
atlas_count_table = atlas_count_table.reindex(
    index=recording_names,
    columns=ordered_present_regions,
)
print("\nExact cortical-area counts by recording (all recorded ROIs):")
print(atlas_count_table.to_string())
print("\nsaved atlas inventory with nonzero_ROI counts ->", atlas_inventory_path)

# %% [markdown]
# ## Inspect one example recording in detail
#
# Now that we know the scale of the complete dataset, load the full session
# selected by `DETAILED_RECORDING`. Nothing below applies the ``nonzero_ROI``
# mask. The two stacked-trace displays sample rows only for readability; later
# panels explicitly show every neuron. Swap in another full recording name,
# e.g. ``"mouse07_ane"``, in the settings to inspect a different session.

# %%
rec = dataio.load_recording(DETAILED_RECORDING)
print(rec)
print("Available recordings:", dataio.list_recordings(include_example=True))

# %% [markdown]
# ## What is inside a Recording?
# All time-series are ``(N_neurons, T_frames)`` and **row-aligned** — row *i* is
# the same neuron everywhere. Sampling rate is 7.65 Hz.

# %%
print(f"data_info        : {rec.data_info!r}  (states compared: {rec.state_labels})")
print(f"n_neurons (N)    : {rec.n_neurons}")
print(f"n_frames  (T)    : {rec.n_frames}  (~{rec.n_frames / rec.fs / 60:.1f} min)")
print(f"dFF              : {rec.dFF.shape}  raw fluorescence dF/F")
print(f"spike_deconv     : {rec.spike_deconv.shape}  OASIS-deconvolved spikes")
print(f"spike_smoothed   : {rec.spike_smoothed.shape}  Gaussian-smoothed event proxy")
print(f"centroid (px)    : {rec.centroid.shape}  (x, y)")
if rec.atlas is not None:
    print(
        f"atlas labels     : {len(rec.atlas)} row-aligned acronyms "
        f"({len(set(rec.atlas))} exact labels)"
    )
print(f"animal_info      : {rec.animal_info}")
if rec.nonzero_ROI is not None:
    print(f"nonzero_ROI      : {int(rec.nonzero_ROI.sum())}/{rec.n_neurons} neurons "
          f"in the optional paper-specific analysis mask")

# %% [markdown]
# ## The brain-state vector
# ``state`` labels every frame. For sleep recordings: 0 = awake, 0.5 = quiet
# awake, 1 = NREM, 2 = REM. For anesthesia: 0 = awake, 1 = anesthesia. We show
# all recorded labels here without applying an analysis-specific frame mask.

# %%
codes = dataio.state_codes(rec)
vals, counts = np.unique(rec.state, return_counts=True)
for v, c in zip(vals, counts):
    print(f"  state {v:>4}  = {codes.get(v, '?'):<12}  {c:6d} frames ({100*c/rec.n_frames:.1f}%)")

# %% [markdown]
# ## Raw ΔF/F versus the network-analysis signal
#
# The top and bottom panels below contain the **same randomly selected neuron
# rows in the same order** over the selected recorded-time window. The top shows
# raw 7.65-Hz ΔF/F. The bottom shows ``spike_smoothed``: the OASIS-deconvolved
# estimate after the dataset's 15-frame (1.96-s) Gaussian smoothing. This
# smoothed deconvolution-derived matrix, rather than the native unsmoothed
# ``spike_deconv`` matrix, is the input used by Kiyooka et al.'s network
# analyses. These are event proxies, not direct electrical spike measurements.
#
# Both signals are only median-centered and vertically offset for display. They
# keep their original time samples and amplitudes, and each signal has its own
# vertical spacing and scale bar because their units differ. Change
# `COMPARISON_NEURON_COUNT` or `COMPARISON_TIME_RANGE_MIN` in the settings to
# choose another number of neurons or another start/end time. The default window
# is the first five minutes.


# %%
def trace_activity_view(
    recording,
    neuron_ids: np.ndarray,
    time_range_min: tuple[float, float] | None = None,
) -> dict[str, object]:
    """Build the compact timing/state mapping used by stacked-trace plots."""
    duration_min = recording.n_frames / recording.fs / 60
    boundary_minutes = [
        (int(boundary) + 1) / recording.fs / 60
        for boundary in np.asarray(recording.boundary_ind).ravel()
        if 0 <= int(boundary) < recording.n_frames - 1
    ]
    return {
        "name": recording.name,
        "data_info": recording.data_info,
        "n_neurons": recording.n_neurons,
        "n_frames": recording.n_frames,
        "fs": recording.fs,
        "duration_min": duration_min,
        "time_limits_min": viz.resolve_time_limits(duration_min, time_range_min),
        "state": recording.state,
        "codes": dict(dataio.state_codes(recording)),
        "trace_ids": neuron_ids,
        "dff": recording.dFF[neuron_ids],
        "boundary_minutes": boundary_minutes,
        "acquisition_segments": ts.acquisition_segments(
            recording.n_frames,
            recording.boundary_ind,
        ),
    }


def make_signal_comparison_figure(
    recording,
    neuron_ids: np.ndarray,
    time_range_min: tuple[float, float] | None = None,
):
    """Plot matching raw and network-input traces for selected neurons."""
    view = trace_activity_view(recording, neuron_ids, time_range_min)
    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=(5, 5, 0.45))
    dff_ax = fig.add_subplot(grid[0])
    spike_ax = fig.add_subplot(grid[1], sharex=dff_ax)
    state_ax = fig.add_subplot(grid[2], sharex=dff_ax)

    viz.plot_stacked_dff(
        dff_ax,
        view,
        line_width=0.45,
        annotate_boundaries=False,
    )
    dff_ax.set_title("Top: raw fluorescence (ΔF/F; dFF)")

    viz.plot_stacked_signal(
        spike_ax,
        view,
        recording.spike_smoothed[neuron_ids],
        signal_label="smoothed deconvolved activity",
        scale_unit="deconvolution units",
        line_width=0.45,
        color="#8b1a1a",
    )
    spike_ax.set_title(
        "Bottom: smoothed OASIS-deconvolved signal "
        "(spike_smoothed; network-analysis input)"
    )
    viz.plot_state_strip(state_ax, view)
    fig.suptitle(
        f"{recording.name} · same random {neuron_ids.size} of "
        f"{recording.n_neurons:,} neurons · {viz.time_window_label(view)}",
        fontsize=15,
    )
    return fig


comparison_neuron_ids = viz.select_trace_neurons(
    rec.n_neurons,
    COMPARISON_NEURON_COUNT,
    TRACE_RANDOM_SEED,
)
comparison_figure = make_signal_comparison_figure(
    rec,
    comparison_neuron_ids,
    COMPARISON_TIME_RANGE_MIN,
)
comparison_path = FIG_DIR / "00_inspect_signal_comparison.png"
comparison_figure.savefig(comparison_path, dpi=160, bbox_inches="tight")
plt.show()
print("saved ->", comparison_path)

# %% [markdown]
# ## One configurable raw ΔF/F display
#
# This separate figure shows one random sample. The default is 100 neurons.
# Change `DFF_NEURON_COUNT` in the settings to choose another count. As above,
# the complete raw time sequence is retained: no smoothing, temporal
# downsampling, clipping, or z-scoring is applied, and the state strip stays
# aligned to the original frame labels.


# %%
def make_dff_figure(recording, neuron_ids: np.ndarray):
    """Plot one complete raw ΔF/F example at a fixed 15 × 10 inch size."""
    view = trace_activity_view(recording, neuron_ids)
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=(9, 0.48))
    trace_ax = fig.add_subplot(grid[0])
    state_ax = fig.add_subplot(grid[1], sharex=trace_ax)
    line_width = max(0.10, 0.35 * np.sqrt(100 / neuron_ids.size))
    viz.plot_stacked_dff(trace_ax, view, line_width=line_width)
    viz.plot_state_strip(state_ax, view)
    fig.suptitle(
        f"Full-session raw ΔF/F example ({neuron_ids.size} random neurons)",
        fontsize=15,
    )
    return fig


dff_neuron_ids = viz.select_trace_neurons(
    rec.n_neurons,
    DFF_NEURON_COUNT,
    TRACE_RANDOM_SEED,
)
dff_figure = make_dff_figure(rec, dff_neuron_ids)
dff_path = FIG_DIR / "00_inspect_traces.png"
dff_figure.savefig(dff_path, dpi=160, bbox_inches="tight")
plt.show()
print("saved ->", dff_path)

# %% [markdown]
# ## Align every neuron with EEG, EMG, and brain state
#
# A line plot cannot show 7,843 traces legibly. Instead, the top panel is an
# all-neuron raster built from positive OASIS deconvolution samples. Every neuron
# has one row; only time is grouped into approximately one-second display bins.
# Rows are ranked by total activity without using state labels. The binning is
# only for visualization and restarts at state changes and acquisition breaks.
#
# The new v3.0 physiological files are sampled at 5 kHz and contain calibration,
# recovery, and acquisition-gap intervals that are absent from the processed
# calcium matrices. Therefore, total duration is not a valid synchronization
# anchor. ``align_frame_triggers`` detects the deposited two-photon trigger
# bouts and applies the audited per-recording v3.0 mapping to the calcium
# acquisition segments. Physiological sample times are then warped through the
# selected frame triggers onto the same gap-free recorded-time axis as the
# raster and categorical state strip.
#
# The EEG panel shows 0.5–25 Hz power after segment-local anti-aliasing and
# band-pass filtering. The EMG panel shows a segment-local 20–200 Hz, 250-ms RMS
# envelope; its robust display limit prevents isolated amplifier saturation from
# flattening the remaining muscle activity. These operations are display-only.

# %%
display_bin_seconds = 1.0
raster, _, bin_frames, _, bin_centers = viz.binned_spike_raster(
    rec.spike_deconv,
    rec.fs,
    display_bin_seconds,
    rec.boundary_ind,
    rec.state,
)
neuron_rows, occupied_bins = np.nonzero(raster)
bin_centers_min = bin_centers / rec.fs / 60

physiology = physio.load_physiology(rec.name)
physiology_alignment = physio.align_frame_triggers(
    physiology,
    n_frames=rec.n_frames,
    boundary_ind=rec.boundary_ind,
    imaging_fs=rec.fs,
)
eeg_panels, eeg_color_limits = physio.prepare_eeg_spectrogram(
    physiology,
    physiology_alignment,
    imaging_fs=rec.fs,
    max_frequency_hz=EEG_MAX_FREQUENCY_HZ,
    window_seconds=EEG_WINDOW_SECONDS,
)
emg_panels, emg_amplitude_limit = physio.prepare_emg_envelope(
    physiology,
    physiology_alignment,
    imaging_fs=rec.fs,
)

print(
    f"Physiology       : {physiology.n_samples:,} samples at "
    f"{physiology.fs:g} Hz from {len(physiology.source_paths)} file(s)"
)
for segment_number, segment in enumerate(physiology_alignment.segments, start=1):
    selected_stop = segment.trigger_start_offset + segment.n_frames
    print(
        f"  segment {segment_number}: calcium frames "
        f"[{segment.frame_start:,}, {segment.frame_stop:,}) <- trigger bout "
        f"{segment.trigger_group_index}[{segment.trigger_start_offset:,}:"
        f"{selected_stop:,}] of {segment.trigger_group_size:,} rises"
    )
print(
    f"  mapped {physiology_alignment.mapped_trigger_count:,} / "
    f"{physiology_alignment.detected_trigger_count:,} detected rises; "
    f"ignored {physiology_alignment.ignored_trigger_count:,} "
    "calibration/recovery/surplus rises"
)

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
eeg_colorbar_ax = fig.add_subplot(grid[1, 1])

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
    f"{rec.name} · every neuron · positive deconvolution samples "
    f"in ~{bin_frames / rec.fs:.1f}-s display bins\n"
    "frame-trigger-synchronized EEG, EMG, and behavioral state"
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
        vmin=eeg_color_limits[0],
        vmax=eeg_color_limits[1],
        rasterized=True,
    )
if spectrogram_image is None:
    raise RuntimeError("No aligned EEG spectrogram panels were prepared")
eeg_ax.set_ylim(0, EEG_MAX_FREQUENCY_HZ)
eeg_ax.set_yticks(np.linspace(0, EEG_MAX_FREQUENCY_HZ, 6))
eeg_ax.set_ylabel("EEG\nfrequency (Hz)")
eeg_ax.tick_params(axis="x", labelbottom=False)
viz.mark_acquisition_boundaries(eeg_ax, timeline_view)
eeg_colorbar = fig.colorbar(spectrogram_image, cax=eeg_colorbar_ax)
eeg_colorbar.set_label("EEG power\n(dB mV²/Hz)", fontsize=8)
eeg_colorbar.ax.tick_params(labelsize=7)

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
emg_ax.set_ylim(-emg_amplitude_limit, emg_amplitude_limit)
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
state_labels.update(
    {
        "awake": "Wake",
        "quiet_awake": "Quiet awake",
        "nrem": "NREM",
        "rem": "REM",
    }
)
state_short_labels = dict(viz.DEFAULT_STATE_SHORT_LABELS)
state_short_labels.update(
    {
        "awake": "W",
        "quiet_awake": "Q",
        "nrem": "N",
        "rem": "R",
        "anesthesia": "A",
    }
)
viz.plot_state_strip(
    state_ax,
    timeline_view,
    state_labels=state_labels,
    short_labels=state_short_labels,
)

physiology_figure_path = FIG_DIR / "00_inspect_eeg_emg_raster.png"
legacy_raster_path = FIG_DIR / "00_inspect_all_neurons.png"
fig.savefig(physiology_figure_path, dpi=160, bbox_inches="tight")
fig.savefig(legacy_raster_path, dpi=160, bbox_inches="tight")
plt.show()
print("saved ->", physiology_figure_path)
print("saved compatibility copy ->", legacy_raster_path)

# %% [markdown]
# ## Spatial layout and cortical-region labels
# Each neuron has both an (x, y) centroid in the 3 mm × 3 mm imaging field and a
# row-aligned Allen-atlas acronym such as ``MOp2/3`` or ``SSp-bfd2/3``. The
# recording targeted layer 2/3, so the shared ``2/3`` suffix is removed. Every
# supplied area remains explicit, including visual and minor somatosensory
# subdivisions. The source ``root`` label becomes ``Unassigned`` because it is
# the Allen ontology root, not a specific cortical area. A missing label would
# instead appear as ``Unknown``. Motor areas use greens, somatosensory areas use
# warm colors, retrosplenial areas use blues, and visual areas use purples.
# Every recorded neuron is still plotted—``nonzero_ROI`` is not applied. The
# palette and label mapping live in the shared visualization module so the
# spatial map, region raster, and Rastermap row strip use the same encoding.

# %%
if rec.atlas is None:
    raise RuntimeError("This recording did not provide row-aligned atlas labels")
atlas = np.asarray(rec.atlas, dtype=str)
if atlas.shape != (rec.n_neurons,):
    raise ValueError("Atlas labels are not aligned one-to-one with neuron rows")

display_regions = viz.brain_region_labels(atlas)
present_regions = set(display_regions.tolist())
region_order = tuple(
    region for region in viz.BRAIN_REGION_COLORS if region in present_regions
)
print("Cortical-region counts (layer suffix collapsed):")
for region in region_order:
    print(f"  {region:<8} {np.count_nonzero(display_regions == region):5,d}")

um = rec.centroid_um
fig, ax = plt.subplots(figsize=(10.5, 6.2))

# Draw non-area/fallback groups first so named cortical areas remain crisp.
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
        um[selected, 0],
        um[selected, 1],
        s=5,
        color=viz.BRAIN_REGION_COLORS[region],
        alpha=0.88,
        linewidths=0,
        rasterized=True,
        zorder=1 if region in {"Unassigned", "Other", "Unknown"} else 2,
    )

ax.set_aspect("equal")
# Keep the dataset's Cartesian y direction: this places motor regions at the
# top and retrosplenial regions at the bottom, matching the reference map.
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(
    f"{rec.name} · all {rec.n_neurons:,} neurons by cortical region\n"
    "Allen atlas layer 2/3 labels"
)
legend_handles = viz.cortical_region_legend_handles(
    region_order,
    unabridged=True,
)
legend = ax.legend(
    handles=legend_handles,
    loc="center left",
    bbox_to_anchor=(1.01, 0.5),
    frameon=False,
    handletextpad=0.35,
    borderaxespad=0,
)
for text, region in zip(legend.get_texts(), region_order):
    text.set_color(viz.BRAIN_REGION_COLORS[region])
for spine in ax.spines.values():
    spine.set_color("0.35")
    spine.set_linewidth(0.9)
fig.tight_layout()
fig.savefig(FIG_DIR / "00_inspect_positions.png", dpi=130, bbox_inches="tight")
plt.show()
print("saved ->", FIG_DIR / "00_inspect_positions.png")
