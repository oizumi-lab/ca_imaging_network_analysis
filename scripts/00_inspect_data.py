# %% [markdown]
# # 00 · Inspecting the dataset
#
# This is an interactive ``# %%`` script (run it cell-by-cell in VS Code or
# Spyder). It first summarizes every full recording, then opens one complete
# example session and shows every neuron. The aim is to understand the raw
# material before doing any network analysis.
#
# Dataset: *Single-cell calcium imaging across wakefulness, sleep, and anesthesia*
# (RIKEN 20260409-001 v2.0; Kiyooka & Oomoto et al., Cell Reports 2026).

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

from src.funcnet import dataio, visualization as viz
from src.funcnet.paths import FIG_DIR, RAW_DIR, RESULTS_DIR

FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
# The left panel shows spatial scale (number of recorded ROIs). The right panel
# shows the proportion of every raw label in ``state``. Sleep files can contain
# awake, quiet awake, NREM, and REM; anesthesia files contain awake and
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

fig, (ax_rois, ax_time) = plt.subplots(
    1,
    2,
    figsize=(13, 6.8),
    sharey=True,
    gridspec_kw={"width_ratios": [0.9, 1.4]},
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

left = np.zeros(len(inventory))
for label, display_name in state_display:
    values = inventory[f"{label}_pct"].to_numpy()
    ax_time.barh(
        y,
        values,
        left=left,
        color=viz.DEFAULT_STATE_COLORS[label],
        label=display_name,
    )
    for row, (start, width) in enumerate(zip(left, values)):
        if width >= 5:
            text_color = "black" if label in {"quiet_awake", "anesthesia"} else "white"
            ax_time.text(
                start + width / 2,
                row,
                f"{width:.0f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
            )
    left += values
ax_time.set_xlabel("proportion of recorded frames (%)")
ax_time.set_title("Raw brain-state labels")
ax_time.set_xlim(0, 100)
ax_time.grid(axis="x", alpha=0.2)
ax_time.legend(
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
# ## Inspect one example recording in detail
#
# Now that we know the scale of the complete dataset, load the full
# ``mouse01_sleep`` session. It contains 7,843 neurons. Nothing below applies the
# ``nonzero_ROI`` mask or samples neuron rows. Swap in another full recording
# name, e.g. ``"mouse07_ane"``, to inspect a different session.

# %%
rec = dataio.load_recording("mouse01_sleep")
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
# ## Look at every neuron's activity
#
# A line plot cannot show 7,843 traces legibly. Instead, the top panel is an
# all-neuron raster built from positive OASIS deconvolution samples. Every neuron
# has one row; only time is grouped into approximately one-second display bins.
# Rows are ranked by total activity without using state labels. The binning is
# only for visualization and restarts at state changes and acquisition breaks.
# The bottom panel shows the original frame-by-frame brain-state timeline.

# %%
t = np.arange(rec.n_frames) / rec.fs / 60  # minutes
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

fig, axes = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[4, 1], sharex=True)

axes[0].scatter(
    bin_centers_min[occupied_bins],
    neuron_rows,
    s=0.08,
    color="black",
    marker=".",
    linewidths=0,
    rasterized=True,
)
axes[0].set_ylim(rec.n_neurons - 0.5, -0.5)
axes[0].set_ylabel(f"all {rec.n_neurons:,} neurons\n(activity-ranked)")
axes[0].set_title(
    f"{rec.name} · every neuron · positive deconvolution samples "
    f"in ~{bin_frames / rec.fs:.1f}-s display bins"
)

axes[1].plot(t, rec.state, lw=0.8, color="k")
axes[1].set_yticks(sorted(codes))
axes[1].set_yticklabels([codes[v] for v in sorted(codes)])
axes[1].set_xlabel("time (min)")
axes[1].set_ylabel("state")
fig.tight_layout()

fig.savefig(FIG_DIR / "00_inspect_all_neurons.png", dpi=160)
plt.show()
print("saved ->", FIG_DIR / "00_inspect_all_neurons.png")

# %% [markdown]
# ## Spatial layout of the neurons
# Each neuron has an (x, y) centroid in the imaging field (3 mm × 3 mm). This is
# what we will later colour by functional module.

# %%
um = rec.centroid_um
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(um[:, 0], um[:, 1], s=6, alpha=0.6)
ax.set_aspect("equal")
ax.invert_yaxis()  # image convention
ax.set_xlabel("x (µm)")
ax.set_ylabel("y (µm)")
ax.set_title(f"All neuron positions (N = {rec.n_neurons:,})")
fig.tight_layout()
fig.savefig(FIG_DIR / "00_inspect_positions.png", dpi=130)
plt.show()
print("saved ->", FIG_DIR / "00_inspect_positions.png")
