# %% [markdown]
# # 03 · A basic Rastermap walkthrough
#
# This tutorial follows the essential workflow used in the Rastermap paper:
# select an active population, fit the official MouseLand implementation to the
# complete neuron-by-time matrix, sort neurons by the one-dimensional embedding,
# and average neighboring sorted neurons into readable "superneurons".
#
# Run one `# %%` cell at a time in VS Code or Spyder. The figures are displayed
# interactively and also saved in `results/figures`.
#
# References:
#
# - Stringer et al. (2025), Nature Neuroscience:
#   https://doi.org/10.1038/s41593-024-01783-4
# - Official implementation: https://github.com/MouseLand/rastermap

# %% Step 0 — imports
import gc
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
from rastermap import Rastermap

from src.funcnet import dataio, rastermap_tools as rmt, visualization as viz
from src.funcnet.paths import FIG_DIR

FIG_DIR.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Step 1 — settings
#
# Change `RECORDING_NAME` to analyze another sleep or anesthesia recording.
# Rastermap uses every selected neuron and every recorded frame; the detail
# window affects only the zoomed display.
#
# The lag is expressed in frames. Five frames in the paper's 3.2-Hz calcium
# recording span 1.56 s, so 12 frames are the closest match at 7.65 Hz.
# `SUPERNEURON_SIZE` controls only the display average, not the model fit.

# %%
RECORDING_NAME = "mouse01_sleep"

N_CLUSTERS = 100
N_PCS = 128
LOCALITY = 0.0
TIME_LAG_WINDOW = 12
TIME_BIN = 1
MEAN_TIME = True
SUPERNEURON_SIZE = 50
RANDOM_SEED = 0

DISPLAY_BIN_SECONDS = 1.0
DETAIL_WINDOW_MIN = (8.0, 12.0)
N_SANITY_ROWS = 100
SHOW_FIGURES = True

if TIME_BIN != 1:
    raise ValueError(
        "Keep TIME_BIN = 1 in this tutorial so model columns, frames, and "
        "state labels remain exactly aligned."
    )


# %% [markdown]
# ## Step 2 — load the complete recording and define the fit population
#
# The primary population is the intersection of:
#
# 1. finite, nonconstant rows that Rastermap can normalize; and
# 2. `nonzero_ROI`, the dataset's supplied active-neuron mask.
#
# This avoids fitting thousands of silent ROIs and does not introduce an
# arbitrary OASIS-amplitude or firing-rate threshold. The mask was computed
# from the complete recording, so this remains an exploratory, within-session
# analysis rather than an independent prediction test. No neurons are sampled
# randomly for the fit.

# %%
recording = dataio.load_recording(RECORDING_NAME)
print(recording)

if recording.nonzero_ROI is None:
    raise ValueError(
        f"{recording.name} has no nonzero_ROI mask; this tutorial requires the "
        "dataset's documented active-neuron selection."
    )

valid_mask = rmt.valid_activity_rows(recording.spike_deconv)
dataset_active_mask = np.asarray(recording.nonzero_ROI, dtype=bool)
fit_mask = valid_mask & dataset_active_mask
fit_roi_rows = np.flatnonzero(fit_mask)
if fit_roi_rows.size < 2:
    raise ValueError(f"{recording.name}: fewer than two usable active neurons")

activity = np.ascontiguousarray(
    recording.spike_deconv[fit_roi_rows],
    dtype=np.float32,
)
recording_name = recording.name
n_recorded_neurons = recording.n_neurons
n_frames = recording.n_frames
fs = recording.fs
duration_min = n_frames / fs / 60
state = recording.state.astype(np.float32, copy=True)
state_codes = dict(dataio.state_codes(recording))
boundary_ind = recording.boundary_ind.astype(np.int64, copy=True)
boundary_minutes = [
    (int(boundary) + 1) / fs / 60
    for boundary in boundary_ind
    if 0 <= int(boundary) < n_frames - 1
]
selected_atlas_labels = (
    np.asarray(recording.atlas, dtype=object)[fit_roi_rows]
    if recording.atlas is not None
    else np.full(fit_roi_rows.size, None, dtype=object)
)

(
    activity_ranked_raster,
    _,
    _,
    activity_order,
    display_bin_centers,
) = viz.binned_spike_raster(
    activity,
    fs,
    DISPLAY_BIN_SECONDS,
    boundary_ind,
    state,
)

print(f"Complete fit matrix: {activity.shape[0]:,} neurons × {n_frames:,} frames")
print(
    f"Dataset-active rows: {dataset_active_mask.sum():,}/{n_recorded_neurons:,}; "
    f"finite and nonconstant: {fit_roi_rows.size:,}"
)
assert activity.shape == (fit_roi_rows.size, n_frames)

del recording
gc.collect()


# %% [markdown]
# ## Step 3 — fit the official Rastermap model
#
# `Rastermap.fit` performs normalization, PCA, clustering, one-dimensional
# cluster sorting, and continuous position upsampling. `model.isort` is the
# resulting neuron order. Atlas labels and brain states are annotations only;
# neither enters the fit.

# %%
model = Rastermap(
    n_clusters=N_CLUSTERS,
    n_PCs=N_PCS,
    locality=LOCALITY,
    time_lag_window=TIME_LAG_WINDOW,
    time_bin=TIME_BIN,
    mean_time=MEAN_TIME,
    bin_size=SUPERNEURON_SIZE,
    random_state=RANDOM_SEED,
    keep_norm_X=True,
    verbose=True,
).fit(activity, compute_X_embedding=True)

model_good = np.asarray(model.igood, dtype=bool).ravel()
model_isort = np.asarray(model.isort, dtype=np.int64).ravel()
sorted_local_rows = model_isort[model_good[model_isort]]
complete_local_order, n_fitted = viz.rastermap_display_order(
    activity.shape[0],
    sorted_local_rows,
)
sorted_roi_rows = fit_roi_rows[sorted_local_rows]

assert np.unique(sorted_local_rows).size == sorted_local_rows.size
assert np.array_equal(np.sort(sorted_roi_rows), np.sort(fit_roi_rows[model_good]))
assert np.unique(complete_local_order).size == activity.shape[0]

embedding_valid = np.asarray(model.embedding_valid).ravel()
isort_valid = np.asarray(model.isort_valid, dtype=np.int64).ravel()
np.testing.assert_array_equal(isort_valid, np.argsort(embedding_valid))

print(f"Rastermap runtime: {model.runtime:.2f} s")
print(f"Embedded neurons: {n_fitted:,}/{activity.shape[0]:,} selected neurons")
print(
    f"Continuous embedding range: {embedding_valid.min():.1f}–{embedding_valid.max():.1f}"
)


# %% [markdown]
# ## Step 4 — one compact input and PCA sanity check
#
# This figure is intentionally diagnostic rather than a second analysis. The
# first two panels confirm that the expected sparse OASIS matrix reached the
# model and that its stored input is centered and scaled. The final panel shows
# how much normalized-matrix energy is represented by the retained PCs. PCA
# energy alone is not evidence that the final fine neuron order is stable.

# %%
detail_start_min, detail_stop_min = viz.resolve_time_limits(
    duration_min,
    DETAIL_WINDOW_MIN,
)
detail_start = max(0, int(round(detail_start_min * 60 * fs)))
detail_stop = min(n_frames, int(round(detail_stop_min * 60 * fs)))

sanity_rng = np.random.default_rng(RANDOM_SEED)
n_sanity_rows = min(N_SANITY_ROWS, activity.shape[0])
sanity_rows = np.sort(
    sanity_rng.choice(activity.shape[0], size=n_sanity_rows, replace=False)
)

singular_values = np.asarray(model.sv, dtype=np.float64)
normalized_energy = 0.0
for chunk_start in range(0, n_frames, 1024):
    chunk = model.X[model_good, chunk_start : chunk_start + 1024].astype(
        np.float64,
        copy=False,
    )
    normalized_energy += float(np.sum(chunk * chunk, dtype=np.float64))
cumulative_pc_energy = np.cumsum(singular_values**2) / normalized_energy
retained_energy_fraction = float(cumulative_pc_energy[-1])

normalized_row_means = model.X[model_good].mean(axis=1)
assert np.max(np.abs(normalized_row_means)) < 1e-5
assert np.all(np.diff(singular_values) <= np.finfo(float).eps * 100)

sanity_figure, sanity_axes = plt.subplots(
    1,
    3,
    figsize=(16, 5),
    constrained_layout=True,
)
raw_detail = activity[sanity_rows, detail_start:detail_stop]
raw_limit = max(float(np.percentile(raw_detail, 99.5)), np.finfo(float).eps)
sanity_axes[0].imshow(
    raw_detail,
    aspect="auto",
    cmap="gray_r",
    vmin=0,
    vmax=raw_limit,
    extent=(detail_start_min, detail_stop_min, n_sanity_rows, 0),
    interpolation="nearest",
    rasterized=True,
)
sanity_axes[0].set_title("OASIS input · display sample")
sanity_axes[0].set_xlabel("recorded time (min)")
sanity_axes[0].set_ylabel(f"{n_sanity_rows} sampled active neurons")

sanity_axes[1].imshow(
    model.X[sanity_rows, detail_start:detail_stop],
    aspect="auto",
    cmap="RdBu_r",
    vmin=-2,
    vmax=2,
    extent=(detail_start_min, detail_stop_min, n_sanity_rows, 0),
    interpolation="nearest",
    rasterized=True,
)
sanity_axes[1].set_title("Official normalized model input")
sanity_axes[1].set_xlabel("recorded time (min)")
sanity_axes[1].set_ylabel("same neurons")

pc_numbers = np.arange(1, singular_values.size + 1)
sanity_axes[2].plot(pc_numbers, singular_values, color="black", lw=1.2)
sanity_axes[2].set_yscale("log")
sanity_axes[2].set_xlabel("principal component")
sanity_axes[2].set_ylabel("singular value")
sanity_axes[2].set_title("PCA spectrum and retained energy")
energy_axis = sanity_axes[2].twinx()
energy_axis.plot(pc_numbers, 100 * cumulative_pc_energy, color="tab:blue", lw=1.2)
energy_axis.set_ylabel("full normalized-matrix energy (%)", color="tab:blue")
energy_axis.tick_params(axis="y", colors="tab:blue")

sanity_figure.suptitle(
    f"{recording_name} · Rastermap input sanity · "
    f"{100 * retained_energy_fraction:.1f}% energy in {singular_values.size} PCs"
)
sanity_path = FIG_DIR / "03_rastermap_01_input_sanity.png"
sanity_figure.savefig(sanity_path, dpi=150, bbox_inches="tight")
print("saved ->", sanity_path)
if SHOW_FIGURES:
    plt.show()


# %% [markdown]
# ## Step 5 — plot the Rastermap-sorted population
#
# The binary raster keeps one row per selected neuron and the strip at its left
# reports that neuron's cortical area. The grayscale panels follow the paper's
# readable presentation: adjacent sorted neurons are averaged in groups of 50
# and each resulting superneuron is z-scored across time.
#
# The complete timeline is shown first. The lower panel magnifies the chosen
# detail window but does not refit the model. Sleep or anesthesia labels are
# aligned exactly to recorded frames.

# %%
activity_row_for_local = np.empty(activity.shape[0], dtype=np.int64)
activity_row_for_local[activity_order] = np.arange(activity.shape[0])
sorted_binary_raster = activity_ranked_raster[
    activity_row_for_local[complete_local_order]
]
display_bin_minutes = display_bin_centers / fs / 60

superneurons = rmt.ordered_superneurons(
    np.asarray(model.X),
    sorted_local_rows,
    SUPERNEURON_SIZE,
)

timeline_view = {
    "state": state,
    "codes": state_codes,
    "fs": fs,
    "n_frames": n_frames,
    "duration_min": duration_min,
    "time_limits_min": (0.0, duration_min),
    "boundary_minutes": boundary_minutes,
}
detail_timeline_view = dict(timeline_view)
detail_timeline_view["time_limits_min"] = (detail_start_min, detail_stop_min)

ordered_figure = plt.figure(figsize=(17, 12), constrained_layout=True)
ordered_grid = ordered_figure.add_gridspec(
    6,
    2,
    width_ratios=(0.22, 20),
    height_ratios=(3.2, 4.0, 0.35, 3.0, 0.35, 0.65),
)
area_axis = ordered_figure.add_subplot(ordered_grid[0, 0])
binary_axis = ordered_figure.add_subplot(ordered_grid[0, 1])
full_axis = ordered_figure.add_subplot(ordered_grid[1, 1])
full_state_axis = ordered_figure.add_subplot(ordered_grid[2, 1])
detail_axis = ordered_figure.add_subplot(ordered_grid[3, 1])
detail_state_axis = ordered_figure.add_subplot(ordered_grid[4, 1])
legend_axis = ordered_figure.add_subplot(ordered_grid[5, :])
legend_axis.axis("off")
for empty_cell in (
    ordered_grid[1, 0],
    ordered_grid[2, 0],
    ordered_grid[3, 0],
    ordered_grid[4, 0],
):
    ordered_figure.add_subplot(empty_cell).axis("off")

_, cortical_legend_handles = viz.plot_cortical_region_strip(
    area_axis,
    selected_atlas_labels,
    complete_local_order,
    n_fitted=n_fitted,
)
binary_rows, binary_time_bins = np.nonzero(sorted_binary_raster)
viz.shade_states(binary_axis, timeline_view, alpha=0.08)
binary_axis.scatter(
    display_bin_minutes[binary_time_bins],
    binary_rows,
    s=0.15,
    marker=".",
    color="black",
    alpha=0.75,
    linewidths=0,
    rasterized=True,
)
binary_axis.set_xlim(0, duration_min)
binary_axis.set_ylim(activity.shape[0], 0)
binary_axis.set_ylabel("active neurons\n(Rastermap order)")
binary_axis.set_title("Binary positive-deconvolution raster · complete recording")
binary_axis.tick_params(axis="x", labelbottom=False)
viz.mark_acquisition_boundaries(binary_axis, timeline_view)
if n_fitted < activity.shape[0]:
    binary_axis.axhline(n_fitted - 0.5, color="tab:red", lw=0.8, ls=(0, (3, 2)))

full_image = full_axis.imshow(
    superneurons,
    aspect="auto",
    origin="upper",
    extent=(0, duration_min, superneurons.shape[0], 0),
    cmap="gray_r",
    vmin=0,
    vmax=1.5,
    interpolation="nearest",
    rasterized=True,
)
full_axis.set_ylabel(f"superneurons\n(≤{SUPERNEURON_SIZE} neurons)")
full_axis.set_title("Sorted, averaged, normalized activity · complete recording")
full_axis.tick_params(axis="x", labelbottom=False)
viz.mark_acquisition_boundaries(full_axis, timeline_view)
viz.plot_state_strip(full_state_axis, timeline_view)

detail_axis.imshow(
    superneurons[:, detail_start:detail_stop],
    aspect="auto",
    origin="upper",
    extent=(
        detail_start_min,
        detail_stop_min,
        superneurons.shape[0],
        0,
    ),
    cmap="gray_r",
    vmin=0,
    vmax=1.5,
    interpolation="nearest",
    rasterized=True,
)
detail_axis.set_xlim(detail_start_min, detail_stop_min)
detail_axis.set_ylabel("superneurons")
detail_axis.set_title("Same fit · enlarged detail window")
detail_axis.tick_params(axis="x", labelbottom=False)
viz.plot_state_strip(detail_state_axis, detail_timeline_view)

ordered_figure.colorbar(
    full_image,
    ax=(full_axis, detail_axis),
    label="normalized deconvolved activity (z score)",
    fraction=0.025,
)
legend_axis.legend(
    handles=cortical_legend_handles,
    loc="center",
    ncol=6,
    frameon=False,
    title="Cortical area (one exact color cell per neuron in the upper raster)",
)
ordered_figure.suptitle(
    f"{recording_name} · official Rastermap · {n_fitted:,} fitted active neurons"
)
ordered_path = FIG_DIR / "03_rastermap_02_sorted_activity.png"
ordered_figure.savefig(ordered_path, dpi=150, bbox_inches="tight")
print("saved ->", ordered_path)
if SHOW_FIGURES:
    plt.show()


# %% [markdown]
# ## Step 6 — interpretation guardrail
#
# Validation across time splits, model seeds, brain states, and nested activity
# populations found reproducible **coarse population structure**, but a less
# stable exact neuron-by-neuron order. Therefore use the superneuron-scale
# patterns as the primary visualization and treat fine adjacency as exploratory.
# The cortical strip is annotation, not evidence that Rastermap modules are
# anatomically localized.

# %%
print("Rastermap tutorial complete.")
print(f"  selected active neurons: {activity.shape[0]:,}/{n_recorded_neurons:,}")
print(f"  fitted neurons:          {n_fitted:,}")
print(f"  PCA energy retained:     {100 * retained_energy_fraction:.2f}%")
print("  interpretation: coarse structure is more robust than exact fine order")
