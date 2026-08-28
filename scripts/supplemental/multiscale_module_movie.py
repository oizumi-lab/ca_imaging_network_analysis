# %% [markdown]
# # Supplemental · Multi-scale network analysis: module-distribution movie
#
# ## Where this script fits
# Scripts 05 and 06 compared selected spatial scales. A movie lets you follow the
# complete transition from single neurons to local population averages. It is
# a visual synthesis of the numbered workflow:
#
# ```text
# activity → equal-density networks → modules → spatial scale → module geography
# ```
#
# The purpose of the movie is not decoration. It helps us see *when* intermixed
# single-cell modules begin to look like spatially localized mesoscale modules,
# while the accompanying labels show how the node and module counts change.
#
# This script uses the following spatial coarse-graining sweep:
#
# ```text
# no averaging -> 2 -> 5 -> 10 -> 20 -> 30 -> 40 neighbours per parcel
# ```
#
# At every scale we repeat the paper's network workflow: spatially neighbouring
# neurons are averaged into parcels, parcel activity is correlated, the top 5%
# of absolute correlations are retained, and Louvain partitions are estimated.
# The settings select either the highest-Q partition or a consensus across all
# runs. Each network node is shown once at its parcel centroid, with colour
# denoting module assignment. The point count therefore falls from one point per
# neuron at no averaging to roughly one point per 40 neurons at the final scale.
#
# Louvain labels have no identity across independent scales, so colours are
# matched one-to-one by maximum normalized correlation between module-membership
# vectors. The movie cuts directly between the seven estimated partitions and
# holds each scale for two seconds; there are no interpolated transition frames.

# %%
from __future__ import annotations

import gc
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

# This file is one level below scripts/, so move up twice to reach the repo root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter, writers
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform

from src.funcnet import coarsegrain as cg
from src.funcnet import dataio, network as net
from src.funcnet.paths import FIG_DIR, RESULTS_DIR

warnings.filterwarnings("ignore", message="invalid value encountered in divide")


# %% [markdown]
# ## Settings
#
# The defaults use the example recording, its NREM window, 2,500 active neurons,
# and Q-max clustering over 20 Louvain runs. Set ``DATASET = "anesthesia"`` to use
# an anesthesia recording (for example, ``RECORDING = "mouse05_ane"``) and
# choose ``STATE = "awake"`` or ``"anesthesia"``. Sleep supports ``"awake"``
# or ``"nrem"``. Set ``CLUSTERING_METHOD`` to ``"consensus"`` when a consensus
# across all runs is preferred. For a fast preview, set ``MAX_NEURONS = 800``
# and ``N_RUNS = 10``.

# %%
SCALES = (1, 2, 5, 10, 20, 30, 40)
DATASET = "sleep"  # "sleep" or "anesthesia"
RECORDING = "mouse02_sleep"  # complete example recording downloaded by script 00
STATE = "nrem"  # sleep: "awake"/"nrem"; anesthesia: "awake"/"anesthesia"
WINDOW = None  # None selects 1500 frames for sleep or 2900 for anesthesia
WINDOW_START = 0  # offset within the selected state's usable frames
MAX_NEURONS = 2500  # responsive preview; set None to use every active neuron
WINDOW_FRAMES = {"sleep": 1500, "ane": 2900}
DEFAULT_RECORDINGS = {"sleep": "mouse02_sleep", "anesthesia": "mouse05_ane"}
DATA_INFO = {"sleep": "sleep", "anesthesia": "ane"}
VALID_STATES = {
    "sleep": ("awake", "nrem"),
    "anesthesia": ("awake", "anesthesia"),
}

K = 0.05
GAMMA = 1.0
N_RUNS = 20  # preview default; use 200 for paper-scale optimization
CLUSTERING_METHOD = "qmax"  # "qmax" (default) or "consensus"
CONSENSUS_REPS = 10
LOUVAIN_SEED = 12345

FPS = 12
HOLD_SECONDS = 2.0
DPI = 150
MARKER_SIZE = None  # None scales point size automatically with parcel count
OUTPUT = None  # None -> results/movies/<recording>_<state>_<method>.mp4

MOVIE_DIR = RESULTS_DIR / "movies"
STATE_TITLES = {
    "awake": "Awake",
    "quiet_awake": "Quiet awake",
    "nrem": "NREM sleep",
    "rem": "REM sleep",
    "anesthesia": "Anesthesia",
}


@dataclass(frozen=True)
class ScaleMap:
    """One estimated partition and the spatial centroid of every network node."""

    neighbor_size: int
    parcel_count: int
    module_count: int
    modularity: float
    assignment_method: str
    parcel_coords: np.ndarray
    parcel_modules: np.ndarray
    neuron_modules: np.ndarray


# %% [markdown]
# ## Partition preparation and cross-scale colour tracking


# %%
def spatial_partitions(
    coords: np.ndarray, scales: tuple[int, ...]
) -> dict[int, np.ndarray]:
    """Build every requested greedy parcel assignment from one distance matrix."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n_neurons, 2)")
    if not scales or scales[0] != 1 or any(scale < 1 for scale in scales):
        raise ValueError("scales must start at 1 and contain only positive integers")
    if tuple(sorted(set(scales))) != scales:
        raise ValueError("scales must be unique and increasing")

    partitions = {1: np.arange(coords.shape[0], dtype=int)}
    coarse_scales = scales[1:]
    if not coarse_scales:
        return partitions

    # close_clustering is deterministic but otherwise recomputes this O(N^2)
    # matrix for every scale. Reuse it here, then release it before correlations.
    distances = squareform(pdist(coords))
    try:
        for scale in coarse_scales:
            partitions[scale] = cg.close_clustering(
                coords[:, 0], coords[:, 1], scale, D=distances
            )
    finally:
        del distances
    return partitions


def estimate_scale_maps(
    activity: np.ndarray,
    coords: np.ndarray,
    partitions: dict[int, np.ndarray],
    *,
    density: float,
    gamma: float,
    n_runs: int,
    clustering_method: str,
    consensus_reps: int,
    seed: int,
) -> dict[int, ScaleMap]:
    """Estimate parcel-network modules and retain every parcel centroid."""
    method = normalize_clustering_method(clustering_method)

    maps: dict[int, ScaleMap] = {}
    scales = tuple(partitions)
    for scale in scales:
        parcel_index = partitions[scale]
        if scale == 1:
            parcel_activity = activity
            parcel_coords = coords
        else:
            parcel_activity, parcel_x, parcel_y = cg.coarse_grain(
                activity,
                coords[:, 0],
                coords[:, 1],
                parcel_index,
            )
            parcel_coords = np.column_stack([parcel_x, parcel_y])

        result = net.modularity_from_activity(
            parcel_activity,
            density=density,
            gamma=gamma,
            n_runs=n_runs,
            negative=True,
            seed=seed,
        )
        if method == "consensus":
            parcel_modules = np.asarray(
                net.consensus_partition(
                    result["ci_all"],
                    reps=consensus_reps,
                    seed=seed,
                ),
                dtype=int,
            )
            assignment_method = f"{n_runs}-run consensus"
        else:
            parcel_modules = np.asarray(result["ci_max"], dtype=int)
            assignment_method = f"Q-max of {n_runs} runs"
        neuron_modules = parcel_modules[parcel_index]
        maps[scale] = ScaleMap(
            neighbor_size=scale,
            parcel_count=parcel_modules.size,
            module_count=int(np.unique(parcel_modules).size),
            modularity=float(result["Q_max"]),
            assignment_method=assignment_method,
            parcel_coords=parcel_coords,
            parcel_modules=parcel_modules,
            neuron_modules=neuron_modules,
        )
        print(
            f"  neighbor size {scale:>2}: {parcel_modules.size:>5} parcels, "
            f"{maps[scale].module_count:>3} modules, {assignment_method}, "
            f"max-run Q={maps[scale].modularity:.3f}",
            flush=True,
        )

        if scale != 1:
            del parcel_activity
        del result, parcel_modules, neuron_modules
        gc.collect()
    return maps


def normalize_clustering_method(value: str) -> str:
    """Normalize supported Q-max/consensus setting spellings."""
    if not isinstance(value, str):
        raise TypeError("clustering method must be a string")
    method = value.strip().casefold().replace("-", "").replace("_", "")
    if method in {"qmax", "maxq"}:
        return "qmax"
    if method == "consensus":
        return "consensus"
    raise ValueError("clustering method must be 'qmax' or 'consensus'")


def _size_ranked_ids(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Assign compact display IDs from largest to smallest first-scale module."""
    unique, counts = np.unique(labels, return_counts=True)
    unique = unique[np.argsort(-counts, kind="stable")]
    display = np.empty(labels.size, dtype=int)
    for display_id, label in enumerate(unique):
        display[labels == label] = display_id
    return display, unique.size


def _correlation_aligned_ids(
    previous_ids: np.ndarray,
    current_labels: np.ndarray,
    next_unused_id: int,
) -> tuple[np.ndarray, int, float]:
    """Match modules by maximum normalized membership-vector correlation.

    Each module is represented by its binary neuron-membership vector. The
    matching score is their normalized dot product (cosine correlation), so a
    large module cannot dominate merely because its raw overlap count is large.
    Hungarian assignment finds the one-to-one mapping with the largest total
    score while keeping every module visually distinguishable within a frame.
    """
    previous_ids = np.asarray(previous_ids, dtype=int)
    current_labels = np.asarray(current_labels)
    if previous_ids.shape != current_labels.shape or previous_ids.ndim != 1:
        raise ValueError("previous_ids and current_labels must be aligned 1-D arrays")

    previous = np.unique(previous_ids)
    current = np.unique(current_labels)
    overlap = np.zeros((previous.size, current.size), dtype=np.int64)
    for column, label in enumerate(current):
        overlap[:, column] = np.bincount(
            np.searchsorted(previous, previous_ids[current_labels == label]),
            minlength=previous.size,
        )

    previous_sizes = np.bincount(
        np.searchsorted(previous, previous_ids), minlength=previous.size
    )
    current_sizes = np.asarray(
        [np.count_nonzero(current_labels == label) for label in current],
        dtype=np.int64,
    )
    denominator = np.sqrt(np.outer(previous_sizes, current_sizes))
    correlation = np.divide(
        overlap,
        denominator,
        out=np.zeros_like(overlap, dtype=float),
        where=denominator > 0,
    )

    rows, columns = linear_sum_assignment(-correlation)
    mapping: dict[object, int] = {}
    matched_correlations = []
    for row, column in zip(rows, columns):
        if overlap[row, column] > 0:
            mapping[current[column].item()] = int(previous[row])
            matched_correlations.append(float(correlation[row, column]))

    # New/split modules get genuinely new colours rather than silently reusing
    # an extinct module colour and implying continuity unsupported by overlap.
    for label in current:
        key = label.item()
        if key not in mapping:
            mapping[key] = next_unused_id
            next_unused_id += 1

    aligned = np.asarray([mapping[label.item()] for label in current_labels], dtype=int)
    mean_correlation = (
        float(np.mean(matched_correlations)) if matched_correlations else 0.0
    )
    return aligned, next_unused_id, mean_correlation


def _parcel_color_ids(entry: ScaleMap, neuron_color_ids: np.ndarray) -> np.ndarray:
    """Collapse neuron-aligned display colours to one colour per parcel."""
    module_to_color: dict[int, int] = {}
    for module, color_id in zip(entry.neuron_modules, neuron_color_ids):
        module_to_color.setdefault(int(module), int(color_id))
    return np.asarray(
        [module_to_color[int(module)] for module in entry.parcel_modules],
        dtype=int,
    )


def cross_scale_color_ids(scale_maps: dict[int, ScaleMap]) -> dict[int, np.ndarray]:
    """Return parcel colours maximizing membership correlation across scales."""
    scales = tuple(scale_maps)
    first = scales[0]
    first_ids, next_unused = _size_ranked_ids(scale_maps[first].neuron_modules)
    neuron_ids = {first: first_ids}
    for previous, current in zip(scales[:-1], scales[1:]):
        aligned, next_unused, mean_correlation = _correlation_aligned_ids(
            neuron_ids[previous],
            scale_maps[current].neuron_modules,
            next_unused,
        )
        neuron_ids[current] = aligned
        print(
            f"  color match {previous:>2} -> {current:>2}: mean normalized "
            f"membership correlation = {mean_correlation:.3f}",
            flush=True,
        )
    return {
        scale: _parcel_color_ids(scale_maps[scale], neuron_ids[scale])
        for scale in scales
    }


def categorical_palette(n_colors: int) -> np.ndarray:
    """Build an opaque, high-contrast palette large enough for tracked IDs."""
    if n_colors < 1:
        raise ValueError("n_colors must be positive")
    blocks = [
        plt.colormaps[name].resampled(20)(np.arange(20))
        for name in ("tab20", "tab20b", "tab20c")
    ]
    palette = np.vstack(blocks)
    if n_colors > palette.shape[0]:
        extra = plt.colormaps["gist_rainbow"].resampled(n_colors - 59)(
            np.arange(n_colors - 60)
        )
        palette = np.vstack([palette, extra])
    return palette[:n_colors]


# %% [markdown]
# ## Movie rendering
#
# The movie shows only the seven exact partitions. Each one is held for
# ``HOLD_SECONDS = 3`` before a hard cut to the next neighbour size.


# %%
def frame_sequence(
    scales: tuple[int, ...],
    *,
    fps: int,
    hold_seconds: float,
) -> list[int]:
    """Repeat each exact neighbor size for its requested screen time."""
    hold_count = max(1, round(fps * hold_seconds))
    return [scale for scale in scales for _ in range(hold_count)]


def neighbor_title(scale: int) -> str:
    """Human-readable scale label shown at the top of every held frame."""
    if scale == 1:
        return "Neighbor size: no averaging"
    return f"Neighbor size: {scale}"


def render_movie(
    scale_maps: dict[int, ScaleMap],
    color_ids: dict[int, np.ndarray],
    *,
    recording: str,
    state: str,
    output: Path,
    fps: int,
    hold_seconds: float,
    dpi: int,
    marker_size: float | None,
) -> Path:
    """Render one centroid per network parcel to MP4 or GIF."""
    scales = tuple(scale_maps)
    specs = frame_sequence(
        scales,
        fps=fps,
        hold_seconds=hold_seconds,
    )
    max_color_id = max(int(ids.max()) for ids in color_ids.values())
    palette = categorical_palette(max_color_id + 1)
    colors = {scale: palette[ids] for scale, ids in color_ids.items()}

    first_entry = scale_maps[scales[0]]
    first_size = (
        float(np.clip(28_000 / first_entry.parcel_count, 2.0, 60.0))
        if marker_size is None
        else marker_size
    )
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    scatter = ax.scatter(
        first_entry.parcel_coords[:, 0],
        first_entry.parcel_coords[:, 1],
        s=first_size,
        c=colors[scales[0]],
        edgecolors="none",
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.15")
        spine.set_linewidth(1.2)

    full_coords = first_entry.parcel_coords
    x_span = float(np.ptp(full_coords[:, 0]))
    y_span = float(np.ptp(full_coords[:, 1]))
    ax.set_xlim(
        full_coords[:, 0].min() - 0.015 * x_span,
        full_coords[:, 0].max() + 0.015 * x_span,
    )
    # Account for the inverted axis when setting explicit padded limits.
    ax.set_ylim(
        full_coords[:, 1].max() + 0.015 * y_span,
        full_coords[:, 1].min() - 0.015 * y_span,
    )

    title = ax.set_title(neighbor_title(scales[0]), fontsize=17, pad=12)
    info = ax.text(
        0.015,
        0.015,
        "",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="0.15",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
    )

    state_title = STATE_TITLES.get(state, state.replace("_", " ").title())

    def update(frame_number: int):
        scale = specs[frame_number]
        entry = scale_maps[scale]
        point_size = (
            float(np.clip(28_000 / entry.parcel_count, 2.0, 60.0))
            if marker_size is None
            else marker_size
        )
        scatter.set_offsets(entry.parcel_coords)
        scatter.set_sizes(np.full(entry.parcel_count, point_size))
        scatter.set_facecolors(colors[scale])
        title.set_text(neighbor_title(scale))
        info.set_text(
            f"{recording} · {state_title} · {entry.parcel_count:,} centroids · "
            f"{entry.module_count} modules · {entry.assignment_method}"
        )
        return scatter, title, info

    animation = FuncAnimation(
        fig,
        update,
        frames=len(specs),
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".mp4":
        if not writers.is_available("ffmpeg"):
            raise RuntimeError(
                "MP4 output requires ffmpeg; use an output ending in .gif"
            )
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=4000,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
    elif suffix == ".gif":
        writer = PillowWriter(fps=fps)
    else:
        raise ValueError("output must end in .mp4 or .gif")

    print(f"Rendering {len(specs)} frames to {output}", flush=True)
    animation.save(
        output, writer=writer, dpi=dpi, savefig_kwargs={"facecolor": "white"}
    )
    plt.close(fig)
    return output


# %% [markdown]
# ## Step 1 — load one state window
#
# ``DATASET`` chooses sleep or anesthesia, and ``RECORDING`` explicitly names
# its recording file. ``RECORDING = None`` remains available to select the
# default example for that dataset. ``STATE`` then chooses a compatible
# awake/NREM/anesthesia window. We use the dataset's active-neuron mask and copy
# the selected state frames before releasing the much larger full recording object.


# %%
dataset = DATASET.strip().casefold()
if dataset == "ane":
    dataset = "anesthesia"
if dataset not in DEFAULT_RECORDINGS:
    raise ValueError("DATASET must be 'sleep' or 'anesthesia'")

selected_recording = RECORDING or DEFAULT_RECORDINGS[dataset]
rec = dataio.load_recording(selected_recording)
recording_name = rec.name
if rec.data_info != DATA_INFO[dataset]:
    raise ValueError(
        f"{recording_name!r} is a {rec.data_info!r} recording, incompatible with "
        f"DATASET={dataset!r}"
    )

state = STATE.strip().casefold().replace(" ", "_")
if state == "nrem_sleep":
    state = "nrem"
if state not in VALID_STATES[dataset]:
    raise ValueError(
        f"STATE={STATE!r} is incompatible with DATASET={dataset!r}; "
        f"choose from {VALID_STATES[dataset]}"
    )

clustering_method = normalize_clustering_method(CLUSTERING_METHOD)
if not isinstance(WINDOW_START, (int, np.integer)) or WINDOW_START < 0:
    raise ValueError("WINDOW_START must be a non-negative integer")
if WINDOW is not None and (
    not isinstance(WINDOW, (int, np.integer)) or isinstance(WINDOW, (bool, np.bool_))
):
    raise TypeError("WINDOW must be a positive integer or None")
if WINDOW is not None and WINDOW <= 0:
    raise ValueError("WINDOW must be positive")
window = WINDOW or WINDOW_FRAMES[rec.data_info]
available = dataio.state_frames(rec, state)
window_stop = WINDOW_START + window
if available.size < window_stop:
    raise ValueError(
        f"{rec.name} has {available.size} usable {state} frames, fewer than the "
        f"requested slice [{WINDOW_START}:{window_stop}]"
    )

rows = dataio.select_neuron_rows(rec, max_neurons=MAX_NEURONS, seed=0)
if rows.size <= max(SCALES):
    raise ValueError(
        f"At least {max(SCALES) + 1} neurons are needed to retain two parcels "
        f"at neighbor size {max(SCALES)}; selected {rows.size}"
    )

coords = rec.centroid_um[rows]
frames = available[WINDOW_START:window_stop]
activity = rec.spike_smoothed[np.ix_(rows, frames)]
print(
    f"{dataset} dataset · {recording_name} · {state} frames "
    f"[{WINDOW_START}:{window_stop}] · {rows.size:,} active neurons · "
    f"clustering={clustering_method} · scales={list(SCALES)}",
    flush=True,
)

del rec, rows, frames, available
gc.collect()


# %% [markdown]
# ## Step 2 — construct spatial parcels at every scale
#
# ``nnei = 1`` is the identity partition (no averaging). At every larger value,
# each unassigned neuron seeds a parcel with its nearest unassigned neighbours.
# The same neuron coordinates and deterministic parcel rule are used throughout.


# %%
print("Preparing spatial parcels...", flush=True)
partitions = spatial_partitions(coords, SCALES)
for neighbor_size, parcel_index in partitions.items():
    print(
        f"  neighbor size {neighbor_size:>2}: "
        f"{np.unique(parcel_index).size:>5} parcels",
        flush=True,
    )


# %% [markdown]
# ## Step 3 — rebuild the functional network and estimate its modules
#
# For each parcel size: average activity within parcels, correlate parcels,
# retain the strongest ``K = 5%`` of absolute correlations, and run Louvain
# ``N_RUNS`` times. ``CLUSTERING_METHOD = "qmax"`` selects the partition with
# the highest Q; ``"consensus"`` builds one consensus from all run assignments.
# The movie shows one point at each parcel centroid, colored by module. For the
# consensus option, ``max-run Q`` is only a diagnostic of the run ensemble and
# is not the modularity of the consensus partition itself.


# %%
print("Estimating module maps...", flush=True)
scale_maps = estimate_scale_maps(
    activity,
    coords,
    partitions,
    density=K,
    gamma=GAMMA,
    n_runs=N_RUNS,
    clustering_method=clustering_method,
    consensus_reps=CONSENSUS_REPS,
    seed=LOUVAIN_SEED,
)

# The selected window is no longer needed after all partitions are estimated.
del activity
gc.collect()


# %% [markdown]
# ## Step 4 — align module colours between neighbouring scales
#
# Louvain's numeric module labels are arbitrary. This step uses Hungarian
# matching to maximize normalized neuron-membership correlation between modules
# at adjacent scales. Matched modules retain exactly the same color, preventing
# a label permutation from appearing as a spatial reorganization; new or split
# modules receive a new categorical color.


# %%
color_ids = cross_scale_color_ids(scale_maps)

# %% [markdown]
# ## Static overview for the tutorial slides
#
# The movie is useful interactively; this contact sheet records the same exact
# partitions in a single reproducible tutorial figure.

# %%
max_color_id = max(int(ids.max()) for ids in color_ids.values())
palette = categorical_palette(max_color_id + 1)
fig, axes = plt.subplots(2, 4, figsize=(13.2, 7.0))
for ax, scale in zip(axes.ravel(), scale_maps):
    entry = scale_maps[scale]
    point_size = float(np.clip(18_000 / entry.parcel_count, 2.0, 55.0))
    ax.scatter(
        entry.parcel_coords[:, 0],
        entry.parcel_coords[:, 1],
        s=point_size,
        c=palette[color_ids[scale]],
        edgecolors="none",
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    label = "single cells" if scale == 1 else f"{scale} neurons/parcel"
    ax.set_title(
        f"{label}\n{entry.parcel_count:,} nodes · {entry.module_count} modules",
        fontsize=9,
    )
for ax in axes.ravel()[len(scale_maps):]:
    ax.set_axis_off()
fig.suptitle(
    f"Module geography across scales — {recording_name}, {STATE_TITLES.get(state, state)}",
    fontsize=14,
)
fig.tight_layout()
FIG_DIR.mkdir(parents=True, exist_ok=True)
overview_path = FIG_DIR / f"multiscale_overview_{recording_name}_{state}.png"
fig.savefig(overview_path, dpi=160, bbox_inches="tight")
plt.show()
print(f"Saved: {overview_path}", flush=True)

# %% [markdown]
# ## Step 5 — render the movie
#
# Each of the seven estimated partitions is held for ``HOLD_SECONDS``, followed by
# a direct cut to the next scale. Set ``OUTPUT`` to a path ending in ``.gif`` to
# use the Pillow writer instead.


# %%
output = (
    Path(OUTPUT)
    if OUTPUT is not None
    else MOVIE_DIR
    / f"multiscale_modules_{recording_name}_{state}_{clustering_method}.mp4"
)
movie_path = render_movie(
    scale_maps,
    color_ids,
    recording=recording_name,
    state=state,
    output=output,
    fps=FPS,
    hold_seconds=HOLD_SECONDS,
    dpi=DPI,
    marker_size=MARKER_SIZE,
)
print(f"Saved: {movie_path}", flush=True)

# %% [markdown]
# ## Takeaway
#
# The movie follows one state window through the complete spatial-scale sweep.
# At every frame the signals, correlations, equal-density graph, and modules are
# recomputed at that scale. The animation therefore visualizes a change in the
# level of observation—not a motion of neurons or modules through time.
