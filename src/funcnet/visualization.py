"""Preparation and plotting helpers for calcium-activity visualizations.

The functions in this module are reusable, axis-level building blocks.  They do
not load recordings or save files: tutorials retain those orchestration choices
so readers can see which data and output they selected.  Display preparation is
also kept separate from network estimation; temporal binning here is strictly a
rendering choice and must not be mistaken for analysis preprocessing.

Most activity plotting functions accept the compact ``view`` mapping assembled
by ``scripts/02_visualization_activity.py``.  Keeping a documented view contract
lets future tutorials reuse the plots without importing an executable tutorial.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NotRequired, TypedDict

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

from . import timeseries as ts


# These categorical defaults match the state labels exposed by ``dataio``.  They
# are plain dictionaries so a caller can copy or override them without modifying
# module state.
DEFAULT_STATE_COLORS = {
    "awake": "royalblue",
    "quiet_awake": "0.65",
    "nrem": "crimson",
    "rem": "mediumpurple",
    "anesthesia": "goldenrod",
}
DEFAULT_STATE_LABELS = {
    "awake": "Awake",
    "quiet_awake": "Quiet awake",
    "nrem": "NREM sleep",
    "rem": "REM sleep",
    "anesthesia": "Anesthesia",
}
DEFAULT_STATE_SHORT_LABELS = {
    "awake": "Awake",
    "quiet_awake": "Quiet",
    "nrem": "NREM",
    "rem": "REM",
    "anesthesia": "Anesthesia",
}
DEFAULT_SESSION_TITLES = {
    "sleep": "Sleep recording: wakefulness and sleep stages",
    "ane": "Anesthesia recording: awake and isoflurane anesthesia",
}

# Compact cortical-area categories used by the spatial inspection map,
# brain-region activity raster, and row-aligned Rastermap diagnostics. ``Other``
# means that the source contains a valid Allen-atlas acronym outside the nine
# prominently displayed regions;
# ``Unknown`` is reserved for a genuinely missing/unassigned source label.  The
# distinction prevents missing metadata from silently becoming an anatomical
# assignment.  Dictionary insertion order is also the stable legend order.
CORTICAL_REGION_COLORS = {
    # Motor areas: widely separated dark and medium greens.
    "MOs": "#005a32",
    "MOp": "#41ab5d",
    # Primary somatosensory subdivisions: orange, red, coral, pink, magenta.
    "SSp-ll": "#8c2d04",
    "SSp-ul": "#e6550d",
    "SSp-un": "#cb181d",
    "SSp-bfd": "#fb6a4a",
    "SSp-tr": "#f768a1",
    # Retrosplenial areas: widely separated dark and medium blues.
    "RSPagl": "#08306b",
    "RSPd": "#2b8cbe",
    # Grouped valid labels are gray; missing labels remain visibly distinct.
    "Other": "#666666",
    "Unknown": "#bdbdbd",
}

# Unabridged Allen-atlas names for the compact spatial-map categories.  Keeping
# this separate from the color mapping lets dense row strips retain short
# acronyms while explanatory figure legends can show both forms.
CORTICAL_REGION_NAMES = {
    "MOs": "Secondary motor area",
    "MOp": "Primary motor area",
    "SSp-ll": "Primary somatosensory area, lower limb",
    "SSp-ul": "Primary somatosensory area, upper limb",
    "SSp-un": "Primary somatosensory area, unassigned",
    "SSp-bfd": "Primary somatosensory area, barrel field",
    "SSp-tr": "Primary somatosensory area, trunk",
    "RSPagl": "Retrosplenial area, lateral agranular part",
    "RSPd": "Retrosplenial area, dorsal part",
    "Other": "Other atlas areas (grouped)",
    "Unknown": "Missing or unassigned atlas label",
}

# Exact atlas regions present in the activity dataset. Unlike the compact
# cortical map above, this palette keeps the visual and minor somatosensory
# acronyms separate so a brain-region-grouped raster has one block per supplied
# anatomical label. Dictionary insertion order is the stable display order.
BRAIN_REGION_COLORS = {
    # Motor areas: widely separated dark and medium greens.
    "MOs": "#005a32",
    "MOp": "#41ab5d",
    # Primary somatosensory subdivisions: orange through red, pink, and magenta.
    "SSp-ll": "#8c2d04",
    "SSp-ul": "#e6550d",
    "SSp-un": "#cb181d",
    "SSp-bfd": "#fb6a4a",
    "SSp-tr": "#f768a1",
    "SSp-m": "#c51b8a",
    "SSp-n": "#7a0177",
    # Retrosplenial areas: widely separated dark and medium blues.
    "RSPagl": "#08306b",
    "RSPd": "#2b8cbe",
    # Exact visual-area categories remain distinguishable in purples.
    "VISa": "#3f007d",
    "VISam": "#54278f",
    "VISp": "#6a51a3",
    "VISpm": "#807dba",
    "VISrl": "#9e9ac8",
    "root": "#4d4d4d",
    "Other": "#666666",
    "Unknown": "#bdbdbd",
}


def display_cortical_region(atlas_label: object) -> str:
    """Map one exact atlas label to a compact display category.

    The recordings target layer 2/3, so a terminal ``2/3`` suffix is removed.
    Valid but less common atlas acronyms are grouped into ``Other``.  Empty and
    conventional missing-value strings remain separately visible as
    ``Unknown`` rather than being assigned to an anatomical catch-all.
    """
    if atlas_label is None:
        return "Unknown"
    if isinstance(atlas_label, bytes):
        atlas_label = atlas_label.decode("utf-8", errors="replace")
    text = str(atlas_label).strip()
    if text.casefold() in {"", "nan", "none", "unknown", "unassigned", "na", "n/a"}:
        return "Unknown"
    area = text.removesuffix("2/3")
    if area in CORTICAL_REGION_COLORS and area not in {"Other", "Unknown"}:
        return area
    return "Other"


def cortical_region_labels(atlas_labels: Sequence[object] | np.ndarray) -> np.ndarray:
    """Return one compact cortical-region category per original ROI row."""
    atlas = np.asarray(atlas_labels, dtype=object)
    if atlas.ndim != 1:
        raise ValueError("atlas_labels must be a one-dimensional ROI-aligned array")
    if atlas.size == 0:
        raise ValueError("atlas_labels must contain at least one ROI label")
    return np.asarray([display_cortical_region(label) for label in atlas], dtype=object)


def display_brain_region(atlas_label: object) -> str:
    """Return one exact layer-collapsed atlas region for the activity raster."""
    if atlas_label is None:
        return "Unknown"
    if isinstance(atlas_label, bytes):
        atlas_label = atlas_label.decode("utf-8", errors="replace")
    text = str(atlas_label).strip()
    if text.casefold() in {"", "nan", "none", "unknown", "unassigned", "na", "n/a"}:
        return "Unknown"
    region = text.removesuffix("2/3")
    if region in BRAIN_REGION_COLORS and region not in {"Other", "Unknown"}:
        return region
    return "Other"


def brain_region_labels(atlas_labels: Sequence[object] | np.ndarray) -> np.ndarray:
    """Return one exact display atlas region per original ROI row."""
    atlas = np.asarray(atlas_labels, dtype=object)
    if atlas.ndim != 1:
        raise ValueError("atlas_labels must be a one-dimensional ROI-aligned array")
    if atlas.size == 0:
        raise ValueError("atlas_labels must contain at least one ROI label")
    return np.asarray([display_brain_region(label) for label in atlas], dtype=object)


def brain_region_order(
    atlas_labels: Sequence[object] | np.ndarray,
    activity_order: np.ndarray,
) -> np.ndarray:
    """Group every ROI by exact atlas region, retaining rank within groups.

    ``activity_order`` contains original ROI indices from most to least active.
    Region groups follow the stable order of :data:`BRAIN_REGION_COLORS`.
    A stable category sort therefore changes only the between-region ordering;
    neurons within each region keep their existing whole-session activity rank.
    """
    regions = brain_region_labels(atlas_labels)
    order = np.asarray(activity_order)
    n_neurons = regions.size
    if order.ndim != 1 or order.size != n_neurons:
        raise ValueError("activity_order must contain one ROI index per neuron")
    if not np.issubdtype(order.dtype, np.integer):
        raise TypeError("activity_order must contain integer ROI indices")
    if (
        np.any(order < 0)
        or np.any(order >= n_neurons)
        or np.unique(order).size != n_neurons
    ):
        raise ValueError("activity_order must be a permutation of every ROI")

    region_to_index = {
        region: index for index, region in enumerate(BRAIN_REGION_COLORS)
    }
    region_ids = np.asarray(
        [region_to_index[region] for region in regions[order]],
        dtype=np.int16,
    )
    return order[np.argsort(region_ids, kind="stable")].astype(np.int64, copy=False)


def cortical_region_legend_handles(
    regions: Sequence[str] | np.ndarray | None = None,
    *,
    unabridged: bool = False,
) -> list[Line2D]:
    """Build stable, point-style legend handles for cortical categories.

    Passing ``regions`` restricts the legend to categories actually present,
    while retaining the shared anatomical order.  ``None`` requests every
    category, including the distinct ``Other`` and ``Unknown`` entries. Set
    ``unabridged`` to show each acronym together with its full atlas-area name.
    """
    if regions is None:
        shown = set(CORTICAL_REGION_COLORS)
    else:
        region_array = np.asarray(regions, dtype=object)
        if region_array.ndim != 1:
            raise ValueError("regions must be one-dimensional")
        shown = set(region_array.tolist())
        unexpected = shown.difference(CORTICAL_REGION_COLORS)
        if unexpected:
            raise ValueError(f"Unrecognized cortical categories: {sorted(unexpected)}")
    return [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=5,
            markerfacecolor=color,
            markeredgewidth=0,
            label=(
                f"{region} — {CORTICAL_REGION_NAMES[region]}"
                if unabridged
                else region
            ),
        )
        for region, color in CORTICAL_REGION_COLORS.items()
        if region in shown
    ]


def brain_region_legend_handles(
    regions: Sequence[str] | np.ndarray | None = None,
) -> list[Line2D]:
    """Build stable legend handles for exact atlas regions in the raster."""
    if regions is None:
        shown = set(BRAIN_REGION_COLORS)
    else:
        region_array = np.asarray(regions, dtype=object)
        if region_array.ndim != 1:
            raise ValueError("regions must be one-dimensional")
        shown = set(region_array.tolist())
        unexpected = shown.difference(BRAIN_REGION_COLORS)
        if unexpected:
            raise ValueError(f"Unrecognized brain regions: {sorted(unexpected)}")
    return [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=5,
            markerfacecolor=color,
            markeredgewidth=0,
            label=region,
        )
        for region, color in BRAIN_REGION_COLORS.items()
        if region in shown
    ]


def plot_cortical_region_strip(
    ax,
    atlas_labels: Sequence[object] | np.ndarray,
    roi_order: np.ndarray,
    n_fitted: int | None = None,
) -> tuple[Any, list[Line2D]]:
    """Draw one exact cortical-area color cell per neuron in ``roi_order``.

    ``roi_order`` may contain either every original ROI or an explicit selected
    subset, but every listed ROI must be unique.  If ``n_fitted`` is supplied, a
    dashed separator marks where appended unfitted rows begin.  The returned
    handles include only categories present in the strip and can be passed
    directly to a figure- or axis-level legend.
    """
    regions = cortical_region_labels(atlas_labels)
    order = np.asarray(roi_order)
    n_recorded_neurons = regions.size
    if order.ndim != 1 or order.size == 0:
        raise ValueError("roi_order must be a non-empty one-dimensional array")
    if not np.issubdtype(order.dtype, np.integer):
        raise TypeError("roi_order must contain integer ROI indices")
    if (
        np.any(order < 0)
        or np.any(order >= n_recorded_neurons)
        or np.unique(order).size != order.size
    ):
        raise ValueError("roi_order must contain unique valid ROI rows")
    n_display_neurons = order.size
    if n_fitted is not None:
        if isinstance(n_fitted, (bool, np.bool_)) or not isinstance(
            n_fitted,
            (int, np.integer),
        ):
            raise TypeError("n_fitted must be an integer or None")
        if not 0 <= n_fitted <= n_display_neurons:
            raise ValueError("n_fitted must lie between zero and the displayed count")

    ordered_regions = regions[order]
    region_order = tuple(CORTICAL_REGION_COLORS)
    region_to_index = {region: index for index, region in enumerate(region_order)}
    color_ids = np.asarray(
        [region_to_index[region] for region in ordered_regions],
        dtype=np.int16,
    )
    image = ax.imshow(
        color_ids[:, np.newaxis],
        aspect="auto",
        origin="upper",
        extent=(0, 1, n_display_neurons - 0.5, -0.5),
        cmap=ListedColormap([CORTICAL_REGION_COLORS[name] for name in region_order]),
        vmin=-0.5,
        vmax=len(region_order) - 0.5,
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(n_display_neurons, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("area", fontsize=8, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if n_fitted is not None and 0 < n_fitted < n_display_neurons:
        ax.axhline(n_fitted - 0.5, color="tab:red", lw=0.8, ls=(0, (3, 2)))

    handles = cortical_region_legend_handles(ordered_regions)
    return image, handles


def plot_brain_region_strip(
    ax,
    atlas_labels: Sequence[object] | np.ndarray,
    roi_order: np.ndarray,
) -> tuple[Any, list[Line2D]]:
    """Draw one exact atlas-region color cell per neuron in ``roi_order``."""
    regions = brain_region_labels(atlas_labels)
    order = np.asarray(roi_order)
    n_recorded_neurons = regions.size
    if order.ndim != 1 or order.size == 0:
        raise ValueError("roi_order must be a non-empty one-dimensional array")
    if not np.issubdtype(order.dtype, np.integer):
        raise TypeError("roi_order must contain integer ROI indices")
    if (
        np.any(order < 0)
        or np.any(order >= n_recorded_neurons)
        or np.unique(order).size != order.size
    ):
        raise ValueError("roi_order must contain unique valid ROI rows")

    ordered_regions = regions[order]
    region_names = tuple(BRAIN_REGION_COLORS)
    region_to_index = {
        region: index for index, region in enumerate(region_names)
    }
    color_ids = np.asarray(
        [region_to_index[region] for region in ordered_regions],
        dtype=np.int16,
    )
    image = ax.imshow(
        color_ids[:, np.newaxis],
        aspect="auto",
        origin="upper",
        extent=(0, 1, order.size - 0.5, -0.5),
        cmap=ListedColormap([BRAIN_REGION_COLORS[name] for name in region_names]),
        vmin=-0.5,
        vmax=len(region_names) - 0.5,
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(order.size, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("region", fontsize=8, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = brain_region_legend_handles(ordered_regions)
    return image, handles


class DFFHeatmapPanel(TypedDict):
    """One acquisition/state-aligned panel returned by :func:`binned_dff_heatmaps`."""

    start_min: float
    stop_min: float
    values: np.ndarray


class ActivityView(TypedDict):
    """Compact recording representation consumed by the activity plot helpers.

    ``scripts/02_visualization_activity.py`` builds this mapping while one large
    recording is in memory.  Declaring every key here makes the axis-level plots
    reusable without requiring callers to reverse-engineer a tutorial-local
    dictionary schema.
    """

    name: str
    data_info: str
    n_neurons: int
    n_frames: int
    fs: float
    duration_min: float
    time_limits_min: tuple[float, float]
    state: np.ndarray
    codes: dict[float, str]
    trace_ids: np.ndarray | None
    dff: np.ndarray | None
    dff_view: str
    dff_heatmaps: list[DFFHeatmapPanel] | None
    dff_color_limit: float | None
    dff_bin_frames: int | None
    raster: np.ndarray
    population_active_fraction: np.ndarray
    bin_frames: int
    bin_centers_min: np.ndarray
    neuron_order: np.ndarray
    brain_regions: np.ndarray
    brain_region_order: np.ndarray
    rastermap_X_embedding: NotRequired[np.ndarray | None]
    rastermap_embedding: NotRequired[np.ndarray | None]
    rastermap_isort: NotRequired[np.ndarray | None]
    rastermap_valid_rows: NotRequired[np.ndarray | None]
    rastermap_runtime_seconds: NotRequired[float | None]
    rastermap_cached: NotRequired[bool]
    rastermap_display_selected_only: NotRequired[bool]
    rastermap_stop_min: NotRequired[float | None]
    rastermap_parameters: NotRequired[dict[str, int | float | bool | str] | None]
    rastermap_version: NotRequired[str | None]
    boundary_minutes: list[float]
    acquisition_segments: list[tuple[int, int]]


def resolve_time_limits(
    duration_min: float,
    requested_range: Sequence[float] | np.ndarray | None,
) -> tuple[float, float]:
    """Validate a requested minute range and clip it to one recording.

    ``None`` retains the complete sequence.  A ``(start, stop)`` pair is
    interpreted on the original recorded-time axis, not relative to a state.
    """
    duration_min = float(duration_min)
    if not np.isfinite(duration_min) or duration_min <= 0:
        raise ValueError("duration_min must be finite and positive")
    if requested_range is None:
        return 0.0, duration_min
    if (
        not isinstance(requested_range, (tuple, list, np.ndarray))
        or len(requested_range) != 2
    ):
        raise ValueError("requested_range must be None or a (start, stop) pair")

    start, stop = map(float, requested_range)
    if not np.isfinite(start) or not np.isfinite(stop) or start < 0 or stop <= start:
        raise ValueError("time ranges must satisfy 0 <= start < stop")
    if start >= duration_min:
        raise ValueError(
            f"The requested view starts at {start:g} min, after this "
            f"{duration_min:.2f}-min recording ends"
        )
    return start, min(stop, duration_min)


def select_trace_neurons(n_neurons: int, n_select: int, seed: int) -> np.ndarray:
    """Return a reproducible, unbiased neuron sample for a trace figure."""
    if isinstance(n_neurons, (bool, np.bool_)) or not isinstance(
        n_neurons, (int, np.integer)
    ):
        raise TypeError("n_neurons must be an integer")
    if n_neurons <= 0:
        raise ValueError("n_neurons must be positive")
    if isinstance(n_select, (bool, np.bool_)) or not isinstance(
        n_select, (int, np.integer)
    ):
        raise TypeError("n_select must be an integer")
    if n_select <= 0:
        raise ValueError("n_select must be positive")

    rng = np.random.default_rng(seed)
    n_select = min(int(n_select), int(n_neurons))
    return np.sort(rng.choice(n_neurons, size=n_select, replace=False))


def rastermap_display_order(
    n_neurons: int,
    rastermap_isort: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Return a complete ROI order headed by Rastermap-sorted neurons.

    Rastermap cannot assign a position to a non-finite or constant activity
    row because its per-neuron z-score is undefined.  The fitted ROI indices
    retain their exact Rastermap order; any such unfitted rows are appended in
    their original ROI-row order so the binary display still shows every
    recorded neuron exactly once.  The second return value marks the boundary
    between fitted and appended rows.
    """
    if isinstance(n_neurons, (bool, np.bool_)) or not isinstance(
        n_neurons,
        (int, np.integer),
    ):
        raise TypeError("n_neurons must be an integer")
    if n_neurons <= 0:
        raise ValueError("n_neurons must be positive")

    rastermap_isort = np.asarray(rastermap_isort)
    if rastermap_isort.ndim != 1:
        raise ValueError("rastermap_isort must be one-dimensional")
    if rastermap_isort.size == 0:
        raise ValueError("rastermap_isort must contain at least one fitted ROI")
    if not np.issubdtype(rastermap_isort.dtype, np.integer):
        raise TypeError("rastermap_isort must contain integer ROI indices")
    if np.any(rastermap_isort < 0) or np.any(rastermap_isort >= n_neurons):
        raise IndexError("rastermap_isort contains an ROI outside the recording")
    if np.unique(rastermap_isort).size != rastermap_isort.size:
        raise ValueError("rastermap_isort must not contain duplicate ROI indices")

    fitted_order = rastermap_isort.astype(np.int64, copy=False)
    is_fitted = np.zeros(int(n_neurons), dtype=bool)
    is_fitted[fitted_order] = True
    appended_order = np.flatnonzero(~is_fitted)
    complete_order = np.concatenate((fitted_order, appended_order))
    return complete_order, fitted_order.size


def binned_spike_raster(
    spike_deconv: np.ndarray,
    fs: float,
    bin_seconds: float,
    boundary_ind: np.ndarray,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Build a time-binned raster from positive deconvolution samples.

    Bins restart at acquisition and state boundaries, so no display bin mixes
    discontinuous acquisitions or two brain states.  The return values are the
    activity-ranked raster, original-frame active-neuron counts, nominal bin
    width, neuron ordering, and center frame of every display bin.

    Processing is streamed one bin at a time to avoid allocating a second full
    ``n_neurons × n_frames`` boolean matrix for large recordings.
    """
    spike_deconv = np.asarray(spike_deconv)
    state = np.asarray(state)
    if spike_deconv.ndim != 2:
        raise ValueError("spike_deconv must be a neuron-by-frame matrix")
    n_neurons, n_frames = spike_deconv.shape
    if state.ndim != 1 or state.size != n_frames:
        raise ValueError("state must be one-dimensional and match the frame count")
    if (
        not np.isfinite(fs)
        or fs <= 0
        or not np.isfinite(bin_seconds)
        or bin_seconds <= 0
    ):
        raise ValueError("fs and bin_seconds must be finite and positive")

    bin_frames = max(1, int(round(bin_seconds * fs)))
    state_changes = np.flatnonzero(np.diff(state) != 0) + 1
    display_segments = ts.acquisition_segments(n_frames, boundary_ind, state_changes)
    bin_limits = [
        (start, min(start + bin_frames, segment_stop))
        for segment_start, segment_stop in display_segments
        for start in range(segment_start, segment_stop, bin_frames)
    ]

    raster = np.zeros((n_neurons, len(bin_limits)), dtype=np.uint8)
    active_counts = np.zeros(n_frames, dtype=np.int32)
    positive_counts = np.zeros(n_neurons, dtype=np.int32)
    bin_centers = np.empty(len(bin_limits), dtype=float)
    for bin_index, (start, stop) in enumerate(bin_limits):
        positive = spike_deconv[:, start:stop] > 0
        raster[:, bin_index] = np.any(positive, axis=1)
        active_counts[start:stop] = np.sum(positive, axis=0, dtype=np.int32)
        positive_counts += np.sum(positive, axis=1, dtype=np.int32)
        bin_centers[bin_index] = (start + stop - 1) / 2

    # A stable sort retains original row order for neurons tied on activity.
    order = np.argsort(-positive_counts, kind="stable")
    return raster[order], active_counts, bin_frames, order, bin_centers


def binned_dff_heatmaps(
    dff: np.ndarray,
    neuron_order: np.ndarray,
    fs: float,
    bin_seconds: float,
    boundary_ind: np.ndarray,
    state: np.ndarray,
) -> tuple[list[DFFHeatmapPanel], float, int]:
    """Build state/boundary-aligned, all-neuron ΔF/F heatmap panels.

    Rows follow ``neuron_order``.  Each display bin averages neighboring frames,
    then each neuron is translated by its median display-bin value; there is no
    per-neuron scaling or z-scoring.  Every neuron and frame contributes to one
    heatmap cell.
    """
    dff = np.asarray(dff)
    neuron_order = np.asarray(neuron_order)
    state = np.asarray(state)
    if dff.ndim != 2:
        raise ValueError("dff must be a neuron-by-frame matrix")
    n_neurons, n_frames = dff.shape
    if neuron_order.ndim != 1 or neuron_order.size != n_neurons:
        raise ValueError("neuron_order must contain one row index per neuron")
    if state.ndim != 1 or state.size != n_frames:
        raise ValueError("state must be one-dimensional and match the frame count")
    if n_frames == 0:
        raise ValueError("dff must contain at least one frame")
    if (
        not np.isfinite(fs)
        or fs <= 0
        or not np.isfinite(bin_seconds)
        or bin_seconds <= 0
    ):
        raise ValueError("fs and bin_seconds must be finite and positive")

    bin_frames = max(1, int(round(bin_seconds * fs)))
    state_changes = np.flatnonzero(np.diff(state) != 0) + 1
    display_segments = ts.acquisition_segments(n_frames, boundary_ind, state_changes)
    panels: list[DFFHeatmapPanel] = []

    for segment_start, segment_stop in display_segments:
        bin_limits = [
            (start, min(start + bin_frames, segment_stop))
            for start in range(segment_start, segment_stop, bin_frames)
        ]
        values = np.empty((n_neurons, len(bin_limits)), dtype=np.float32)
        for column, (start, stop) in enumerate(bin_limits):
            values[:, column] = np.nanmean(dff[neuron_order, start:stop], axis=1)
        panels.append(
            {
                "start_min": segment_start / fs / 60,
                "stop_min": segment_stop / fs / 60,
                "values": values,
            }
        )

    binned_session = np.concatenate([panel["values"] for panel in panels], axis=1)
    baseline = np.nanmedian(binned_session, axis=1)
    del binned_session
    for panel in panels:
        panel["values"] -= baseline[:, np.newaxis]

    # One robust scale keeps all temporal panels of a session comparable.  The
    # limit clips only colors during rendering, never the prepared values.
    color_limit = max(
        float(np.nanpercentile(np.abs(panel["values"]), 99)) for panel in panels
    )
    if not np.isfinite(color_limit) or color_limit <= 0:
        color_limit = 1.0
    return panels, color_limit, bin_frames


def shade_states(
    ax,
    view: ActivityView,
    alpha: float = 0.07,
    state_colors: Mapping[str, Any] | None = None,
) -> None:
    """Add faint categorical state spans behind a recorded-time axis."""
    colors = DEFAULT_STATE_COLORS if state_colors is None else state_colors
    for start, stop, code in ts.state_runs(np.asarray(view["state"])):
        label = view["codes"][code]
        ax.axvspan(
            start / view["fs"] / 60,
            stop / view["fs"] / 60,
            color=colors[label],
            alpha=alpha,
            lw=0,
            zorder=0,
        )


def mark_acquisition_boundaries(
    ax,
    view: ActivityView,
    annotate: bool = False,
) -> None:
    """Mark internal microscope acquisition breaks on a recorded-time axis."""
    for boundary_min in view["boundary_minutes"]:
        if not view["time_limits_min"][0] <= boundary_min <= view["time_limits_min"][1]:
            continue
        ax.axvline(boundary_min, color="0.2", ls=(0, (3, 2)), lw=0.8, zorder=3)
        if annotate:
            transform = blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(
                boundary_min,
                0.98,
                " acquisition break",
                rotation=90,
                ha="left",
                va="top",
                fontsize=7,
                color="0.25",
                transform=transform,
                clip_on=True,
                zorder=4,
            )


def plot_state_strip(
    ax,
    view: ActivityView,
    state_colors: Mapping[str, Any] | None = None,
    state_labels: Mapping[str, str] | None = None,
    short_labels: Mapping[str, str] | None = None,
) -> None:
    """Plot exact frame-wise brain states on an axis sharing recorded time."""
    colors = DEFAULT_STATE_COLORS if state_colors is None else state_colors
    labels = DEFAULT_STATE_LABELS if state_labels is None else state_labels
    short = DEFAULT_STATE_SHORT_LABELS if short_labels is None else short_labels

    all_states = [
        (code, label)
        for code, label in view["codes"].items()
        if np.any(np.isclose(view["state"], code))
    ]
    shown_start, shown_stop = visible_frame_range(view)
    shown_state = view["state"][shown_start:shown_stop]
    shown_states = [
        (code, label)
        for code, label in all_states
        if np.any(np.isclose(shown_state, code))
    ]

    state_index = np.full(view["n_frames"], -1, dtype=np.int8)
    for index, (code, _label) in enumerate(all_states):
        state_index[np.isclose(view["state"], code)] = index
    if np.any(state_index < 0):
        unknown = np.unique(view["state"][state_index < 0])
        raise ValueError(f"Unrecognized state codes: {unknown}")

    cmap = ListedColormap([colors[label] for _code, label in all_states])
    ax.imshow(
        state_index[np.newaxis, :],
        aspect="auto",
        interpolation="nearest",
        extent=(0, view["duration_min"], 0, 1),
        origin="lower",
        cmap=cmap,
        vmin=-0.5,
        vmax=len(all_states) - 0.5,
    )

    # Keep the strip readable by writing labels only inside sufficiently long
    # visible bouts.  The legend still identifies every short bout.
    shown_duration = view["time_limits_min"][1] - view["time_limits_min"][0]
    min_label_min = max(0.20, 0.045 * shown_duration)
    for start, stop, code in ts.state_runs(np.asarray(view["state"])):
        visible_start = max(start, shown_start)
        visible_stop = min(stop, shown_stop)
        width_min = (visible_stop - visible_start) / view["fs"] / 60
        if width_min < min_label_min:
            continue
        label = view["codes"][code]
        text_color = "black" if label in {"quiet_awake", "anesthesia"} else "white"
        ax.text(
            (visible_start + visible_stop) / 2 / view["fs"] / 60,
            0.5,
            short[label],
            ha="center",
            va="center",
            color=text_color,
            fontsize=7,
            clip_on=True,
        )

    handles = [
        Patch(facecolor=colors[label], label=labels[label])
        for _code, label in shown_states
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.005, 0.5),
        frameon=False,
        fontsize=8,
        handlelength=1.2,
    )
    ax.set_xlim(*view["time_limits_min"])
    ax.set_yticks([])
    ax.set_ylabel("state", rotation=0, ha="right", va="center")
    ax.set_xlabel("recorded time (min)")
    mark_acquisition_boundaries(ax, view)
    for spine in ax.spines.values():
        spine.set_visible(False)


def nice_scale_bar(target: float) -> float:
    """Choose a readable ``1/2/5 × 10ⁿ`` scale no larger than ``target``."""
    if not np.isfinite(target) or target <= 0:
        return 1.0
    power = 10.0 ** np.floor(np.log10(target))
    for multiplier in (5.0, 2.0, 1.0):
        candidate = multiplier * power
        if candidate <= target:
            return float(candidate)
    return float(power)


def time_window_label(view: ActivityView) -> str:
    """Return a label for a full sequence or selected recorded-time window."""
    start, stop = view["time_limits_min"]
    if np.isclose(start, 0) and np.isclose(stop, view["duration_min"]):
        return "full recorded sequence"
    return f"recorded minutes {start:g}–{stop:g}"


def visible_frame_range(view: ActivityView) -> tuple[int, int]:
    """Return the half-open original-frame range visible in a minute window."""
    start = max(0, int(np.floor(view["time_limits_min"][0] * 60 * view["fs"])))
    stop = min(
        view["n_frames"],
        int(np.ceil(view["time_limits_min"][1] * 60 * view["fs"])),
    )
    return start, stop


def plot_stacked_signal(
    ax,
    view: ActivityView,
    signal: np.ndarray,
    *,
    signal_label: str,
    scale_unit: str,
    spacing: float | None = None,
    line_width: float = 0.27,
    color: Any = "0.08",
    annotate_boundaries: bool = True,
) -> None:
    """Plot row-aligned, median-centered signals with common vertical spacing.

    The function is signal-agnostic so the same selected neuron rows can be
    compared without changing their order. ``spacing`` translates traces
    vertically but never rescales their amplitudes; signals with different
    units should therefore use separate spacing values and scale bars.
    """
    signal = np.asarray(signal)
    trace_ids = view["trace_ids"]
    if signal.ndim != 2:
        raise ValueError("signal must be a neuron-by-frame matrix")
    if trace_ids is None:
        raise ValueError("trace_ids are required for a stacked-signal plot")
    trace_ids = np.asarray(trace_ids)
    if trace_ids.ndim != 1 or trace_ids.size != signal.shape[0]:
        raise ValueError("trace_ids must contain one row ID per signal trace")
    if signal.shape[0] == 0:
        raise ValueError("signal must contain at least one neuron")
    if signal.shape[1] != view["n_frames"]:
        raise ValueError("signal frame count must match the activity view")

    time_min = np.arange(view["n_frames"]) / view["fs"] / 60
    centered = signal - np.nanmedian(signal, axis=1, keepdims=True)
    if spacing is None:
        q01, q99 = np.nanpercentile(centered, (1, 99))
        robust_range = q99 - q01
        if not np.isfinite(robust_range) or robust_range <= 0:
            finite_nonzero = np.abs(centered[np.isfinite(centered) & (centered != 0)])
            robust_range = (
                float(np.nanpercentile(finite_nonzero, 99))
                if finite_nonzero.size
                else 0.0
            )
        spacing = (
            1.15 * robust_range
            if np.isfinite(robust_range) and robust_range > 0
            else 1.0
        )
    elif not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")
    if not np.isfinite(line_width) or line_width <= 0:
        raise ValueError("line_width must be finite and positive")
    # Negative offsets put the first neuron at the top while positive signal
    # transients continue to point upward on the page.
    offsets = -np.arange(centered.shape[0]) * spacing
    visible_start, visible_stop = visible_frame_range(view)

    shade_states(ax, view)
    for trace, offset in zip(centered, offsets):
        for start, stop in view["acquisition_segments"]:
            start = max(start, visible_start)
            stop = min(stop, visible_stop)
            if stop <= start:
                continue
            ax.plot(
                time_min[start:stop],
                trace[start:stop] + offset,
                color=color,
                lw=line_width,
                rasterized=True,
                zorder=2,
            )

    tick_rows = np.unique(
        np.linspace(0, centered.shape[0] - 1, min(11, centered.shape[0])).astype(int)
    )
    ax.set_yticks(offsets[tick_rows])
    ax.set_yticklabels(trace_ids[tick_rows], fontsize=7)
    ax.set_ylim(offsets[-1] - spacing, spacing)
    ax.set_xlim(*view["time_limits_min"])
    ax.set_ylabel(f"neuron row (0-based)\n({signal_label}, offset)")
    ax.tick_params(axis="x", labelbottom=False)

    scale = nice_scale_bar(0.6 * spacing)
    y0 = offsets[min(2, centered.shape[0] - 1)]
    transform = blended_transform_factory(ax.transAxes, ax.transData)
    ax.plot(
        [0.985, 0.985],
        [y0, y0 + scale],
        color="black",
        lw=1.5,
        transform=transform,
        clip_on=False,
        zorder=4,
    )
    mark_acquisition_boundaries(ax, view, annotate=annotate_boundaries)
    ax.text(
        0.975,
        y0 + scale / 2,
        f"{scale:g} {scale_unit}",
        ha="right",
        va="center",
        fontsize=8,
        transform=transform,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1},
        zorder=4,
    )


def plot_stacked_dff(
    ax,
    view: ActivityView,
    session_titles: Mapping[str, str] | None = None,
    spacing: float | None = None,
    line_width: float = 0.27,
    annotate_boundaries: bool = True,
) -> None:
    """Plot raw, median-centered ΔF/F traces with common vertical spacing."""
    titles = DEFAULT_SESSION_TITLES if session_titles is None else session_titles
    plot_stacked_signal(
        ax,
        view,
        view["dff"],
        signal_label="raw ΔF/F",
        scale_unit="ΔF/F",
        spacing=spacing,
        line_width=line_width,
        annotate_boundaries=annotate_boundaries,
    )
    ax.set_title(
        f"{view['name']} · {titles[view['data_info']]}\n"
        f"random {np.asarray(view['dff']).shape[0]} of "
        f"{view['n_neurons']:,} neurons · {time_window_label(view)}"
    )


def plot_all_dff_heatmap(
    ax,
    view: ActivityView,
    session_titles: Mapping[str, str] | None = None,
):
    """Plot temporally binned ΔF/F for every activity-ranked neuron."""
    titles = DEFAULT_SESSION_TITLES if session_titles is None else session_titles
    limit = view["dff_color_limit"]
    image = None
    for panel in view["dff_heatmaps"]:
        if (
            panel["stop_min"] <= view["time_limits_min"][0]
            or panel["start_min"] >= view["time_limits_min"][1]
        ):
            continue
        image = ax.imshow(
            panel["values"],
            aspect="auto",
            origin="upper",
            extent=(panel["start_min"], panel["stop_min"], view["n_neurons"], 0),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="hanning",
            interpolation_stage="rgba",
        )

    ax.set_xlim(*view["time_limits_min"])
    ax.set_ylim(view["n_neurons"], 0)
    ax.set_yticks([0, view["n_neurons"] - 1])
    ax.set_yticklabels(["1", f"{view['n_neurons']:,}"])
    ax.set_ylabel("all neurons\n(activity-ranked)")
    ax.set_title(
        f"{view['name']} · {titles[view['data_info']]}\n"
        f"all {view['n_neurons']:,} neurons · {time_window_label(view)}"
    )
    ax.tick_params(axis="x", labelbottom=False)
    mark_acquisition_boundaries(ax, view, annotate=True)
    actual_bin_seconds = view["dff_bin_frames"] / view["fs"]
    ax.text(
        0.995,
        0.015,
        f"display bins ≤ {actual_bin_seconds:.2f} s",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85, "pad": 2},
        zorder=3,
    )
    return image


def plot_population_fraction(
    ax,
    view: ActivityView,
    session_titles: Mapping[str, str] | None = None,
) -> None:
    """Plot the smoothed original-resolution active-neuron fraction."""
    titles = DEFAULT_SESSION_TITLES if session_titles is None else session_titles
    time_min = np.arange(view["n_frames"]) / view["fs"] / 60
    visible_start, visible_stop = visible_frame_range(view)
    shade_states(ax, view, alpha=0.10)
    for start, stop in view["acquisition_segments"]:
        start = max(start, visible_start)
        stop = min(stop, visible_stop)
        if stop <= start:
            continue
        ax.plot(
            time_min[start:stop],
            view["population_active_fraction"][start:stop],
            color="0.08",
            lw=0.8,
            zorder=2,
        )
    ax.set_xlim(*view["time_limits_min"])
    ax.set_ylim(bottom=0)
    ax.set_ylabel("active neurons\n(%, 5-s mean)")
    ax.tick_params(axis="x", labelbottom=False)
    ax.grid(axis="y", color="0.85", lw=0.5)
    ax.set_title(
        f"{view['name']} · {titles[view['data_info']]} · " f"{time_window_label(view)}"
    )
    mark_acquisition_boundaries(ax, view, annotate=True)


def plot_spike_raster(ax, view: ActivityView) -> None:
    """Plot a compact, time-binned raster containing every neuron."""
    shade_states(ax, view, alpha=0.10)
    visible_bins = np.flatnonzero(
        (view["bin_centers_min"] >= view["time_limits_min"][0])
        & (view["bin_centers_min"] <= view["time_limits_min"][1])
    )
    neuron_rows, local_time_bins = np.nonzero(view["raster"][:, visible_bins])
    ax.scatter(
        view["bin_centers_min"][visible_bins[local_time_bins]],
        neuron_rows,
        s=0.15,
        marker=".",
        color="0.02",
        alpha=0.78,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    ax.set_xlim(*view["time_limits_min"])
    ax.set_ylim(view["n_neurons"], 0)
    ax.set_yticks([0, view["n_neurons"] - 1])
    ax.set_yticklabels(["1", f"{view['n_neurons']:,}"])
    ax.set_ylabel("all neurons\n(activity-ranked)")
    ax.tick_params(axis="x", labelbottom=False)
    mark_acquisition_boundaries(ax, view)
    actual_bin_seconds = view["bin_frames"] / view["fs"]
    ax.text(
        0.995,
        0.015,
        f"display bins ≤ {actual_bin_seconds:.2f} s",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85, "pad": 2},
        zorder=3,
    )


def plot_brain_region_spike_raster(ax, view: ActivityView) -> None:
    """Plot every neuron's binary events grouped by exact atlas region.

    The compact raster is stored in activity-ranked order.  This function maps
    those rows back to original ROI indices, then applies the cortical-region
    order prepared for the view. Activity rank is retained within each region.
    """
    activity_order = np.asarray(view["neuron_order"])
    region_order = np.asarray(view["brain_region_order"])
    n_neurons = view["n_neurons"]
    for name, order in (
        ("neuron_order", activity_order),
        ("brain_region_order", region_order),
    ):
        if (
            order.ndim != 1
            or order.size != n_neurons
            or not np.issubdtype(order.dtype, np.integer)
            or np.unique(order).size != n_neurons
            or np.any(order < 0)
            or np.any(order >= n_neurons)
        ):
            raise ValueError(f"{name} must be a permutation of every recorded ROI")

    activity_row_for_roi = np.empty(n_neurons, dtype=np.int64)
    activity_row_for_roi[activity_order] = np.arange(n_neurons)
    region_raster = view["raster"][activity_row_for_roi[region_order]]

    shade_states(ax, view, alpha=0.10)
    visible_bins = np.flatnonzero(
        (view["bin_centers_min"] >= view["time_limits_min"][0])
        & (view["bin_centers_min"] <= view["time_limits_min"][1])
    )
    neuron_rows, local_time_bins = np.nonzero(region_raster[:, visible_bins])
    ax.scatter(
        view["bin_centers_min"][visible_bins[local_time_bins]],
        neuron_rows,
        s=0.15,
        marker=".",
        color="0.02",
        alpha=0.78,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    ax.set_xlim(*view["time_limits_min"])
    ax.set_ylim(n_neurons, 0)
    ax.set_yticks([0, n_neurons - 1])
    ax.set_yticklabels(["1", f"{n_neurons:,}"])
    ax.set_ylabel("all neurons\n(grouped by atlas region)")
    ax.tick_params(axis="x", labelbottom=False)
    mark_acquisition_boundaries(ax, view)

    regions = brain_region_labels(view["brain_regions"])[region_order]
    boundaries = np.flatnonzero(regions[1:] != regions[:-1]) + 1
    for boundary in boundaries:
        ax.axhline(boundary - 0.5, color="0.55", lw=0.45, zorder=1)

    actual_bin_seconds = view["bin_frames"] / view["fs"]
    ax.text(
        0.995,
        0.015,
        f"activity-ranked within each region; display bins ≤ {actual_bin_seconds:.2f} s",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85, "pad": 2},
        zorder=3,
    )


def plot_rastermap_spike_raster(ax, view: ActivityView) -> None:
    """Plot selected or complete binary events with rows ordered by Rastermap.

    The compact raster in ``view`` is stored in activity-ranked order.  This
    function maps those rows back to original ROI indices and then applies the
    all-session ``rastermap_isort`` permutation.  An active-neuron analysis
    displays only fitted selected rows.  The legacy all-neuron mode appends
    mathematically unfittable rows below a separator.
    """
    rastermap_isort = view["rastermap_isort"]
    valid_rows = view["rastermap_valid_rows"]
    if rastermap_isort is None or valid_rows is None:
        raise ValueError("This activity view does not contain a Rastermap fit")

    selected_only = bool(view.get("rastermap_display_selected_only", False))
    if selected_only:
        complete_roi_order = np.asarray(rastermap_isort, dtype=np.int64)
        n_fitted = complete_roi_order.size
    else:
        complete_roi_order, n_fitted = rastermap_display_order(
            view["n_neurons"],
            rastermap_isort,
        )
    if valid_rows.size != n_fitted or not np.array_equal(
        np.sort(valid_rows),
        np.sort(rastermap_isort),
    ):
        raise ValueError("Rastermap order and fitted-row metadata are inconsistent")

    activity_order = np.asarray(view["neuron_order"])
    if (
        activity_order.ndim != 1
        or activity_order.size != view["n_neurons"]
        or not np.issubdtype(activity_order.dtype, np.integer)
        or np.unique(activity_order).size != view["n_neurons"]
        or np.any(activity_order < 0)
        or np.any(activity_order >= view["n_neurons"])
    ):
        raise ValueError("neuron_order must be a permutation of every recorded ROI")
    activity_row_for_roi = np.empty(view["n_neurons"], dtype=np.int64)
    activity_row_for_roi[activity_order] = np.arange(view["n_neurons"])
    rastermap_raster = view["raster"][activity_row_for_roi[complete_roi_order]]

    shade_states(ax, view, alpha=0.10)
    visible_bins = np.flatnonzero(
        (view["bin_centers_min"] >= view["time_limits_min"][0])
        & (view["bin_centers_min"] <= view["time_limits_min"][1])
    )
    neuron_rows, local_time_bins = np.nonzero(rastermap_raster[:, visible_bins])
    ax.scatter(
        view["bin_centers_min"][visible_bins[local_time_bins]],
        neuron_rows,
        s=0.15,
        marker=".",
        color="0.02",
        alpha=0.78,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    ax.set_xlim(*view["time_limits_min"])
    n_display_neurons = complete_roi_order.size
    ax.set_ylim(n_display_neurons, 0)
    ax.set_yticks([0, n_display_neurons - 1])
    ax.set_yticklabels(["1", f"{n_display_neurons:,}"])
    ax.set_ylabel(
        "active selected neurons\n(Rastermap order)"
        if selected_only
        else "all neurons\n(Rastermap order)"
    )
    ax.tick_params(axis="x", labelbottom=False)
    mark_acquisition_boundaries(ax, view)

    n_unfitted = view["n_neurons"] - n_fitted
    if selected_only:
        order_text = (
            f"{n_fitted:,}/{view['n_neurons']:,} active selected rows "
            "Rastermap-sorted"
        )
    elif n_unfitted:
        ax.axhline(n_fitted - 0.5, color="tab:red", lw=0.8, ls=(0, (3, 2)))
        order_text = (
            f"first {n_fitted:,} rows Rastermap-sorted; "
            f"bottom {n_unfitted:,} non-finite/constant rows appended"
        )
    else:
        order_text = f"all {n_fitted:,} rows Rastermap-sorted"
    actual_bin_seconds = view["bin_frames"] / view["fs"]
    ax.text(
        0.995,
        0.015,
        f"{order_text}; display bins ≤ {actual_bin_seconds:.2f} s",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85, "pad": 2},
        zorder=3,
    )


def plot_rastermap_embedding(
    ax,
    view: ActivityView,
    vmin: float = 0.0,
    vmax: float = 1.5,
):
    """Plot Rastermap's normalized superneuron representation.

    Rastermap determines the row order from the explicitly selected neurons in
    the complete session. Rows shown here average adjacent neurons *after*
    sorting; they are never a random sample.
    """
    values = view["rastermap_X_embedding"]
    valid_rows = view["rastermap_valid_rows"]
    parameters = view["rastermap_parameters"]
    if values is None or valid_rows is None or parameters is None:
        raise ValueError("This activity view does not contain a Rastermap fit")
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("Rastermap color limits must satisfy finite vmin < vmax")

    n_superneurons = values.shape[0]
    image = ax.imshow(
        values,
        aspect="auto",
        origin="upper",
        extent=(0, view["rastermap_stop_min"], n_superneurons, 0),
        cmap="gray_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_xlim(*view["time_limits_min"])
    ax.set_ylim(n_superneurons, 0)
    ax.set_yticks([0, max(0, n_superneurons - 1)])
    ax.set_yticklabels(["1", f"{n_superneurons:,}"])
    superneuron_size = int(parameters["superneuron_size"])
    ax.set_ylabel(
        "Rastermap superneurons\n" f"(≤{superneuron_size} adjacent neurons each)"
    )
    ax.tick_params(axis="x", labelbottom=False)
    mark_acquisition_boundaries(ax, view)

    n_valid = valid_rows.size
    selected_only = bool(view.get("rastermap_display_selected_only", False))
    if selected_only:
        selection_text = str(parameters.get("selection_label", "active selected"))
        neuron_text = (
            f"{n_valid:,}/{view['n_neurons']:,} {selection_text} neurons fitted"
        )
    elif n_valid == view["n_neurons"]:
        neuron_text = f"all {n_valid:,} recorded neurons fitted"
    else:
        neuron_text = (
            f"{n_valid:,}/{view['n_neurons']:,} neurons fitted; "
            f"{view['n_neurons'] - n_valid:,} non-finite/constant omitted"
        )
    lag_seconds = parameters["time_lag_window"] * parameters["time_bin"] / view["fs"]
    ax.set_title(
        f"{view['name']} · {neuron_text} · {time_window_label(view)}\n"
        f"Rastermap {view['rastermap_version']} · "
        f"clusters={parameters['n_clusters']}, PCs={parameters['n_PCs']}, "
        f"locality={parameters['locality']:g}, lag={parameters['time_lag_window']} "
        f"samples ({lag_seconds:.2f} s)"
    )
    cache_label = "cached fit" if view["rastermap_cached"] else "new fit"
    ax.text(
        0.995,
        0.015,
        f"{cache_label}; mean_time={parameters['mean_time']}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85, "pad": 2},
        zorder=3,
    )
    return image


def plot_spatial_modules(
    ax,
    coords: np.ndarray,
    communities: np.ndarray,
    title: str | None = None,
    node_size: float = 18,
    show_counts: bool = True,
    color_cycle: int = 20,
    edge_linewidth: float = 0.2,
):
    """Plot nodes at spatial coordinates, colored by module assignment.

    This shared axis-level helper is used for both single-cell and coarse-grained
    module maps.  It deliberately leaves figure layout and scientific annotations
    to the calling tutorial.
    """
    coords = np.asarray(coords)
    communities = np.asarray(communities)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n_nodes, 2)")
    if communities.ndim != 1 or communities.size != coords.shape[0]:
        raise ValueError("communities must contain one label per coordinate")
    if color_cycle <= 0:
        raise ValueError("color_cycle must be positive")

    # Map arbitrary numeric labels to a bounded categorical palette while
    # retaining the original module labels for counts and downstream analysis.
    color_ids = communities.astype(int, copy=False) % color_cycle
    collection = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=color_ids,
        cmap="tab20",
        s=node_size,
        edgecolor="k",
        lw=edge_linewidth,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        if show_counts:
            title = (
                f"{title}\n{np.unique(communities).size} modules, "
                f"{communities.size} nodes"
            )
        ax.set_title(title)
    return collection


__all__ = [
    "ActivityView",
    "CORTICAL_REGION_COLORS",
    "CORTICAL_REGION_NAMES",
    "DEFAULT_SESSION_TITLES",
    "DEFAULT_STATE_COLORS",
    "DEFAULT_STATE_LABELS",
    "DEFAULT_STATE_SHORT_LABELS",
    "DFFHeatmapPanel",
    "binned_dff_heatmaps",
    "binned_spike_raster",
    "cortical_region_labels",
    "cortical_region_legend_handles",
    "BRAIN_REGION_COLORS",
    "brain_region_labels",
    "brain_region_legend_handles",
    "brain_region_order",
    "display_cortical_region",
    "mark_acquisition_boundaries",
    "nice_scale_bar",
    "plot_all_dff_heatmap",
    "display_brain_region",
    "plot_cortical_region_strip",
    "plot_brain_region_spike_raster",
    "plot_brain_region_strip",
    "plot_population_fraction",
    "plot_rastermap_embedding",
    "plot_rastermap_spike_raster",
    "plot_spatial_modules",
    "plot_spike_raster",
    "plot_stacked_dff",
    "plot_stacked_signal",
    "plot_state_strip",
    "rastermap_display_order",
    "resolve_time_limits",
    "select_trace_neurons",
    "shade_states",
    "time_window_label",
    "visible_frame_range",
]
