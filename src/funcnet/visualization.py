"""Focused plotting helpers used by the modularity hands-on.

Display binning in this module is only for visualization. Network estimation
uses the unbinned smoothed-deconvolution signals in :mod:`funcnet.network`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

from . import timeseries as ts


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

# Exact layer-collapsed Allen-atlas categories used throughout the course.
# Related areas share color families: motor greens, somatosensory warm colors,
# retrosplenial blues, and visual purples.
CORTICAL_REGION_COLORS = {
    "MOs": "#005a32",
    "MOp": "#41ab5d",
    "SSp-ll": "#8c2d04",
    "SSp-ul": "#e6550d",
    "SSp-un": "#cb181d",
    "SSp-bfd": "#fb6a4a",
    "SSp-tr": "#f768a1",
    "SSp-m": "#dd3497",
    "SSp-n": "#ae017e",
    "RSPagl": "#08306b",
    "RSPd": "#2b8cbe",
    "VISa": "#3f007d",
    "VISam": "#54278f",
    "VISp": "#6a51a3",
    "VISpm": "#807dba",
    "VISrl": "#9e9ac8",
    "Unassigned": "#737373",
    "Other": "#666666",
    "Unknown": "#bdbdbd",
}
CORTICAL_REGION_NAMES = {
    "MOs": "Secondary motor area",
    "MOp": "Primary motor area",
    "SSp-ll": "Primary somatosensory area, lower limb",
    "SSp-ul": "Primary somatosensory area, upper limb",
    "SSp-un": "Primary somatosensory area, unassigned",
    "SSp-bfd": "Primary somatosensory area, barrel field",
    "SSp-tr": "Primary somatosensory area, trunk",
    "SSp-m": "Primary somatosensory area, mouth",
    "SSp-n": "Primary somatosensory area, nose",
    "RSPagl": "Retrosplenial area, lateral agranular part",
    "RSPd": "Retrosplenial area, dorsal part",
    "VISa": "Anterior visual area",
    "VISam": "Anteromedial visual area",
    "VISp": "Primary visual area",
    "VISpm": "Posteromedial visual area",
    "VISrl": "Rostrolateral visual area",
    "Unassigned": "Atlas root (no specific cortical-area assignment)",
    "Other": "Other valid atlas area",
    "Unknown": "Missing atlas label",
}


def display_cortical_region(atlas_label: object) -> str:
    """Map one source atlas label to a layer-collapsed display category."""
    if atlas_label is None:
        return "Unknown"
    if isinstance(atlas_label, bytes):
        atlas_label = atlas_label.decode("utf-8", errors="replace")
    text = str(atlas_label).strip()
    if text in {"Unassigned", "Other", "Unknown"}:
        return text
    if text.casefold() in {"", "nan", "none", "unknown", "unassigned", "na", "n/a"}:
        return "Unknown"
    area = text.removesuffix("2/3")
    if area.casefold() == "root":
        return "Unassigned"
    if area in CORTICAL_REGION_COLORS and area not in {
        "Unassigned",
        "Other",
        "Unknown",
    }:
        return area
    return "Other"


def cortical_region_labels(
    atlas_labels: Sequence[object] | np.ndarray,
) -> np.ndarray:
    """Return one displayed cortical-area category per neuron."""
    atlas = np.asarray(atlas_labels, dtype=object)
    if atlas.ndim != 1:
        raise ValueError("atlas_labels must be a one-dimensional neuron-aligned array")
    if atlas.size == 0:
        raise ValueError("atlas_labels must contain at least one label")
    return np.asarray([display_cortical_region(label) for label in atlas], dtype=object)


def cortical_region_legend_handles(
    regions: Sequence[str] | np.ndarray | None = None,
    *,
    unabridged: bool = False,
) -> list[Line2D]:
    """Build point-style legend handles in the shared anatomical order."""
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


class ActivityView(TypedDict):
    """Minimum recorded-time information shared by state/timeline plots."""

    n_frames: int
    fs: float
    duration_min: float
    time_limits_min: tuple[float, float]
    state: np.ndarray
    codes: dict[float, str]
    boundary_minutes: list[float]


def _visible_frame_range(view: ActivityView) -> tuple[int, int]:
    start = max(0, int(np.floor(view["time_limits_min"][0] * 60 * view["fs"])))
    stop = min(
        view["n_frames"],
        int(np.ceil(view["time_limits_min"][1] * 60 * view["fs"])),
    )
    return start, stop


def binned_spike_raster(
    spike_deconv: np.ndarray,
    fs: float,
    bin_seconds: float,
    boundary_ind: np.ndarray,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Build a time-binned raster from positive deconvolution samples.

    Bins restart at acquisition and state boundaries, so no display bin mixes
    discontinuous acquisitions or two brain states. The returned rows are
    activity-ranked using a stable sort.
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

    order = np.argsort(-positive_counts, kind="stable")
    return raster[order], active_counts, bin_frames, order, bin_centers


def shade_states(
    ax,
    view: ActivityView,
    alpha: float = 0.05,
    state_colors: Mapping[str, Any] | None = None,
) -> None:
    """Add faint state-colored spans behind a recorded-time axis."""
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
    shown_start, shown_stop = _visible_frame_range(view)
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
    """Plot nodes at spatial coordinates, colored by module assignment."""
    coords = np.asarray(coords)
    communities = np.asarray(communities)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n_nodes, 2)")
    if communities.ndim != 1 or communities.size != coords.shape[0]:
        raise ValueError("communities must contain one label per coordinate")
    if color_cycle <= 0:
        raise ValueError("color_cycle must be positive")

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
    "DEFAULT_STATE_COLORS",
    "DEFAULT_STATE_LABELS",
    "DEFAULT_STATE_SHORT_LABELS",
    "binned_spike_raster",
    "cortical_region_labels",
    "cortical_region_legend_handles",
    "display_cortical_region",
    "mark_acquisition_boundaries",
    "plot_spatial_modules",
    "plot_state_strip",
    "shade_states",
]
