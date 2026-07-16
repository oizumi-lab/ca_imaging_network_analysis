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
from typing import Any, TypedDict

import numpy as np
from matplotlib.colors import ListedColormap
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


def plot_stacked_dff(
    ax,
    view: ActivityView,
    session_titles: Mapping[str, str] | None = None,
) -> None:
    """Plot raw, median-centered ΔF/F traces with common vertical spacing."""
    titles = DEFAULT_SESSION_TITLES if session_titles is None else session_titles
    time_min = np.arange(view["n_frames"]) / view["fs"] / 60
    centered = view["dff"] - np.nanmedian(view["dff"], axis=1, keepdims=True)
    q01, q99 = np.nanpercentile(centered, (1, 99))
    robust_range = q99 - q01
    spacing = (
        1.15 * robust_range if np.isfinite(robust_range) and robust_range > 0 else 1.0
    )
    # Negative offsets put the first neuron at the top while positive calcium
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
                color="0.08",
                lw=0.27,
                rasterized=True,
                zorder=2,
            )

    tick_rows = np.unique(
        np.linspace(0, centered.shape[0] - 1, min(11, centered.shape[0])).astype(int)
    )
    ax.set_yticks(offsets[tick_rows])
    ax.set_yticklabels(view["trace_ids"][tick_rows], fontsize=7)
    ax.set_ylim(offsets[-1] - spacing, spacing)
    ax.set_xlim(*view["time_limits_min"])
    ax.set_ylabel("neuron row (0-based)\n(raw ΔF/F, offset)")
    ax.set_title(
        f"{view['name']} · {titles[view['data_info']]}\n"
        f"random {centered.shape[0]} of {view['n_neurons']:,} neurons · "
        f"{time_window_label(view)}"
    )
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
    mark_acquisition_boundaries(ax, view, annotate=True)
    ax.text(
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
    "DEFAULT_SESSION_TITLES",
    "DEFAULT_STATE_COLORS",
    "DEFAULT_STATE_LABELS",
    "DEFAULT_STATE_SHORT_LABELS",
    "DFFHeatmapPanel",
    "binned_dff_heatmaps",
    "binned_spike_raster",
    "mark_acquisition_boundaries",
    "nice_scale_bar",
    "plot_all_dff_heatmap",
    "plot_population_fraction",
    "plot_spatial_modules",
    "plot_spike_raster",
    "plot_stacked_dff",
    "plot_state_strip",
    "resolve_time_limits",
    "select_trace_neurons",
    "shade_states",
    "time_window_label",
    "visible_frame_range",
]
