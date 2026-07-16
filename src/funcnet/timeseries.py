"""Reusable operations on frame indices and neural time series.

This module contains the small, data-independent transformations shared by the
tutorials: splitting frame selections into windows or contiguous bouts,
respecting microscope acquisition breaks, circular-shift nulls, and simple
smoothing.  Keeping them here avoids a generic ``utils`` module and makes their
time-axis assumptions easy to find and test.

The functions operate on NumPy arrays and do not know about
:class:`funcnet.dataio.Recording`.  Recording-specific lookup remains in
``dataio``; callers pass the resulting frame indices or activity matrix here.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np


def frame_windows(
    frames: np.ndarray,
    width: int,
    max_windows: int | None = None,
) -> list[np.ndarray]:
    """Split an ordered frame-index sequence into full, fixed-width windows.

    Parameters
    ----------
    frames
        One-dimensional frame indices in the order in which they should be
        analysed.  The indices do not have to be contiguous in original time.
    width
        Number of selected frames in each returned window.  An incomplete tail
        is deliberately omitted so every network uses the same sample count.
    max_windows
        Optional cap on the number of windows.  ``None`` returns every full
        window; zero returns an empty list.

    Notes
    -----
    A window follows the order of ``frames``.  If preserving separate temporal
    bouts matters (for example for a shuffle null), use :func:`contiguous_runs`
    inside each returned selection.
    """
    frames = np.asarray(frames)
    if frames.ndim != 1:
        raise ValueError("frames must be a one-dimensional array")
    if isinstance(width, (bool, np.bool_)) or not isinstance(width, (int, np.integer)):
        raise TypeError("width must be an integer")
    if width <= 0:
        raise ValueError("width must be positive")
    if max_windows is not None:
        if isinstance(max_windows, (bool, np.bool_)) or not isinstance(
            max_windows, (int, np.integer)
        ):
            raise TypeError("max_windows must be an integer or None")
        if max_windows < 0:
            raise ValueError("max_windows must be non-negative")

    n_windows = frames.size // width
    if max_windows is not None:
        n_windows = min(n_windows, int(max_windows))
    return [frames[i * width : (i + 1) * width] for i in range(n_windows)]


def contiguous_runs(frames: np.ndarray) -> list[np.ndarray]:
    """Return column positions belonging to contiguous bouts in original time.

    ``frames`` commonly contains several state bouts concatenated together.  A
    returned array indexes positions *within that input sequence*, rather than
    containing the original frame numbers.  This distinction lets callers use
    the result directly to slice an activity matrix whose columns correspond to
    ``frames``.
    """
    frames = np.asarray(frames)
    if frames.ndim != 1:
        raise ValueError("frames must be a one-dimensional array")
    if frames.size == 0:
        return []

    cuts = np.flatnonzero(np.diff(frames) != 1) + 1
    return [run for run in np.split(np.arange(frames.size), cuts) if run.size]


def circular_shuffle(
    activity: np.ndarray,
    frames: np.ndarray,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> np.ndarray:
    """Circularly shift each neuron independently within contiguous bouts.

    This null transformation preserves every neuron/bout's values and temporal
    shape while disrupting most cross-neuron timing.  Restricting each roll to a
    contiguous bout prevents samples from crossing gaps created when separate
    state epochs are concatenated.

    Parameters
    ----------
    activity
        ``(n_neurons, n_selected_frames)`` activity matrix.
    frames
        Original frame number for every activity column.
    rng
        NumPy ``Generator`` or legacy ``RandomState``.  Supplying one makes the
        shuffle reproducible; otherwise a fresh generator is used.
    """
    activity = np.asarray(activity)
    frames = np.asarray(frames)
    if activity.ndim != 2:
        raise ValueError("activity must be a two-dimensional neuron-by-frame matrix")
    if frames.ndim != 1 or frames.size != activity.shape[1]:
        raise ValueError("frames must be one-dimensional and match activity columns")

    rng = np.random.default_rng() if rng is None else rng
    runs = contiguous_runs(frames)
    shuffled = np.array(activity, copy=True)
    for neuron in range(activity.shape[0]):
        for run in runs:
            if run.size <= 1:
                continue
            # ``Generator`` calls this method ``integers``; ``RandomState`` uses
            # ``randint``.  Supporting both preserves seeded tutorial behavior.
            if hasattr(rng, "integers"):
                lag = int(rng.integers(1, run.size))
            else:
                lag = int(rng.randint(1, run.size))
            shuffled[neuron, run] = np.roll(activity[neuron, run], lag)
    return shuffled


def state_runs(state: np.ndarray) -> Iterator[tuple[int, int, float]]:
    """Yield ``(start, stop, code)`` for contiguous runs in a state vector."""
    state = np.asarray(state)
    if state.ndim != 1:
        raise ValueError("state must be a one-dimensional array")
    if state.size == 0:
        return

    changes = np.flatnonzero(np.diff(state) != 0) + 1
    starts = np.r_[0, changes]
    stops = np.r_[changes, state.size]
    for start, stop in zip(starts, stops):
        yield int(start), int(stop), float(state[start])


def acquisition_segments(
    n_frames: int,
    boundary_ind: np.ndarray,
    extra_splits: Iterable[int] = (),
) -> list[tuple[int, int]]:
    """Return half-open frame segments that never cross an acquisition break.

    Dataset ``boundary_ind`` values identify the final frame before a microscope
    break, so each corresponding segment stops at ``boundary + 1``.  Values in
    ``extra_splits`` already denote half-open split positions; state-transition
    indices can therefore be passed without adjustment.
    """
    if isinstance(n_frames, (bool, np.bool_)) or not isinstance(
        n_frames, (int, np.integer)
    ):
        raise TypeError("n_frames must be an integer")
    if n_frames < 0:
        raise ValueError("n_frames must be non-negative")
    if n_frames == 0:
        return []

    stops = {
        int(boundary) + 1
        for boundary in np.asarray(boundary_ind).ravel()
        if 0 <= int(boundary) < n_frames
    }
    stops.update(int(split) for split in extra_splits if 0 < int(split) < n_frames)
    stops.add(int(n_frames))

    segments: list[tuple[int, int]] = []
    start = 0
    for stop in sorted(stops):
        if stop > start:
            segments.append((start, stop))
        start = stop
    return segments


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    """Centered moving average without zero-padding artifacts at the edges."""
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("values must be a one-dimensional array")
    if values.size == 0:
        return np.asarray(values, dtype=float)
    if isinstance(width, (bool, np.bool_)) or not isinstance(width, (int, np.integer)):
        raise TypeError("width must be an integer")
    # ``np.convolve(..., mode="same")`` returns the longer operand's length.
    # Clamp here so a short acquisition segment always retains its input shape.
    width = min(values.size, max(1, int(width)))

    kernel = np.ones(width, dtype=float)
    total = np.convolve(values, kernel, mode="same")
    weight = np.convolve(np.ones_like(values, dtype=float), kernel, mode="same")
    return total / weight


def segmented_moving_average(
    values: np.ndarray,
    width: int,
    boundary_ind: np.ndarray,
) -> np.ndarray:
    """Smooth each acquisition segment without averaging across a break."""
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("values must be a one-dimensional array")
    if values.size == 0:
        return np.asarray(values, dtype=float)

    smoothed = np.empty(values.size, dtype=float)
    for start, stop in acquisition_segments(values.size, boundary_ind):
        smoothed[start:stop] = moving_average(values[start:stop], width)
    return smoothed


__all__ = [
    "acquisition_segments",
    "circular_shuffle",
    "contiguous_runs",
    "frame_windows",
    "moving_average",
    "segmented_moving_average",
    "state_runs",
]
