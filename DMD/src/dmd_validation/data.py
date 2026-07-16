"""Selective HDF5 access and deterministic legal-window construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RecordingMetadata:
    name: str
    path: Path
    paradigm: str
    n_neurons: int
    n_frames: int
    state: np.ndarray
    segment_stops: np.ndarray
    file_size: int
    file_mtime_ns: int

    @property
    def segment_starts(self) -> np.ndarray:
        return np.r_[0, self.segment_stops[:-1]]


@dataclass(frozen=True)
class Window:
    window_id: str
    recording: str
    label: str
    state_code: float
    start: int
    stop: int
    n_frames: int
    segment: int
    kind: str
    split: str
    temporal_position: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _decode_matlab_text(dataset: h5py.Dataset) -> str:
    codes = np.asarray(dataset).ravel(order="F")
    return "".join(chr(int(code)) for code in codes if int(code)).strip()


def load_metadata(path: str | Path, expected_paradigm: str | None = None) -> RecordingMetadata:
    """Read small metadata only; activity matrices remain on disk."""
    path = Path(path)
    with h5py.File(path, "r") as mat:
        paradigm = _decode_matlab_text(mat["data_info"])
        if expected_paradigm is not None and paradigm != expected_paradigm:
            raise ValueError(f"{path.name}: expected {expected_paradigm!r}, found {paradigm!r}")
        state = np.asarray(mat["state"]).ravel(order="F").astype(float, copy=False)
        n_frames = int(state.size)
        if "nonzero_ROI" in mat:
            n_neurons = int(mat["nonzero_ROI"].size)
        else:
            shape = mat["spike_deconv"].shape
            n_neurons = int(shape[1] if shape[0] == n_frames else shape[0])
        valid_shapes = {(n_frames, n_neurons), (n_neurons, n_frames)}
        for signal in ("dFF", "spike_deconv", "spike_smoothed"):
            if mat[signal].shape not in valid_shapes:
                raise ValueError(
                    f"{path.name}:{signal} shape {mat[signal].shape} is inconsistent "
                    f"with N={n_neurons}, T={n_frames}"
                )
        raw_stops = np.asarray(mat["frame/boundary_ind"]).ravel(order="F")
    segment_stops = np.unique(np.rint(raw_stops).astype(int))
    segment_stops = segment_stops[(segment_stops > 0) & (segment_stops <= n_frames)]
    if segment_stops.size == 0 or segment_stops[-1] != n_frames:
        segment_stops = np.r_[segment_stops, n_frames]
    stat = path.stat()
    return RecordingMetadata(
        name=path.stem,
        path=path.resolve(),
        paradigm=paradigm,
        n_neurons=n_neurons,
        n_frames=n_frames,
        state=state,
        segment_stops=segment_stops,
        file_size=stat.st_size,
        file_mtime_ns=stat.st_mtime_ns,
    )


def state_runs(meta: RecordingMetadata, state_code: float | None = None) -> list[tuple[int, int, float, int]]:
    """Return half-open constant-state runs, split at acquisition boundaries."""
    runs: list[tuple[int, int, float, int]] = []
    for segment, (segment_start, segment_stop) in enumerate(
        zip(meta.segment_starts, meta.segment_stops, strict=True)
    ):
        values = meta.state[segment_start:segment_stop]
        changes = np.r_[0, np.flatnonzero(np.diff(values) != 0) + 1, values.size]
        for local_start, local_stop in zip(changes[:-1], changes[1:], strict=True):
            start = int(segment_start + local_start)
            stop = int(segment_start + local_stop)
            code = float(meta.state[start])
            if state_code is None or np.isclose(code, state_code):
                runs.append((start, stop, code, segment))
    return runs


def maximin_windows(
    meta: RecordingMetadata,
    label: str,
    state_code: float,
    n_frames: int,
    count: int,
    development_positions: Iterable[int],
) -> list[Window]:
    """Globally maximize the minimum center spacing, then choose earliest ties.

    All windows have the same length, so maximizing the minimum center spacing
    is equivalent to maximizing the minimum difference between sorted starts.
    Feasibility at a proposed integer spacing is monotone and the lexicographically
    earliest feasible sequence is obtained greedily; an integer binary search
    therefore gives the global maximin solution without inspecting DMD values.
    """
    candidates: list[tuple[int, int, int]] = []
    for run_start, run_stop, _, segment in state_runs(meta, state_code):
        for start in range(run_start, run_stop - n_frames + 1):
            candidates.append((start, start + n_frames, segment))
    if not candidates:
        raise ValueError(f"{meta.name}:{label} has no legal {n_frames}-frame window")

    candidates.sort(key=lambda item: item[0])

    def earliest_sequence(spacing: int) -> list[tuple[int, int, int]]:
        sequence: list[tuple[int, int, int]] = []
        minimum_start = -np.inf
        for item in candidates:
            if item[0] >= minimum_start:
                sequence.append(item)
                minimum_start = item[0] + spacing
                if len(sequence) == count:
                    break
        return sequence

    lower = n_frames
    upper = candidates[-1][0] - candidates[0][0] + 1
    if len(earliest_sequence(lower)) < count:
        raise ValueError(f"{meta.name}:{label} cannot supply {count} non-overlapping windows")
    while lower + 1 < upper:
        middle = (lower + upper) // 2
        if len(earliest_sequence(middle)) >= count:
            lower = middle
        else:
            upper = middle
    selected = earliest_sequence(lower)

    development_positions = set(int(position) for position in development_positions)
    windows: list[Window] = []
    for position, (start, stop, segment) in enumerate(sorted(selected)):
        split = "development" if position in development_positions else "evaluation"
        windows.append(
            Window(
                window_id=f"{meta.name}__{label}__w{position + 1:02d}__{split}",
                recording=meta.name,
                label=label,
                state_code=float(state_code),
                start=int(start),
                stop=int(stop),
                n_frames=int(stop - start),
                segment=int(segment),
                kind="deployment",
                split=split,
                temporal_position=position,
            )
        )
    validate_windows(meta, windows)
    return windows


def centered_long_window(
    meta: RecordingMetadata,
    label: str,
    state_code: float,
    n_frames: int,
) -> Window:
    """Center a diagnostic block in the longest legal constant-label bout."""
    runs = [run for run in state_runs(meta, state_code) if run[1] - run[0] >= n_frames]
    if not runs:
        raise ValueError(f"{meta.name}:{label} has no {n_frames}-frame constant-label bout")
    run_start, run_stop, _, segment = max(runs, key=lambda run: (run[1] - run[0], -run[0]))
    start = int(run_start + ((run_stop - run_start) - n_frames) // 2)
    window = Window(
        window_id=f"{meta.name}__{label}__long",
        recording=meta.name,
        label=label,
        state_code=float(state_code),
        start=start,
        stop=start + n_frames,
        n_frames=n_frames,
        segment=int(segment),
        kind="long",
        split="diagnostic",
        temporal_position=0,
    )
    validate_windows(meta, [window])
    return window


def validate_windows(meta: RecordingMetadata, windows: Iterable[Window]) -> None:
    """Fail loudly on label changes, segment crossings, or malformed intervals."""
    for window in windows:
        if not (0 <= window.start < window.stop <= meta.n_frames):
            raise ValueError(f"Illegal bounds in {window.window_id}")
        if window.stop - window.start != window.n_frames:
            raise ValueError(f"Length mismatch in {window.window_id}")
        if not np.all(np.isclose(meta.state[window.start:window.stop], window.state_code)):
            raise ValueError(f"State change inside {window.window_id}")
        segment_start = int(meta.segment_starts[window.segment])
        segment_stop = int(meta.segment_stops[window.segment])
        if window.start < segment_start or window.stop > segment_stop:
            raise ValueError(f"Acquisition boundary inside {window.window_id}")


def windows_frame(windows: Iterable[Window]) -> pd.DataFrame:
    return pd.DataFrame([window.to_dict() for window in windows]).sort_values(
        ["recording", "kind", "label", "start"], ignore_index=True
    )


def read_signal(meta: RecordingMetadata, signal: str, start: int, stop: int) -> np.ndarray:
    """Read one contiguous time slice and return neuron-by-time float64 data."""
    with h5py.File(meta.path, "r") as mat:
        dataset = mat[signal]
        if dataset.shape == (meta.n_frames, meta.n_neurons):
            values = np.asarray(dataset[start:stop, :], dtype=np.float64).T
        elif dataset.shape == (meta.n_neurons, meta.n_frames):
            values = np.asarray(dataset[:, start:stop], dtype=np.float64)
        else:
            raise ValueError(f"Unexpected {signal} shape {dataset.shape} in {meta.path.name}")
    if values.shape != (meta.n_neurons, stop - start):
        raise ValueError(f"Selective read returned {values.shape}, expected {(meta.n_neurons, stop-start)}")
    return values


def read_windows(
    meta: RecordingMetadata,
    signal: str,
    windows: Iterable[Window],
) -> list[np.ndarray]:
    return [read_signal(meta, signal, window.start, window.stop) for window in windows]
