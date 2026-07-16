"""Candidate signal construction, train-only scaling, and frozen PCA bases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.linalg

from .data import RecordingMetadata, Window, read_windows


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    signal: str
    bin_frames: int
    scaling: str
    description: str

    @classmethod
    def from_config(cls, arm_id: str, values: dict[str, object]) -> "ArmSpec":
        return cls(
            arm_id=arm_id,
            signal=str(values["signal"]),
            bin_frames=int(values["bin_frames"]),
            scaling=str(values["scaling"]),
            description=str(values["description"]),
        )


@dataclass
class FrozenRepresentation:
    recording: str
    arm_id: str
    mean: np.ndarray
    scale: np.ndarray
    eligible_rows: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    total_energy: float
    orthogonality_error: float
    bin_frames: int
    fs_hz: float

    def transform(self, raw_window: np.ndarray, rank: int | None = None) -> np.ndarray:
        """Apply the frozen arm transform and return PC-by-time scores."""
        arm_window = construct_arm_window(raw_window, self.bin_frames, self.fs_hz)
        scaled = (arm_window[self.eligible_rows] - self.mean[:, None]) / self.scale[:, None]
        components = self.components if rank is None else self.components[:, :rank]
        return components.T @ scaled

    def scaled_window(self, raw_window: np.ndarray) -> np.ndarray:
        arm_window = construct_arm_window(raw_window, self.bin_frames, self.fs_hz)
        return (arm_window[self.eligible_rows] - self.mean[:, None]) / self.scale[:, None]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            recording=np.asarray(self.recording),
            arm_id=np.asarray(self.arm_id),
            mean=self.mean,
            scale=self.scale,
            eligible_rows=self.eligible_rows,
            components=self.components,
            singular_values=self.singular_values,
            explained_variance_ratio=self.explained_variance_ratio,
            total_energy=np.asarray(self.total_energy),
            orthogonality_error=np.asarray(self.orthogonality_error),
            bin_frames=np.asarray(self.bin_frames),
            fs_hz=np.asarray(self.fs_hz),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FrozenRepresentation":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                recording=str(data["recording"]),
                arm_id=str(data["arm_id"]),
                mean=data["mean"],
                scale=data["scale"],
                eligible_rows=data["eligible_rows"].astype(int),
                components=data["components"],
                singular_values=data["singular_values"],
                explained_variance_ratio=data["explained_variance_ratio"],
                total_energy=float(data["total_energy"]),
                orthogonality_error=float(data["orthogonality_error"]),
                bin_frames=int(data["bin_frames"]),
                fs_hz=float(data["fs_hz"]),
            )


def construct_arm_window(raw_window: np.ndarray, bin_frames: int, fs_hz: float) -> np.ndarray:
    """Construct one arm inside one window, never joining separate windows."""
    raw_window = np.asarray(raw_window, dtype=np.float64)
    if raw_window.ndim != 2:
        raise ValueError("Activity must be neuron by time")
    if bin_frames == 1:
        return raw_window
    if bin_frames <= 0:
        raise ValueError("bin_frames must be positive")
    n_complete = raw_window.shape[1] // bin_frames
    if n_complete == 0:
        raise ValueError("Window is shorter than one bin")
    trimmed = raw_window[:, : n_complete * bin_frames]
    event_mass = trimmed.reshape(raw_window.shape[0], n_complete, bin_frames).sum(axis=2)
    return event_mass / (bin_frames / fs_hz)


def load_arm_windows(
    meta: RecordingMetadata,
    windows: Iterable[Window],
    spec: ArmSpec,
    fs_hz: float,
) -> list[np.ndarray]:
    raw = read_windows(meta, spec.signal, windows)
    return [construct_arm_window(values, spec.bin_frames, fs_hz) for values in raw]


def fit_scaler(
    windows: Iterable[np.ndarray],
    scaling: str,
    rms_epsilon: float = 1e-3,
    variance_tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Fit neuronwise centering/scaling from development windows only."""
    windows = [np.asarray(window, dtype=np.float64) for window in windows]
    if not windows:
        raise ValueError("At least one development window is required")
    n_neurons = windows[0].shape[0]
    if any(window.ndim != 2 or window.shape[0] != n_neurons for window in windows):
        raise ValueError("Development windows must have aligned neuron rows")
    counts = sum(window.shape[1] for window in windows)
    finite = np.ones(n_neurons, dtype=bool)
    sums = np.zeros(n_neurons, dtype=np.float64)
    for window in windows:
        finite &= np.all(np.isfinite(window), axis=1)
        sums += np.where(np.isfinite(window), window, 0.0).sum(axis=1)
    mean = sums / counts
    sum_squares = np.zeros(n_neurons, dtype=np.float64)
    for window in windows:
        centered = np.where(np.isfinite(window), window - mean[:, None], 0.0)
        sum_squares += np.einsum("ij,ij->i", centered, centered, optimize=True)
    rms = np.sqrt(sum_squares / counts)
    eligible = finite & (rms > variance_tolerance)
    if not np.any(eligible):
        raise ValueError("No finite, nonconstant neuron remains after train-only filtering")
    if scaling == "rms":
        scale = rms[eligible] + rms_epsilon
    elif scaling == "center":
        scale = np.ones(np.count_nonzero(eligible), dtype=np.float64)
    else:
        raise ValueError(f"Unknown scaling rule: {scaling}")
    audit = {
        "input_neurons": n_neurons,
        "eligible_neurons": int(np.count_nonzero(eligible)),
        "nonfinite_neurons": int(np.count_nonzero(~finite)),
        "constant_neurons": int(np.count_nonzero(finite & (rms <= variance_tolerance))),
        "development_samples_per_neuron": counts,
    }
    return mean[eligible], scale, np.flatnonzero(eligible), audit


def apply_scaler(
    windows: Iterable[np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
    eligible_rows: np.ndarray,
) -> list[np.ndarray]:
    return [
        (np.asarray(window)[eligible_rows] - mean[:, None]) / scale[:, None]
        for window in windows
    ]


def randomized_pca(
    scaled_windows: Iterable[np.ndarray],
    max_rank: int,
    oversamples: int,
    power_iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Deterministic Halko-style truncated SVD on pooled development samples."""
    matrix = np.concatenate([np.asarray(window, dtype=np.float64) for window in scaled_windows], axis=1)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("PCA input contains non-finite values")
    max_rank = min(int(max_rank), min(matrix.shape))
    sketch_rank = min(min(matrix.shape), max_rank + int(oversamples))
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((matrix.shape[1], sketch_rank))
    q, _ = scipy.linalg.qr(matrix @ omega, mode="economic", check_finite=False)
    for _ in range(int(power_iterations)):
        z, _ = scipy.linalg.qr(matrix.T @ q, mode="economic", check_finite=False)
        q, _ = scipy.linalg.qr(matrix @ z, mode="economic", check_finite=False)
    small = q.T @ matrix
    u_small, singular_values, _ = scipy.linalg.svd(
        small, full_matrices=False, check_finite=False, lapack_driver="gesdd"
    )
    components = (q @ u_small[:, :max_rank]).astype(np.float64, copy=False)
    singular_values = singular_values[:max_rank]
    for index in range(components.shape[1]):
        pivot = int(np.argmax(np.abs(components[:, index])))
        if components[pivot, index] < 0:
            components[:, index] *= -1
    gram = components.T @ components
    orthogonality_error = float(np.linalg.norm(gram - np.eye(max_rank), ord="fro"))
    total_energy = float(np.einsum("ij,ij->", matrix, matrix, optimize=True))
    explained = singular_values**2 / total_energy if total_energy > 0 else np.zeros_like(singular_values)
    return components, singular_values, explained, total_energy, orthogonality_error


def fit_representation(
    meta: RecordingMetadata,
    spec: ArmSpec,
    development_windows: Iterable[Window],
    fs_hz: float,
    max_rank: int,
    oversamples: int,
    power_iterations: int,
    seed: int,
) -> tuple[FrozenRepresentation, list[np.ndarray], list[np.ndarray], dict[str, int]]:
    """Fit one recording/arm representation and return its scaled dev windows."""
    arm_windows = load_arm_windows(meta, development_windows, spec, fs_hz)
    mean, scale, eligible_rows, audit = fit_scaler(arm_windows, spec.scaling)
    scaled = apply_scaler(arm_windows, mean, scale, eligible_rows)
    components, singular_values, explained, total_energy, orthogonality_error = randomized_pca(
        scaled,
        max_rank=max_rank,
        oversamples=oversamples,
        power_iterations=power_iterations,
        seed=seed,
    )
    representation = FrozenRepresentation(
        recording=meta.name,
        arm_id=spec.arm_id,
        mean=mean,
        scale=scale,
        eligible_rows=eligible_rows,
        components=components,
        singular_values=singular_values,
        explained_variance_ratio=explained,
        total_energy=total_energy,
        orthogonality_error=orthogonality_error,
        bin_frames=spec.bin_frames,
        fs_hz=fs_hz,
    )
    return representation, arm_windows, scaled, audit
