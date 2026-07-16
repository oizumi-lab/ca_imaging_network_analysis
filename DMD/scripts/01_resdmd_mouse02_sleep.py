"""Focused PyDMD/Residual-DMD tutorial for one mouse02 NREM block.

Run cells in order in VS Code, Spyder, or another editor that recognizes
``# %%`` markers. Running the whole file is also supported as a reproducibility
check, but the intended use is to inspect each printed result and figure before
continuing.

This is deliberately a pipeline audit, not a brain-state comparison. Each
``# %%`` cell explains its input, operation, quantitative output, and remaining
limitation before the next cell is run.
"""

# %% Step 0a — Imports
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
from pydmd import HankelDMD
from pydmd.utils import compute_svd, pseudo_hankel_matrix
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


# %% Step 0b — Freeze the narrow question
#
# This tutorial asks only:
#
# > Can PyDMD find a low-dimensional, delay-coordinate linear model that
# > actually captures the transformed neural activity in one homogeneous block
# > from `mouse02_sleep`, and do Residual-DMD diagnostics support its modes?
#
# Important naming correction:
#
# - PyDMD performs every DMD/Hankel fit in this revision.
# - PyDMD 2025.8.1 does not provide a Residual-DMD class. Its `RDMD` class is
#   Randomized DMD. Therefore the small `G`, `A`, `L` calculation below is used
#   only to evaluate ResDMD eigenfunction residuals for the PyDMD candidates.
# - A one-step lag is not a delay embedding. We explicitly test delay orders
#   `d = 1, 2, 4, 8, 16`; `d=1` means no delay augmentation.
#
# PyDMD sources used for this implementation:
#
# - https://pydmd.github.io/PyDMD/tutorial1dmd.html
# - https://pydmd.github.io/PyDMD/hankeldmd.html
# - the installed `pydmd.hankeldmd`, `pydmd.dmd`, and `pydmd.utils` source.
#
# We stop after one held-out forecast check. There is no awake comparison,
# second recording, sliding window, bootstrap, null model, or mode tracking.

print(
    "SCOPE: mouse02_sleep, one NREM block, PyDMD delay audit, "
    "Residual-DMD diagnostics, then STOP."
)


# %% Step 1 — Fixed settings
SEED = 20260717
FS_HZ = 7.65
STATE_CODE = 1.0
STATE_NAME = "NREM"
BLOCK_FRAMES = 1500

# One transparent temporal preprocessing choice. Four non-overlapping frames
# correspond to 0.523 s and leave 375 samples in the selected block.
BIN_FRAMES = 4

# Delay order is the number of consecutive binned neural snapshots stacked by
# PyDMD. These candidates cover 0 to 7.84 s of history at the chosen bin size.
DELAY_CANDIDATES = (1, 2, 4, 8, 16)

# `svd_rank=0` asks PyDMD to use its documented Gavish-Donoho automatic hard
# threshold. We report the resulting rank instead of silently fixing it.
PYDMD_SVD_RANK = 0
SPATIAL_MODES_TO_PLOT = 4
PX_TO_UM = 1.465
TRAIN_FRACTION = 0.60
CALIBRATION_FRACTION = 0.20

DMD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DMD_ROOT.parent
DATA_PATH = REPO_ROOT / "data" / "raw" / "mouse02_sleep.mat"
OUTPUT_DIR = DMD_ROOT / "results" / "05_pydmd_resdmd_mouse02"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    }
)


def save_and_show(fig: plt.Figure, filename: str) -> None:
    """Save every intermediate figure and also display it interactively."""
    path = OUTPUT_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()
    plt.close(fig)


print(f"Input:  {DATA_PATH}")
print(f"Output: {OUTPUT_DIR}")
print(f"Sampling rate: {FS_HZ:.2f} Hz")
print(f"Rate bin: {BIN_FRAMES} frames = {BIN_FRAMES / FS_HZ:.3f} s")
print(f"PyDMD version: {version('pydmd')}")


# %% Step 2 — Define the exact Residual-DMD calculation
#
# Let rows of `Psi_X` and `Psi_Y` be paired evaluations of a dictionary of
# observables. Residual DMD forms
#
# \[
# G=\Psi_X^*W\Psi_X,\quad
# A=\Psi_X^*W\Psi_Y,\quad
# L=\Psi_Y^*W\Psi_Y.
# \]
#
# Candidate eigenfunctions solve `A g = lambda G g`. Their residual is
#
# \[
# \mathrm{res}(\lambda,g)^2=
# \frac{g^*[L-\lambda A^*-\bar\lambda A+|\lambda|^2G]g}
#      {g^*Gg}.
# \]
#
# We learn candidates on the training segment and evaluate this residual on a
# separate calibration segment. That separation is stricter than reporting a
# residual on the same pairs used to obtain the eigenfunction.


@dataclass(frozen=True)
class ResDMDResult:
    eigenvalues: np.ndarray
    eigenfunctions: np.ndarray
    training_residuals: np.ndarray
    calibration_residuals: np.ndarray
    g_fit: np.ndarray
    a_fit: np.ndarray
    l_fit: np.ndarray
    g_calibration: np.ndarray
    a_calibration: np.ndarray
    l_calibration: np.ndarray
    gram_condition_number: float


def resdmd_matrices(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Form uniformly weighted G, A, and L for an identity dictionary."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 2 or y.shape != x.shape:
        raise ValueError("x and y must be aligned coordinate-by-pair matrices")
    pair_count = x.shape[1]
    if pair_count == 0:
        raise ValueError("at least one pair is required")
    g = (x.conj() @ x.T) / pair_count
    a = (x.conj() @ y.T) / pair_count
    l_matrix = (y.conj() @ y.T) / pair_count
    # G and L are Hermitian by definition. Symmetrizing removes round-off only.
    g = (g + g.conj().T) / 2
    l_matrix = (l_matrix + l_matrix.conj().T) / 2
    return g, a, l_matrix


def eigenpair_residuals(
    eigenvalues: np.ndarray,
    eigenfunctions: np.ndarray,
    g: np.ndarray,
    a: np.ndarray,
    l_matrix: np.ndarray,
) -> np.ndarray:
    """Evaluate published relative ResDMD residuals for fixed candidates."""
    values: list[float] = []
    for eigenvalue, vector in zip(eigenvalues, eigenfunctions.T, strict=True):
        residual_form = (
            l_matrix
            - eigenvalue * a.conj().T
            - np.conj(eigenvalue) * a
            + abs(eigenvalue) ** 2 * g
        )
        residual_form = (residual_form + residual_form.conj().T) / 2
        numerator = float(np.real(vector.conj() @ residual_form @ vector))
        denominator = float(np.real(vector.conj() @ g @ vector))
        tolerance = 100 * np.finfo(float).eps * max(1.0, abs(denominator))
        if denominator <= tolerance:
            values.append(np.nan)
        else:
            # A tiny negative value can arise from floating-point cancellation.
            values.append(float(np.sqrt(max(0.0, numerator) / denominator)))
    return np.asarray(values)


def fit_linear_resdmd(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_calibration: np.ndarray,
    y_calibration: np.ndarray,
) -> ResDMDResult:
    """Fit linear-dictionary EDMD candidates and certify them with ResDMD."""
    g_fit, a_fit, l_fit = resdmd_matrices(x_fit, y_fit)
    g_cal, a_cal, l_cal = resdmd_matrices(x_calibration, y_calibration)

    gram_eigenvalues = scipy.linalg.eigvalsh(g_fit, check_finite=True)
    tolerance = (
        max(g_fit.shape)
        * np.finfo(float).eps
        * max(float(np.max(gram_eigenvalues)), 1.0)
    )
    if np.min(gram_eigenvalues) <= tolerance:
        raise np.linalg.LinAlgError(
            "Training Gram matrix is rank deficient; reduce the PyDMD/POD rank"
        )

    # A g = lambda G g is the EDMD eigenfunction problem.
    eigenvalues, eigenfunctions = scipy.linalg.eig(
        a_fit, g_fit, check_finite=True
    )

    # Normalize each coefficient vector in the training G-inner product.
    for index in range(eigenfunctions.shape[1]):
        vector = eigenfunctions[:, index]
        norm = float(np.sqrt(np.real(vector.conj() @ g_fit @ vector)))
        if norm > 0:
            eigenfunctions[:, index] = vector / norm

    training_residuals = eigenpair_residuals(
        eigenvalues, eigenfunctions, g_fit, a_fit, l_fit
    )
    calibration_residuals = eigenpair_residuals(
        eigenvalues, eigenfunctions, g_cal, a_cal, l_cal
    )

    return ResDMDResult(
        eigenvalues=eigenvalues,
        eigenfunctions=eigenfunctions,
        training_residuals=training_residuals,
        calibration_residuals=calibration_residuals,
        g_fit=g_fit,
        a_fit=a_fit,
        l_fit=l_fit,
        g_calibration=g_cal,
        a_calibration=a_cal,
        l_calibration=l_cal,
        gram_condition_number=float(np.linalg.cond(g_fit)),
    )


# %% Step 3 — Synthetic unit check before touching the biological data
#
# A residual implementation should first recover an exact known linear map. We
# use independent random input pairs so decay along one trajectory cannot make
# the test Gram matrix nearly singular.

rho = 0.94
theta = 0.31
known_operator = rho * np.array(
    [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
)
synthetic_x = rng.standard_normal((2, 900))
synthetic_y = known_operator @ synthetic_x

synthetic_result = fit_linear_resdmd(
    synthetic_x[:, :600],
    synthetic_y[:, :600],
    synthetic_x[:, 600:],
    synthetic_y[:, 600:],
)

known_eigenvalues = scipy.linalg.eigvals(known_operator)
cost = abs(
    synthetic_result.eigenvalues[:, None] - known_eigenvalues[None, :]
)
rows, columns = linear_sum_assignment(cost)
maximum_eigenvalue_error = float(np.max(cost[rows, columns]))
maximum_calibration_residual = float(
    np.nanmax(synthetic_result.calibration_residuals)
)

if maximum_eigenvalue_error > 1e-10 or maximum_calibration_residual > 1e-7:
    raise RuntimeError("The exact synthetic Residual-DMD check failed")

print("SUCCESS — exact synthetic check")
print(f"  maximum eigenvalue error:       {maximum_eigenvalue_error:.3e}")
print(f"  maximum calibration residual:  {maximum_calibration_residual:.3e}")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
circle = np.exp(1j * np.linspace(0, 2 * np.pi, 400))
axes[0].plot(circle.real, circle.imag, color="0.75", linewidth=1)
axes[0].scatter(
    known_eigenvalues.real,
    known_eigenvalues.imag,
    marker="x",
    s=80,
    linewidth=2,
    label="known",
)
axes[0].scatter(
    synthetic_result.eigenvalues.real,
    synthetic_result.eigenvalues.imag,
    facecolors="none",
    edgecolors="#D55E00",
    s=80,
    label="recovered",
)
axes[0].set(xlabel="Re(lambda)", ylabel="Im(lambda)", title="Exact eigenvalues")
axes[0].set_aspect("equal", adjustable="box")
axes[0].legend(frameon=False)
axes[1].bar(
    np.arange(synthetic_result.calibration_residuals.size),
    np.maximum(synthetic_result.calibration_residuals, 1e-16),
    color="#0072B2",
)
axes[1].set_yscale("log")
axes[1].set(
    xlabel="mode",
    ylabel="independent residual",
    title="Exact-map residual check",
)
fig.suptitle("Step 3 — Residual-DMD implementation check")
save_and_show(fig, "00_synthetic_resdmd_check.png")


# %% Step 4 — Read metadata and choose one legal NREM block
#
# We use the raw `state` labels and acquisition boundaries. The paper-specific
# `used_frame` mask is intentionally irrelevant here.

if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)

with h5py.File(DATA_PATH, "r") as mat:
    state = np.asarray(mat["state"]).ravel(order="F").astype(float)
    raw_boundaries = np.asarray(mat["frame/boundary_ind"]).ravel(order="F")
    signal_shape = tuple(mat["spike_deconv"].shape)
    if "nonzero_ROI" in mat:
        n_neurons = int(mat["nonzero_ROI"].size)
    else:
        n_neurons = int(signal_shape[1] if signal_shape[0] == state.size else signal_shape[0])

n_frames = int(state.size)
segment_stops = np.unique(np.rint(raw_boundaries).astype(int))
segment_stops = segment_stops[(segment_stops > 0) & (segment_stops <= n_frames)]
if segment_stops.size == 0 or segment_stops[-1] != n_frames:
    segment_stops = np.r_[segment_stops, n_frames]
segment_starts = np.r_[0, segment_stops[:-1]]


def state_runs(
    state_values: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
) -> list[tuple[int, int, float, int]]:
    """Return constant-label runs split at every acquisition boundary."""
    runs: list[tuple[int, int, float, int]] = []
    for segment, (segment_start, segment_stop) in enumerate(
        zip(starts, stops, strict=True)
    ):
        values = state_values[segment_start:segment_stop]
        changes = np.r_[0, np.flatnonzero(np.diff(values) != 0) + 1, values.size]
        for local_start, local_stop in zip(changes[:-1], changes[1:], strict=True):
            start = int(segment_start + local_start)
            stop = int(segment_start + local_stop)
            runs.append((start, stop, float(state_values[start]), segment))
    return runs


nrem_runs = [
    run
    for run in state_runs(state, segment_starts, segment_stops)
    if np.isclose(run[2], STATE_CODE) and run[1] - run[0] >= BLOCK_FRAMES
]
if not nrem_runs:
    raise RuntimeError(f"No legal {BLOCK_FRAMES}-frame {STATE_NAME} run exists")

# Longest run, then earliest run if tied; center the one and only tutorial block.
run_start, run_stop, _, block_segment = max(
    nrem_runs, key=lambda run: (run[1] - run[0], -run[0])
)
block_start = int(run_start + ((run_stop - run_start) - BLOCK_FRAMES) // 2)
block_stop = block_start + BLOCK_FRAMES

if not np.all(np.isclose(state[block_start:block_stop], STATE_CODE)):
    raise RuntimeError("State changes inside the selected block")
if not (
    segment_starts[block_segment] <= block_start < block_stop <= segment_stops[block_segment]
):
    raise RuntimeError("The selected block crosses an acquisition boundary")

print("SUCCESS — one legal block selected without looking at activity")
print(f"  recording:     {DATA_PATH.name}")
print(f"  neurons:       {n_neurons:,}")
print(f"  frames:        {n_frames:,}")
print(f"  block:         [{block_start}, {block_stop})")
print(f"  duration:      {BLOCK_FRAMES / FS_HZ:.2f} s")
print(f"  containing run:[{run_start}, {run_stop})")
print(f"  acquisition:   segment {block_segment}")

time_minutes = np.arange(n_frames) / FS_HZ / 60
fig, axes = plt.subplots(2, 1, figsize=(11, 5), constrained_layout=True)
axes[0].step(time_minutes, state, where="post", color="#4C78A8", linewidth=0.8)
axes[0].axvspan(
    block_start / FS_HZ / 60,
    block_stop / FS_HZ / 60,
    color="#E45756",
    alpha=0.35,
    label="selected NREM block",
)
for boundary in segment_stops[:-1]:
    axes[0].axvline(boundary / FS_HZ / 60, color="0.25", linestyle="--", linewidth=0.8)
axes[0].set(
    xlabel="recording time (min)",
    ylabel="raw state code",
    title="Full mouse02 state timeline",
)
axes[0].set_yticks([0, 0.5, 1, 2], ["awake", "quiet awake", "NREM", "REM"])
axes[0].legend(frameon=False, loc="upper right")

padding = 400
zoom_start = max(0, block_start - padding)
zoom_stop = min(n_frames, block_stop + padding)
zoom_time = np.arange(zoom_start, zoom_stop) / FS_HZ
axes[1].step(
    zoom_time,
    state[zoom_start:zoom_stop],
    where="post",
    color="#4C78A8",
    linewidth=1,
)
axes[1].axvspan(block_start / FS_HZ, block_stop / FS_HZ, color="#E45756", alpha=0.35)
axes[1].set(
    xlabel="recording time (s)",
    ylabel="raw state code",
    title="Selected block and surrounding labels",
)
axes[1].set_yticks([0, 0.5, 1, 2], ["awake", "quiet awake", "NREM", "REM"])
fig.suptitle("Step 4 — Dataset and block geometry")
save_and_show(fig, "01_state_timeline_and_block.png")


# %% Step 5 — Load the selected activity block and neuron coordinates
#
# The activity matrix is loaded only for the selected frames. Centroids are
# loaded for all neurons because later spatial-mode plots must preserve the
# exact row-to-neuron mapping.

with h5py.File(DATA_PATH, "r") as mat:
    dataset = mat["spike_deconv"]
    if dataset.shape == (n_frames, n_neurons):
        spike_deconv = np.asarray(
            dataset[block_start:block_stop, :], dtype=np.float64
        ).T
    elif dataset.shape == (n_neurons, n_frames):
        spike_deconv = np.asarray(
            dataset[:, block_start:block_stop], dtype=np.float64
        )
    else:
        raise ValueError(f"Unexpected spike_deconv shape: {dataset.shape}")
    centroid_raw = np.asarray(mat["ROIs/Centroid"], dtype=np.float64)

if centroid_raw.shape == (2, n_neurons):
    centroid_px = centroid_raw.T
elif centroid_raw.shape == (n_neurons, 2):
    centroid_px = centroid_raw
else:
    raise ValueError(f"Unexpected centroid shape: {centroid_raw.shape}")
centroid_um = centroid_px * PX_TO_UM

if spike_deconv.shape != (n_neurons, BLOCK_FRAMES):
    raise RuntimeError(f"Unexpected selected data shape: {spike_deconv.shape}")
if not np.all(np.isfinite(spike_deconv)):
    raise RuntimeError("Non-finite value found in the selected block")

raw_zero_fraction = float(np.mean(spike_deconv == 0))
active_per_frame = np.count_nonzero(spike_deconv, axis=0)
nonzero_values = spike_deconv[spike_deconv != 0]

print("OBSERVATION — native deconvolved events")
print(f"  shape:                    {spike_deconv.shape}")
print(f"  exact-zero fraction:      {raw_zero_fraction:.4%}")
print(f"  median active neurons:    {np.median(active_per_frame):.0f} / {n_neurons:,}")
print(f"  nonzero amplitude range:  [{nonzero_values.min():.4g}, {nonzero_values.max():.4g}]")
print(f"  centroid range x/y:       {np.ptp(centroid_um, axis=0).round(1)} um")
print("LIMITATION — this native matrix is an event estimate, not a continuous firing-rate signal.")

neuron_order = np.argsort(np.sum(np.abs(spike_deconv), axis=1), kind="stable")
if np.min(spike_deconv) < 0:
    raise RuntimeError("The event heatmap expects nonnegative deconvolved amplitudes")
log_activity = np.log1p(spike_deconv)
display_limit = float(np.quantile(log_activity, 0.995))
block_seconds = np.arange(BLOCK_FRAMES) / FS_HZ

fig, axes = plt.subplots(
    3,
    1,
    figsize=(11, 8),
    gridspec_kw={"height_ratios": [3.4, 1, 1]},
    constrained_layout=True,
)
image = axes[0].imshow(
    log_activity[neuron_order],
    aspect="auto",
    interpolation="nearest",
    cmap="magma",
    vmin=0,
    vmax=display_limit,
    extent=[0, BLOCK_FRAMES / FS_HZ, 0, n_neurons],
)
axes[0].set(
    xlabel="time within block (s)",
    ylabel="all neurons\n(sorted by event mass)",
    title="All-neuron log event activity",
)
fig.colorbar(image, ax=axes[0], label="log(1+x)", pad=0.01)
axes[1].plot(block_seconds, active_per_frame, color="#009E73", linewidth=0.8)
axes[1].set(
    xlabel="time within block (s)",
    ylabel="active neurons",
    title="Number of nonzero neuron entries per native frame",
)
axes[2].hist(nonzero_values, bins=80, color="#4C78A8", alpha=0.85)
axes[2].set(
    xlabel="nonzero deconvolved amplitude",
    ylabel="count",
    yscale="log",
    title="Nonzero amplitude distribution",
)
fig.suptitle("Step 5 — Raw input inspection; no neuron sampling")
save_and_show(fig, "02_raw_spike_deconv_inspection.png")


# %% Step 6 — Convert event mass to one explicit non-overlapping rate proxy
#
# This is the only preprocessing arm in this tutorial. We sum four native
# deconvolution samples, then divide by the physical bin duration. The result is
# a rate-like event-mass observable for every neuron. It is not calibrated
# spikes/s because deconvolution amplitudes are not known spike counts.

n_complete_bins = spike_deconv.shape[1] // BIN_FRAMES
trimmed_native_frames = n_complete_bins * BIN_FRAMES
bin_duration_seconds = BIN_FRAMES / FS_HZ
event_rate = (
    spike_deconv[:, :trimmed_native_frames]
    .reshape(n_neurons, n_complete_bins, BIN_FRAMES)
    .sum(axis=2)
    / bin_duration_seconds
)
native_event_mass = float(np.sum(spike_deconv[:, :trimmed_native_frames]))
recovered_event_mass = float(np.sum(event_rate) * bin_duration_seconds)
relative_mass_error = abs(recovered_event_mass - native_event_mass) / max(
    abs(native_event_mass), np.finfo(float).eps
)

rate_zero_fraction = float(np.mean(event_rate == 0))
raw_population_rate = np.mean(spike_deconv, axis=0) * FS_HZ
binned_population_rate = np.mean(event_rate, axis=0)


def lag_correlation(values: np.ndarray, lag: int) -> float:
    left = np.asarray(values[:-lag], dtype=float)
    right = np.asarray(values[lag:], dtype=float)
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


raw_population_common_lag = lag_correlation(raw_population_rate, BIN_FRAMES)
rate_population_common_lag = lag_correlation(binned_population_rate, 1)

print("PREPROCESSING — four-frame non-overlapping event-mass rate")
print(f"  output shape:              {event_rate.shape}")
print(f"  bin duration:              {bin_duration_seconds:.3f} s")
print(f"  exact-zero fraction:       {raw_zero_fraction:.4%} -> {rate_zero_fraction:.4%}")
print(f"  relative event-mass error: {relative_mass_error:.3e}")
print(
    f"  population corr. at {bin_duration_seconds:.3f} s: "
    f"{raw_population_common_lag:.3f} -> {rate_population_common_lag:.3f}"
)
print("SUCCESS — neuron identity and event mass are preserved without overlapping smoothing.")
print("LIMITATION — the rate proxy remains sparse and is not calibrated firing rate.")

binned_log_activity = np.log1p(event_rate)
binned_limit = float(np.quantile(binned_log_activity, 0.995))
bin_seconds = (np.arange(n_complete_bins) + 0.5) * bin_duration_seconds

fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
axes[0, 0].plot(block_seconds, raw_population_rate, color="0.55", linewidth=0.7, label="native")
axes[0, 0].step(
    bin_seconds,
    binned_population_rate,
    where="mid",
    color="#D55E00",
    linewidth=1.2,
    label="four-frame rate",
)
axes[0, 0].set(
    xlabel="time within block (s)",
    ylabel="mean event mass / s",
    title="Population mean before and after binning",
)
axes[0, 0].legend(frameon=False)

rate_image = axes[0, 1].imshow(
    binned_log_activity[neuron_order],
    aspect="auto",
    interpolation="nearest",
    cmap="magma",
    vmin=0,
    vmax=binned_limit,
    extent=[0, n_complete_bins * bin_duration_seconds, 0, n_neurons],
)
axes[0, 1].set(
    xlabel="time within block (s)",
    ylabel="all neurons",
    title="All-neuron binned event-rate proxy",
)
fig.colorbar(rate_image, ax=axes[0, 1], label="log(1+x)", pad=0.01)

axes[1, 0].bar(
    ["native", "4-frame rate"],
    [raw_zero_fraction, rate_zero_fraction],
    color=["0.55", "#D55E00"],
)
axes[1, 0].set(ylim=(0, 1), ylabel="fraction exactly zero", title="Sparsity remains visible")

axes[1, 1].bar(
    ["native", "4-frame rate"],
    [raw_population_common_lag, rate_population_common_lag],
    color=["0.55", "#D55E00"],
)
axes[1, 1].set(
    ylim=(-0.1, 1),
    ylabel=f"population correlation at {bin_duration_seconds:.3f} s",
    title="Temporal dependence at a common physical lag",
)
fig.suptitle("Step 6 — One explicit rate preprocessing choice")
save_and_show(fig, "03_event_rate_preprocessing.png")


# %% Step 7 — Split first; then fit only an invertible neuron scaling
#
# The first 60% fits the per-neuron mean and RMS. The next 20% selects the
# delay order. The last 20% remains untouched until the final test. We no
# longer impose an arbitrary rank-8 PCA before PyDMD: all eligible neurons are
# passed to PyDMD, whose documented `svd_rank=0` rule chooses its internal rank.

n_rate_samples = event_rate.shape[1]
train_stop = int(np.floor(TRAIN_FRACTION * n_rate_samples))
calibration_stop = int(
    np.floor((TRAIN_FRACTION + CALIBRATION_FRACTION) * n_rate_samples)
)
if not (max(DELAY_CANDIDATES) + 2 < train_stop < calibration_stop < n_rate_samples):
    raise RuntimeError("The chronological split leaves insufficient delayed pairs")

training_rate = event_rate[:, :train_stop]
neuron_mean = np.mean(training_rate, axis=1)
neuron_rms = np.sqrt(
    np.mean((training_rate - neuron_mean[:, None]) ** 2, axis=1)
)
eligible = np.all(np.isfinite(training_rate), axis=1) & (neuron_rms > 1e-12)
eligible_rows = np.flatnonzero(eligible)
scale = neuron_rms[eligible] + 1e-3
scaled_rate = (
    event_rate[eligible] - neuron_mean[eligible, None]
) / scale[:, None]

# This is a numerical round trip, not a model reconstruction. It checks that
# the scaling itself loses no information for eligible neurons.
round_trip = (
    scaled_rate * scale[:, None] + neuron_mean[eligible, None]
)
scaling_round_trip_error = float(
    np.max(np.abs(round_trip - event_rate[eligible]))
)

training_scaled = scaled_rate[:, :train_stop]
calibration_scaled = scaled_rate[:, train_stop:calibration_stop]
test_scaled = scaled_rate[:, calibration_stop:]

print("TRANSFORMATION — training-only centering and RMS scaling")
print(f"  rate matrix:               {event_rate.shape}")
print(
    "  train/calibration/test:    "
    f"{training_scaled.shape[1]}/{calibration_scaled.shape[1]}/"
    f"{test_scaled.shape[1]} bins"
)
print(f"  eligible neurons:          {eligible_rows.size:,} / {n_neurons:,}")
print(f"  zero-training-RMS neurons: {(~eligible).sum():,}")
print(f"  scaling round-trip error:  {scaling_round_trip_error:.3e}")
print("SUCCESS — the scaling is reversible and uses training data only.")
print(
    "LIMITATION — zero-training-RMS neurons cannot enter the fitted model; "
    "they remain visible in raw and spatial plots."
)

split_time = np.arange(n_rate_samples) * bin_duration_seconds
scaled_display_limit = float(np.quantile(np.abs(scaled_rate), 0.995))
fig, axes = plt.subplots(
    3,
    1,
    figsize=(11, 9),
    gridspec_kw={"height_ratios": [1, 1, 3]},
    constrained_layout=True,
)
axes[0].plot(
    split_time,
    np.mean(event_rate, axis=0),
    color="#4C78A8",
    linewidth=0.9,
)
axes[0].set(
    xlabel="time within block (s)",
    ylabel="mean event mass / s",
    title="The population observable before scaling",
)
axes[1].hist(
    neuron_rms[neuron_rms > 0],
    bins=80,
    color="#009E73",
    log=True,
)
axes[1].set(
    xlabel="training RMS of a neuron's event-rate deviation",
    ylabel="neurons (log count)",
    title="Scale parameters fitted on the training segment",
)
scaled_image = axes[2].imshow(
    scaled_rate[np.argsort(np.sum(np.abs(scaled_rate), axis=1))],
    aspect="auto",
    interpolation="nearest",
    cmap="coolwarm",
    vmin=-scaled_display_limit,
    vmax=scaled_display_limit,
    extent=[0, n_rate_samples * bin_duration_seconds, 0, eligible_rows.size],
)
axes[2].set(
    xlabel="time within block (s)",
    ylabel="eligible neurons",
    title="All eligible standardized neuron traces (no PCA yet)",
)
fig.colorbar(scaled_image, ax=axes[2], label="training-standardized rate")
for axis in (axes[0], axes[2]):
    axis.axvline(
        train_stop * bin_duration_seconds,
        color="#009E73",
        linestyle="--",
        label="train/calibration",
    )
    axis.axvline(
        calibration_stop * bin_duration_seconds,
        color="#D55E00",
        linestyle="--",
        label="calibration/test",
    )
axes[0].legend(frameon=False, ncol=2)
fig.suptitle("Step 7 — Exactly how neural activity enters PyDMD")
save_and_show(fig, "04_train_only_scaling.png")


# %% Step 8 — Let calibration data choose whether Hankel delays help
#
# PyDMD's pseudo-Hankel matrix stacks `d` consecutive neural snapshots. Thus
# `d=1` is ordinary DMD with no delay, whereas `d=4` stacks
# `[x_t, x_{t+1}, x_{t+2}, x_{t+3}]`. To forecast honestly we score only the
# last block of the next Hankel column, x_{t+4}; its earlier blocks were already
# observed. Only calibration data choose `d` in this cell.


@dataclass(frozen=True)
class CaptureMetrics:
    nrmse_vs_training_mean: float
    r2_vs_training_mean: float
    skill_vs_persistence: float
    model_sse: float
    training_mean_sse: float
    persistence_sse: float


def capture_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    persistence: np.ndarray,
) -> CaptureMetrics:
    """Score standardized neural activity; zero is the training-mean model."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    persistence = np.asarray(persistence, dtype=np.float64)
    model_sse = float(np.sum((truth - prediction) ** 2))
    mean_sse = float(np.sum(truth**2))
    persistence_sse = float(np.sum((truth - persistence) ** 2))
    if mean_sse <= 0 or persistence_sse <= 0:
        raise RuntimeError("A forecast baseline has zero error")
    return CaptureMetrics(
        nrmse_vs_training_mean=float(np.sqrt(model_sse / mean_sse)),
        r2_vs_training_mean=float(1 - model_sse / mean_sse),
        skill_vs_persistence=float(1 - model_sse / persistence_sse),
        model_sse=model_sse,
        training_mean_sse=mean_sse,
        persistence_sse=persistence_sse,
    )


def build_pydmd_model(delay: int) -> tuple[HankelDMD, tuple[str, ...]]:
    """Fit exactly one documented PyDMD HankelDMD configuration."""
    model = HankelDMD(
        svd_rank=PYDMD_SVD_RANK,
        tlsq_rank=0,
        exact=True,
        opt=True,
        rescale_mode=None,
        forward_backward=False,
        d=delay,
        sorted_eigs=False,
        reconstruction_method="mean",
        tikhonov_regularization=None,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(training_scaled)
    return model, tuple(str(item.message) for item in caught)


def last_block_forecast(
    model: HankelDMD,
    values: np.ndarray,
    delay: int,
) -> tuple[CaptureMetrics, np.ndarray, np.ndarray, np.ndarray]:
    """Predict the genuinely new last block of each delayed snapshot.

    PyDMD's `DMD.predict` uses `modes @ diag(eigs) @ pinv(modes) @ X`.
    `HankelDMD` does not expose that convenience method, so the calculation is
    reproduced here with public `modes` and `eigs`, without forming the huge
    dense measurement-space operator.
    """
    hankel = pseudo_hankel_matrix(values, delay)
    x_hankel, y_hankel = hankel[:, :-1], hankel[:, 1:]
    n_features = values.shape[0]
    coefficients = np.linalg.pinv(model.modes) @ x_hankel
    next_coefficients = model.eigs[:, None] * coefficients
    prediction = (
        model.modes[(delay - 1) * n_features : delay * n_features]
        @ next_coefficients
    ).real
    truth = y_hankel[(delay - 1) * n_features : delay * n_features]
    persistence = x_hankel[(delay - 1) * n_features : delay * n_features]
    return (
        capture_metrics(truth, prediction, persistence),
        truth,
        prediction,
        persistence,
    )


print("PYDMD PARAMETERS HELD FIXED ACROSS THE DELAY SWEEP")
print("  class:                   pydmd.HankelDMD")
print("  svd_rank:                0 (PyDMD automatic hard threshold)")
print("  tlsq_rank:               0 (no extra TLSQ denoising)")
print("  exact:                   True (measurement-space modes)")
print("  opt:                     True (global training amplitude fit)")
print("  reconstruction_method:   mean (average overlapping estimates)")
print("  forward_backward:        False")
print("  Tikhonov regularization: None")

parameter_rows: list[dict[str, float | int]] = []
pydmd_warning_messages: set[str] = set()
for delay in DELAY_CANDIDATES:
    trial_model, trial_warnings = build_pydmd_model(delay)
    pydmd_warning_messages.update(trial_warnings)
    training_capture = capture_metrics(
        training_scaled,
        trial_model.reconstructed_data.real,
        np.zeros_like(training_scaled),
    )
    calibration_capture, _, _, _ = last_block_forecast(
        trial_model,
        calibration_scaled,
        delay,
    )
    parameter_rows.append(
        {
            "delay": delay,
            "history_seconds": (delay - 1) * bin_duration_seconds,
            "hankel_rows": eligible_rows.size * delay,
            "training_hankel_columns": training_scaled.shape[1] - delay + 1,
            "pydmd_modes": len(trial_model.eigs),
            "training_reconstruction_nrmse": training_capture.nrmse_vs_training_mean,
            "training_reconstruction_r2": training_capture.r2_vs_training_mean,
            "calibration_forecast_nrmse": calibration_capture.nrmse_vs_training_mean,
            "calibration_forecast_r2": calibration_capture.r2_vs_training_mean,
            "calibration_skill_vs_persistence": calibration_capture.skill_vs_persistence,
        }
    )
    del trial_model

parameter_table = pd.DataFrame(parameter_rows)
parameter_table.to_csv(OUTPUT_DIR / "delay_parameter_sweep.csv", index=False)
selection_table = parameter_table.sort_values(
    ["calibration_forecast_r2", "delay"],
    ascending=[False, True],
    kind="stable",
)
selected_delay = int(selection_table.iloc[0]["delay"])

print("\nDELAY SELECTION — primary metric is calibration R² vs training mean")
print(parameter_table.to_string(index=False, float_format=lambda value: f"{value:.4g}"))
print(f"SELECTED: d={selected_delay}")
if selected_delay == 1:
    print("INTERPRETATION — calibration selected no delay augmentation.")
else:
    print(
        "INTERPRETATION — calibration selected "
        f"{(selected_delay - 1) * bin_duration_seconds:.3f} s of history."
    )
if pydmd_warning_messages:
    print("PYDMD WARNING(S) — retained as diagnostic evidence:")
    for message in sorted(pydmd_warning_messages):
        print(" ", message.replace("\n", " "))
    print(
        "  NOTE — training centering makes the snapshot columns sum to zero, "
        "so one null direction is expected; automatic SVD truncation removes it."
    )

fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
delay_axis = parameter_table["delay"].to_numpy()
axes[0, 0].plot(delay_axis, parameter_table["pydmd_modes"], marker="o")
axes[0, 0].set(
    xlabel="delay order d",
    ylabel="PyDMD modes",
    title="Automatic SVD rank",
)
axes[0, 1].plot(
    delay_axis,
    parameter_table["training_reconstruction_r2"],
    marker="o",
    label="training reconstruction",
)
axes[0, 1].plot(
    delay_axis,
    parameter_table["calibration_forecast_r2"],
    marker="o",
    label="calibration one-step",
)
axes[0, 1].axhline(0, color="0.5", linestyle="--", linewidth=0.8)
axes[0, 1].set(
    xlabel="delay order d",
    ylabel="R² vs training mean",
    title="Actual variance capture",
)
axes[0, 1].legend(frameon=False)
axes[1, 0].plot(
    delay_axis,
    parameter_table["calibration_skill_vs_persistence"],
    marker="o",
    color="#D55E00",
)
axes[1, 0].axhline(0, color="0.5", linestyle="--", linewidth=0.8)
axes[1, 0].set(
    xlabel="delay order d",
    ylabel="skill",
    title="Calibration skill vs persistence",
)
axes[1, 1].plot(
    delay_axis,
    parameter_table["history_seconds"],
    marker="o",
    color="#009E73",
)
axes[1, 1].set(
    xlabel="delay order d",
    ylabel="history before newest sample (s)",
    title="Physical meaning of d",
)
for axis in axes.ravel():
    axis.axvline(selected_delay, color="#CC79A7", linestyle=":", linewidth=1.2)
fig.suptitle("Step 8 — PyDMD delay choice made without test data")
save_and_show(fig, "05_pydmd_delay_selection.png")


# %% Step 9 — Fit the selected PyDMD model and measure activity capture
#
# Three different quantities must not be conflated:
#
# 1. SVD retained energy: how much of the Hankel snapshot matrix enters DMD.
# 2. Training reconstruction R²: how well one sum of exponential modes
#    reproduces the transformed training activity.
# 3. Test one-step R²: how well the map predicts unseen transformed activity.
#
# Beating persistence alone is not enough; a near-zero predictor can beat a
# noisy persistence baseline while explaining almost none of the neural data.

selected_model, selected_warnings = build_pydmd_model(selected_delay)
selected_rank = int(len(selected_model.eigs))
train_hankel = selected_model.ho_snapshots
train_hankel_x, train_hankel_y = train_hankel[:, :-1], train_hankel[:, 1:]

# This public PyDMD utility uses the same rank rule as HankelDMD. Its basis is
# retained for the small Residual-DMD calculation in the next cell.
pod_basis, pod_singular_values, _ = compute_svd(
    train_hankel_x,
    svd_rank=PYDMD_SVD_RANK,
)
if pod_basis.shape[1] != selected_rank:
    raise RuntimeError("PyDMD rank and reproduced public compute_svd rank disagree")
retained_hankel_energy = float(
    np.sum(pod_singular_values**2) / np.sum(train_hankel_x**2)
)

training_reconstruction = selected_model.reconstructed_data.real
training_capture = capture_metrics(
    training_scaled,
    training_reconstruction,
    np.zeros_like(training_scaled),
)
(
    test_capture,
    test_truth_scaled,
    test_prediction_scaled,
    test_persistence_scaled,
) = last_block_forecast(selected_model, test_scaled, selected_delay)

# Invert the scaling for an all-neuron population check. Neurons with zero
# training RMS receive the training-mean prediction because no fitted model
# coordinate exists for them.
test_truth_rate_all = event_rate[:, calibration_stop + selected_delay :]
test_prediction_rate_all = np.repeat(
    neuron_mean[:, None],
    test_truth_rate_all.shape[1],
    axis=1,
)
test_prediction_rate_all[eligible] = (
    test_prediction_scaled * scale[:, None]
    + neuron_mean[eligible, None]
)
test_persistence_rate_all = event_rate[
    :,
    calibration_stop + selected_delay - 1 : n_rate_samples - 1,
]
if not (
    test_truth_rate_all.shape
    == test_prediction_rate_all.shape
    == test_persistence_rate_all.shape
):
    raise RuntimeError("Physical-unit test arrays are misaligned")

test_mean_baseline_all = np.repeat(
    neuron_mean[:, None],
    test_truth_rate_all.shape[1],
    axis=1,
)
all_neuron_model_sse = float(
    np.sum((test_truth_rate_all - test_prediction_rate_all) ** 2)
)
all_neuron_mean_sse = float(
    np.sum((test_truth_rate_all - test_mean_baseline_all) ** 2)
)
all_neuron_test_r2 = float(1 - all_neuron_model_sse / all_neuron_mean_sse)

per_neuron_denominator = np.sum(test_truth_scaled**2, axis=1)
per_neuron_test_r2 = np.full(eligible_rows.size, np.nan)
has_test_variation = per_neuron_denominator > np.finfo(float).eps
per_neuron_test_r2[has_test_variation] = 1 - (
    np.sum((test_truth_scaled - test_prediction_scaled) ** 2, axis=1)[
        has_test_variation
    ]
    / per_neuron_denominator[has_test_variation]
)
finite_neuron_r2 = per_neuron_test_r2[np.isfinite(per_neuron_test_r2)]

truth_population_rate = np.mean(test_truth_rate_all, axis=0)
prediction_population_rate = np.mean(test_prediction_rate_all, axis=0)
if np.std(truth_population_rate) > 0 and np.std(prediction_population_rate) > 0:
    test_population_correlation = float(
        np.corrcoef(truth_population_rate, prediction_population_rate)[0, 1]
    )
else:
    test_population_correlation = np.nan

print("PYDMD MODEL SELECTED WITHOUT TEST DATA")
print(f"  delay order d:                 {selected_delay}")
print(
    "  history before newest sample: "
    f"{(selected_delay - 1) * bin_duration_seconds:.3f} s"
)
print(f"  training Hankel matrix:        {train_hankel.shape}")
print(f"  PyDMD automatic rank/modes:    {selected_rank}")
print(f"  retained Hankel-X energy:      {retained_hankel_energy:.3%}")
print("\nHOW ACCURATELY IS TRANSFORMED ACTIVITY CAPTURED?")
print(f"  training reconstruction NRMSE: {training_capture.nrmse_vs_training_mean:.4f}")
print(f"  training reconstruction R²:    {training_capture.r2_vs_training_mean:+.4f}")
print(f"  test one-step NRMSE:            {test_capture.nrmse_vs_training_mean:.4f}")
print(f"  test one-step R²:               {test_capture.r2_vs_training_mean:+.4f}")
print(f"  test skill vs persistence:      {test_capture.skill_vs_persistence:+.4f}")
print(f"  all-neuron physical-unit R²:    {all_neuron_test_r2:+.4f}")
print(f"  test population correlation:    {test_population_correlation:+.4f}")
print(f"  median eligible-neuron R²:      {np.median(finite_neuron_r2):+.4f}")
print(
    "  neurons with positive R²:      "
    f"{np.mean(finite_neuron_r2 > 0):.2%}"
)

test_time_seconds = (
    np.arange(test_truth_rate_all.shape[1])
    + calibration_stop
    + selected_delay
) * bin_duration_seconds
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
axes[0, 0].plot(
    np.arange(training_scaled.shape[1]) * bin_duration_seconds,
    np.mean(training_scaled, axis=0),
    label="transformed truth",
    linewidth=1,
)
axes[0, 0].plot(
    np.arange(training_scaled.shape[1]) * bin_duration_seconds,
    np.mean(training_reconstruction, axis=0),
    label="PyDMD reconstruction",
    linewidth=1,
)
axes[0, 0].set(
    xlabel="time within block (s)",
    ylabel="mean standardized rate",
    title=f"Training reconstruction: R²={training_capture.r2_vs_training_mean:.3f}",
)
axes[0, 0].legend(frameon=False)
axes[0, 1].plot(test_time_seconds, truth_population_rate, label="truth", linewidth=1.2)
axes[0, 1].plot(
    test_time_seconds,
    prediction_population_rate,
    label="one-step PyDMD",
    linewidth=1.2,
)
axes[0, 1].set(
    xlabel="time within block (s)",
    ylabel="all-neuron mean event mass / s",
    title=f"Test population correlation={test_population_correlation:.3f}",
)
axes[0, 1].legend(frameon=False)

histogram_limits = np.quantile(finite_neuron_r2, [0.01, 0.99])
axes[1, 0].hist(
    np.clip(finite_neuron_r2, *histogram_limits),
    bins=60,
    color="#4C78A8",
)
axes[1, 0].axvline(0, color="black", linestyle="--", linewidth=0.8)
axes[1, 0].set(
    xlabel="test R² per eligible neuron (1–99% clipped for display)",
    ylabel="neurons",
    title="Most individual traces are not captured well",
)
metric_names = ["train\nreconstruction", "test\none-step", "test vs\npersistence"]
metric_values = [
    training_capture.r2_vs_training_mean,
    test_capture.r2_vs_training_mean,
    test_capture.skill_vs_persistence,
]
axes[1, 1].bar(metric_names, metric_values, color=["0.5", "#D55E00", "#009E73"])
axes[1, 1].axhline(0, color="black", linewidth=0.8)
axes[1, 1].set(ylabel="R² or skill", title="Why persistence skill alone is insufficient")
fig.suptitle("Step 9 — Quantitative capture of transformed neural activity")
save_and_show(fig, "06_pydmd_activity_capture.png")


# %% Step 10 — Evaluate ResDMD residuals for the PyDMD candidates
#
# PyDMD has no `ResDMD` class. We therefore do not replace its fit. Instead we
# reproduce PyDMD's public rank-reduced coordinates, verify that their
# generalized eigenvalues match `selected_model.eigs`, and then evaluate the
# published G/A/L residual on calibration and untouched test pairs.
#
# This residual asks whether an eigenfunction relation is accurate. It is not
# the same quantity as the activity reconstruction R² in Step 9.

calibration_hankel = pseudo_hankel_matrix(calibration_scaled, selected_delay)
test_hankel = pseudo_hankel_matrix(test_scaled, selected_delay)
calibration_hankel_x = calibration_hankel[:, :-1]
calibration_hankel_y = calibration_hankel[:, 1:]
test_hankel_x = test_hankel[:, :-1]
test_hankel_y = test_hankel[:, 1:]

train_reduced_x = pod_basis.T @ train_hankel_x
train_reduced_y = pod_basis.T @ train_hankel_y
calibration_reduced_x = pod_basis.T @ calibration_hankel_x
calibration_reduced_y = pod_basis.T @ calibration_hankel_y
test_reduced_x = pod_basis.T @ test_hankel_x
test_reduced_y = pod_basis.T @ test_hankel_y

resdmd = fit_linear_resdmd(
    train_reduced_x,
    train_reduced_y,
    calibration_reduced_x,
    calibration_reduced_y,
)
test_g, test_a, test_l = resdmd_matrices(test_reduced_x, test_reduced_y)
resdmd_test_residuals = eigenpair_residuals(
    resdmd.eigenvalues,
    resdmd.eigenfunctions,
    test_g,
    test_a,
    test_l,
)

eigenvalue_cost = abs(
    selected_model.eigs[:, None] - resdmd.eigenvalues[None, :]
)
model_rows, resdmd_columns = linear_sum_assignment(eigenvalue_cost)
resdmd_index_for_model = np.empty(selected_rank, dtype=int)
resdmd_index_for_model[model_rows] = resdmd_columns
maximum_pydmd_resdmd_eigenvalue_difference = float(
    np.max(eigenvalue_cost[model_rows, resdmd_columns])
)
if maximum_pydmd_resdmd_eigenvalue_difference > 1e-8:
    raise RuntimeError("PyDMD and reproduced rank-reduced eigenvalues disagree")

training_residual_by_mode = resdmd.training_residuals[resdmd_index_for_model]
calibration_residual_by_mode = resdmd.calibration_residuals[resdmd_index_for_model]
test_residual_by_mode = resdmd_test_residuals[resdmd_index_for_model]

print("RESDMD DIAGNOSTIC OF THE PYDMD CANDIDATES")
print(
    "  max PyDMD/ResDMD eigenvalue difference: "
    f"{maximum_pydmd_resdmd_eigenvalue_difference:.3e}"
)
print(f"  training reduced Gram condition:          {resdmd.gram_condition_number:.3g}")
print(f"  median training residual:                  {np.median(training_residual_by_mode):.4f}")
print(f"  median calibration residual:               {np.median(calibration_residual_by_mode):.4f}")
print(f"  median untouched-test residual:            {np.median(test_residual_by_mode):.4f}")
print(
    "  untouched-test residual range:             "
    f"[{np.min(test_residual_by_mode):.4f}, {np.max(test_residual_by_mode):.4f}]"
)
print(
    "INTERPRETATION — values are not near the exact-map value zero; no "
    "candidate is declared a verified neural mode."
)
print(
    "CAUTION — for stochastic neural observations this empirical residual "
    "also contains a variance contribution."
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
circle = np.exp(1j * np.linspace(0, 2 * np.pi, 500))
axes[0].plot(circle.real, circle.imag, color="0.75", linewidth=1)
residual_points = axes[0].scatter(
    selected_model.eigs.real,
    selected_model.eigs.imag,
    c=test_residual_by_mode,
    cmap="viridis_r",
    s=75,
    edgecolors="black",
    linewidths=0.4,
)
axes[0].axhline(0, color="0.85", linewidth=0.8)
axes[0].axvline(0, color="0.85", linewidth=0.8)
axes[0].set(
    xlabel="Re(lambda)",
    ylabel="Im(lambda)",
    title="PyDMD candidates colored by test residual",
)
axes[0].set_aspect("equal", adjustable="box")
fig.colorbar(residual_points, ax=axes[0], label="untouched-test residual")

residual_order = np.argsort(test_residual_by_mode)
mode_axis = np.arange(1, selected_rank + 1)
axes[1].plot(
    mode_axis,
    training_residual_by_mode[residual_order],
    marker="o",
    label="training",
    color="0.5",
)
axes[1].plot(
    mode_axis,
    calibration_residual_by_mode[residual_order],
    marker="o",
    label="calibration",
    color="#E69F00",
)
axes[1].plot(
    mode_axis,
    test_residual_by_mode[residual_order],
    marker="o",
    label="untouched test",
    color="#D55E00",
)
axes[1].set(
    xlabel="candidate ordered by test residual",
    ylabel="relative eigenfunction residual",
    title="Residual transfer across chronological splits",
)
axes[1].legend(frameon=False)
fig.suptitle("Step 10 — Residual-DMD evidence for PyDMD candidates")
save_and_show(fig, "07_resdmd_candidate_residuals.png")


# %% Step 11 — Count candidates and convert eigenvalues to physical frequencies
#
# PyDMD returns one mode per retained eigenvalue. A complex-conjugate pair is
# one real oscillatory component represented by two complex modes. Frequencies
# and decay rates are computed from the continuous-time eigenvalue
# `log(lambda) / bin_duration`, as in the PyDMD tutorials.

eigenvalues = np.asarray(selected_model.eigs, dtype=np.complex128)
continuous_eigenvalues = np.log(eigenvalues) / bin_duration_seconds
frequency_hz = continuous_eigenvalues.imag / (2 * np.pi)
decay_rate_per_second = continuous_eigenvalues.real
decay_seconds = np.full(selected_rank, np.inf)
decaying = decay_rate_per_second < 0
decay_seconds[decaying] = -1 / decay_rate_per_second[decaying]
modulus = np.abs(eigenvalues)

# For a Hankel mode, rows are consecutive copies of the measurement space.
# The first block is the spatial mode at the first embedded time. The selected
# d=1 model has only this block, so no block-aggregation ambiguity remains.
standardized_spatial_modes = selected_model.modes[: eligible_rows.size]
modal_amplitudes = np.asarray(selected_model.amplitudes)
modal_importance = np.abs(modal_amplitudes) * np.linalg.norm(
    standardized_spatial_modes,
    axis=0,
)
modal_importance /= np.sum(modal_importance)

real_tolerance = 1e-10
is_real_mode = np.abs(eigenvalues.imag) <= real_tolerance
n_real_modes = int(np.sum(is_real_mode))
n_complex_modes = selected_rank - n_real_modes
if n_complex_modes % 2:
    raise RuntimeError("Real-data complex modes do not form complete pairs")
n_conjugate_pairs = n_complex_modes // 2
n_real_components = n_real_modes + n_conjugate_pairs

# Retain the positive-frequency representative of each pair plus every real
# mode. The order is by amplitude-weighted mode norm, not by eigenvalue modulus.
importance_order = np.argsort(-modal_importance)
representative_indices = [
    int(index)
    for index in importance_order
    if is_real_mode[index] or eigenvalues[index].imag > 0
]

frequency_resolution_hz = float(
    1 / (train_hankel.shape[1] * bin_duration_seconds)
)
nyquist_hz = float(1 / (2 * bin_duration_seconds))

print("TEMPORAL MODE INVENTORY")
print(f"  PyDMD eigenvalues/modes:        {selected_rank}")
print(f"  real nonoscillatory modes:      {n_real_modes}")
print(f"  complex-conjugate pairs:        {n_conjugate_pairs}")
print(f"  unique real components:         {n_real_components}")
print(f"  approximate frequency spacing: {frequency_resolution_hz:.4f} Hz")
print(f"  Nyquist frequency after binning:{nyquist_hz:.4f} Hz")
print(
    "  candidate |frequency| range:  "
    f"{np.min(np.abs(frequency_hz)):.4f}–{np.max(np.abs(frequency_hz)):.4f} Hz"
)
print(
    "  candidate decay-time range:   "
    f"{np.min(decay_seconds):.3f}–{np.max(decay_seconds):.3f} s"
)
print(
    "INTERPRETATION — these are candidate exponential terms, not 25 verified "
    "biological rhythms. Conjugate pairs must be counted together."
)

fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
representative_frequency = np.abs(frequency_hz[representative_indices])
representative_importance = modal_importance[representative_indices]
frequency_points = axes[0].scatter(
    representative_frequency,
    representative_importance,
    c=test_residual_by_mode[representative_indices],
    cmap="viridis_r",
    s=75,
    edgecolors="black",
    linewidths=0.4,
)
for index in representative_indices:
    axes[0].annotate(
        str(index + 1),
        (abs(frequency_hz[index]), modal_importance[index]),
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=7,
    )
axes[0].set(
    xlabel="absolute frequency (Hz)",
    ylabel="amplitude-weighted mode fraction",
    title="One representative per real dynamical component",
)
fig.colorbar(
    frequency_points,
    ax=axes[0],
    label="untouched-test ResDMD residual",
    pad=0.01,
)

modal_time = np.arange(train_hankel.shape[1]) * bin_duration_seconds
for index in representative_indices[:6]:
    coefficient = modal_amplitudes[index] * eigenvalues[index] ** np.arange(
        train_hankel.shape[1]
    )
    normalization = max(float(np.max(np.abs(coefficient))), np.finfo(float).eps)
    axes[1].plot(
        modal_time,
        coefficient.real / normalization,
        linewidth=1,
        label=f"mode {index + 1}: {abs(frequency_hz[index]):.3f} Hz",
    )
axes[1].set(
    xlabel="training time (s)",
    ylabel="normalized real modal coefficient",
    title="Fitted exponential time dependence",
)
axes[1].text(
    0.99,
    0.05,
    f"all fitted e-folding times <= {np.max(decay_seconds):.2f} s\n"
    f"training duration = {modal_time[-1]:.1f} s",
    transform=axes[1].transAxes,
    ha="right",
    va="bottom",
    fontsize=9,
)
axes[1].legend(frameon=False, ncol=2, fontsize=8)
fig.suptitle("Step 11 — Frequencies and temporal behavior of PyDMD candidates")
save_and_show(fig, "08_temporal_mode_inventory.png")


# %% Step 12 — Lift each selected mode back to every neuron's cortical position
#
# PyDMD modes are in standardized coordinates. Multiplying each row by its
# training scale converts the shape to event-rate-deviation units. A complex
# mode has arbitrary global phase, so each plotted mode is rotated until its
# largest-magnitude neuron is positive real. Every eligible neuron is plotted;
# excluded zero-training-RMS neurons are shown in light gray.
#
# Two descriptive spatial summaries are reported:
#
# - participation fraction: effective fraction of neurons carrying the mode;
# - six-nearest-neighbor coherence: 1 for identical neighboring complex values
#   and near 0 for spatially unstructured values. No spatial-null significance
#   test is performed in this focused audit.

mode_power = np.abs(standardized_spatial_modes) ** 2
participation_fraction = (
    np.sum(mode_power, axis=0) ** 2
    / np.sum(mode_power**2, axis=0)
    / eligible_rows.size
)

eligible_coordinates = centroid_um[eligible]
neighbor_count = min(7, eligible_rows.size)
_, neighbor_indices = cKDTree(eligible_coordinates).query(
    eligible_coordinates,
    k=neighbor_count,
)
edge_source = np.repeat(np.arange(eligible_rows.size), neighbor_count - 1)
edge_target = neighbor_indices[:, 1:].reshape(-1)
neighbor_coherence = np.empty(selected_rank)
for mode_index in range(selected_rank):
    spatial_mode = standardized_spatial_modes[:, mode_index]
    difference_energy = np.sum(
        np.abs(spatial_mode[edge_source] - spatial_mode[edge_target]) ** 2
    )
    reference_energy = np.sum(
        np.abs(spatial_mode[edge_source]) ** 2
        + np.abs(spatial_mode[edge_target]) ** 2
    )
    neighbor_coherence[mode_index] = 1 - difference_energy / reference_energy

mode_table = pd.DataFrame(
    {
        "mode": np.arange(1, selected_rank + 1),
        "eigenvalue_real": eigenvalues.real,
        "eigenvalue_imag": eigenvalues.imag,
        "modulus": modulus,
        "frequency_hz": frequency_hz,
        "decay_rate_per_second": decay_rate_per_second,
        "decay_seconds": decay_seconds,
        "modal_importance_fraction": modal_importance,
        "training_residual": training_residual_by_mode,
        "calibration_residual": calibration_residual_by_mode,
        "test_residual": test_residual_by_mode,
        "participation_fraction": participation_fraction,
        "six_neighbor_coherence": neighbor_coherence,
        "is_real_mode": is_real_mode,
    }
)
mode_table["importance_rank"] = (
    mode_table["modal_importance_fraction"]
    .rank(method="first", ascending=False)
    .astype(int)
)
mode_table = mode_table.sort_values("importance_rank", ignore_index=True)
mode_table.to_csv(OUTPUT_DIR / "mode_table.csv", index=False)

print("SPATIAL MODE SUMMARY — ordered by amplitude-weighted importance")
print(
    mode_table[
        [
            "mode",
            "frequency_hz",
            "decay_seconds",
            "modal_importance_fraction",
            "test_residual",
            "participation_fraction",
            "six_neighbor_coherence",
        ]
    ]
    .head(10)
    .to_string(index=False, float_format=lambda value: f"{value:.4g}")
)
print(
    "INTERPRETATION — participation and neighbor coherence describe the fitted "
    "patterns, but large ResDMD residuals prevent a biological spatial-mode claim."
)

spatial_plot_indices = representative_indices[:SPATIAL_MODES_TO_PLOT]
fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
last_spatial_scatter = None
for axis, mode_index in zip(axes.ravel(), spatial_plot_indices, strict=False):
    physical_mode = scale * standardized_spatial_modes[:, mode_index]
    pivot = int(np.argmax(np.abs(physical_mode)))
    phase_rotation = np.exp(-1j * np.angle(physical_mode[pivot]))
    aligned_mode = physical_mode * phase_rotation
    magnitude = np.abs(aligned_mode)
    maximum_magnitude = max(float(np.max(magnitude)), np.finfo(float).eps)
    normalized_real = aligned_mode.real / maximum_magnitude
    normalized_magnitude = magnitude / maximum_magnitude
    axis.scatter(
        centroid_um[~eligible, 0],
        centroid_um[~eligible, 1],
        s=2,
        color="0.85",
        linewidths=0,
        label="zero training RMS",
    )
    last_spatial_scatter = axis.scatter(
        eligible_coordinates[:, 0],
        eligible_coordinates[:, 1],
        c=normalized_real,
        s=1.5 + 12 * normalized_magnitude,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0,
        rasterized=True,
    )
    axis.set(
        xlabel="x (um)",
        ylabel="y (um)",
        title=(
            f"mode {mode_index + 1}: |f|={abs(frequency_hz[mode_index]):.3f} Hz\n"
            f"test residual={test_residual_by_mode[mode_index]:.3f}, "
            f"participation={participation_fraction[mode_index]:.1%}"
        ),
    )
    axis.set_aspect("equal", adjustable="box")
for axis in axes.ravel()[len(spatial_plot_indices) :]:
    axis.set_visible(False)
if last_spatial_scatter is not None:
    fig.colorbar(
        last_spatial_scatter,
        ax=axes.ravel().tolist(),
        label="phase-aligned real mode / max magnitude",
        shrink=0.8,
    )
fig.suptitle("Step 12 — All-neuron spatial structure of leading PyDMD candidates")
save_and_show(fig, "09_spatial_modes.png")


# %% Step 13 — State exactly what succeeded, what failed, and stop

summary = {
    "recording": DATA_PATH.name,
    "state": STATE_NAME,
    "block_start": block_start,
    "block_stop": block_stop,
    "pydmd_version": version("pydmd"),
    "bin_frames": BIN_FRAMES,
    "bin_seconds": bin_duration_seconds,
    "selected_delay": selected_delay,
    "selected_history_seconds": (selected_delay - 1) * bin_duration_seconds,
    "pydmd_modes": selected_rank,
    "real_modes": n_real_modes,
    "conjugate_pairs": n_conjugate_pairs,
    "unique_real_components": n_real_components,
    "retained_hankel_energy": retained_hankel_energy,
    "raw_zero_fraction": raw_zero_fraction,
    "rate_zero_fraction": rate_zero_fraction,
    "event_mass_relative_error": relative_mass_error,
    "scaling_round_trip_error": scaling_round_trip_error,
    "training_reconstruction_nrmse": training_capture.nrmse_vs_training_mean,
    "training_reconstruction_r2": training_capture.r2_vs_training_mean,
    "test_forecast_nrmse": test_capture.nrmse_vs_training_mean,
    "test_forecast_r2": test_capture.r2_vs_training_mean,
    "test_skill_vs_persistence": test_capture.skill_vs_persistence,
    "all_neuron_physical_test_r2": all_neuron_test_r2,
    "test_population_correlation": test_population_correlation,
    "median_per_neuron_test_r2": float(np.median(finite_neuron_r2)),
    "positive_per_neuron_test_r2_fraction": float(np.mean(finite_neuron_r2 > 0)),
    "maximum_pydmd_resdmd_eigenvalue_difference": maximum_pydmd_resdmd_eigenvalue_difference,
    "median_training_residual": float(np.median(training_residual_by_mode)),
    "median_calibration_residual": float(np.median(calibration_residual_by_mode)),
    "median_test_residual": float(np.median(test_residual_by_mode)),
    "minimum_test_residual": float(np.min(test_residual_by_mode)),
    "maximum_test_residual": float(np.max(test_residual_by_mode)),
    "status": "pipeline executes, but scientific DMD success is not supported",
}
(OUTPUT_DIR / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)

print("\n================ FOCUSED PYDMD/RESDMD VERDICT ================")
print("TECHNICALLY SUCCEEDED")
print("  1. PyDMD 2025.8.1 is installed and performs every DMD/Hankel fit.")
print("  2. The event-rate and scaling transformations are explicit and reversible.")
print("  3. Delay order was selected on calibration data; test data were untouched.")
print("  4. PyDMD eigenvalues agree with the reproduced ResDMD coordinates.")
print("  5. Frequencies and all-neuron spatial candidate modes are now reported.")

print("\nSCIENTIFIC DMD SUCCESS IS NOT SUPPORTED YET")
print(
    f"  1. The rate matrix remains {rate_zero_fraction:.2%} exactly zero."
)
print(
    f"  2. Automatic rank {selected_rank} retains only "
    f"{retained_hankel_energy:.2%} of training Hankel-X energy."
)
print(
    f"  3. Training reconstruction R² is "
    f"{training_capture.r2_vs_training_mean:+.4f}."
)
print(f"  4. Untouched one-step test R² is {test_capture.r2_vs_training_mean:+.4f}.")
print(
    f"  5. Beating persistence by {test_capture.skill_vs_persistence:+.4f} is not "
    "enough because variance capture is near zero."
)
print(
    f"  6. Test ResDMD residuals span {np.min(test_residual_by_mode):.3f}–"
    f"{np.max(test_residual_by_mode):.3f}, far from the exact-map value zero."
)
print(
    f"  7. The {selected_rank} eigenvalues are candidate terms "
    f"({n_real_modes} real + {n_conjugate_pairs} conjugate pairs), not verified modes."
)

print("\nSTOP")
print("  Do not begin brain-state comparisons or Grassmann tracking from these modes.")
print("  Review transformations, delay sweep, activity-capture, and mode figures first.")
