"""Systematically tune a local PyDMD model on one mouse02 NREM dataset.

Run this file one ``# %%`` cell at a time in VS Code, Spyder, or another
cell-aware editor. Running the full file is also supported as a reproducible
verification. Every cell states what was done, what the output means, and what
would count as failure before proceeding.

This script answers a narrow pipeline question. It does not compare brain
states and it does not yet perform sliding-window mode tracking.
"""

# %% Step 0a — Imports
from __future__ import annotations

import json
import hashlib
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
from pydmd import BOPDMD, HankelDMD
from pydmd.preprocessing import hankel_preprocessing
from pydmd.utils import pseudo_hankel_matrix
from scipy.sparse.linalg import svds
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm


# %% Step 0b — The correction to the previous window description
#
# The previous tutorial selected 1,500 native frames (196.08 s), but its DMD
# model was fitted to only the first 225 four-frame bins (117.65 s). The next
# 39.22 s selected ``d`` and the final 39.22 s was the test segment. Therefore
# the relevant old DMD fitting duration was 117.65 s, not 196.08 s.
#
# Its "calibration R²" was also easy to misread. It was
#
#     1 - sum((future standardized activity - one-step prediction)^2)
#         / sum((future standardized activity - training mean)^2).
#
# It was a held-out one-step predictive score pooled over neurons and times.
# It was not probabilistic calibration, autonomous trajectory reconstruction,
# a ResDMD residual, or evidence that modes were stable. Values for
# d={1,2,4,8,16} were all essentially zero, so choosing d=1 did not establish
# that delay coordinates were unnecessary. Automatic SVD rank also changed
# with d, confounding that sweep.

OLD_SELECTED_BLOCK_SECONDS = 1500 / 7.65
OLD_DMD_FIT_SECONDS = 225 * 4 / 7.65
OLD_CALIBRATION_SECONDS = 75 * 4 / 7.65

print("CORRECTION — three durations in the previous tutorial")
print(f"  selected raw block: {OLD_SELECTED_BLOCK_SECONDS:.2f} s")
print(f"  actual DMD fit:      {OLD_DMD_FIT_SECONDS:.2f} s")
print(f"  calibration/test:    {OLD_CALIBRATION_SECONDS:.2f} s each")


# %% Step 1 — Freeze the validation design before looking at outcomes
#
# W is the number of binned samples used to fit one local DMD model.
# d is the number of snapshots in each delay vector; its memory is (d-1)dt.
# r is the retained DMD rank. They are distinct quantities and are tuned
# jointly. Larger d can reveal memory in a partially observed process, but it
# also removes d-1 columns and increases the ambient dictionary dimension.
# It is therefore not a free cure for a short W.
#
# Literature-informed choices:
# - inspect delayed singular support and tune explicit rank rather than letting
#   ``svd_rank=0`` silently change rank with d;
# - use chronological targets and never split overlapping Hankel columns at
#   random;
# - compare autonomous multi-horizon forecasts with mean, persistence, and
#   neuronwise AR(1) baselines;
# - prefer the shortest W only when its validation error is within one standard
#   error of the best result;
# - keep an outer time block score-locked until the development gate passes.
#
# Primary references:
# https://pydmd.github.io/PyDMD/tutorial1dmd.html
# https://pydmd.github.io/PyDMD/tutorial2advdmd.html
# https://pydmd.github.io/PyDMD/hankeldmd.html
# https://doi.org/10.1137/18M1192329
# https://doi.org/10.1017/jfm.2022.1052
# https://doi.org/10.1007/s11071-023-09135-w

SEED = 20260717
FS_HZ = 7.65
BIN_FRAMES = 4
DT_SECONDS = BIN_FRAMES / FS_HZ
STATE_CODE = 1.0
STATE_NAME = "NREM"

# The grid brackets the old 117.65-s fit. Twenty through sixty seconds are the
# most relevant future local-window candidates; 90--180 s test whether the
# gain in transitions outweighs poorer local stationarity.
WINDOW_SECONDS = (20, 30, 45, 60, 90, 120, 180)
DELAY_CANDIDATES = (1, 2, 4, 8, 16)
RANK_CANDIDATES = (2, 4, 8, 12, 16, 24, 32)

# An investigator-chosen guard, not a DMD theorem. The ambient delayed matrix
# remains overparameterized, but each fitted operator must have at least eight
# snapshot transitions per retained rank. Unsafe combinations are displayed
# and saved, but cannot win model selection.
MIN_TRANSITIONS_PER_RANK = 8.0

# The same eight development target blocks are used for every W,d,r. The last
# 80 bins are not scored unless a development configuration passes the gate.
FORECAST_HORIZON_BINS = 8
SCORED_HORIZONS = (1, 2, 4, 8)
MAX_WINDOW_BINS = int(round(max(WINDOW_SECONDS) / DT_SECONDS))
DEVELOPMENT_ORIGINS = tuple(range(MAX_WINDOW_BINS, 429, 12))
OUTER_ORIGINS = (436, 452, 468, 484, 500)
OUTER_START_BIN = 436

# A fixed POD observation space is learned from acquisition 1, before the
# acquisition-2 tuning bout. The earlier q=64 choice made the declared
# R^2=0.05 target mathematically unreachable after projection. We therefore
# audit nested q={64,128,256} projection ceilings on development targets and
# use q=256 for this corrected run. This is a representation-capacity audit,
# not a DMD forecast: it is allowed to use development truth but never outer
# truth. The DMD rank is tuned independently below.
POD_CAPACITY_DIMENSIONS = (64, 128, 256)
POD_DIMENSION = 256

# The development gate is deliberately qualitative: useful prediction must be
# positive on at least 75% of origins against both a fixed acquisition-1 mean
# and the candidate window's past-only local mean, beat the dynamic baselines
# on average, and remain positive at horizons 1 and 8. Local-mean R² >= 0.05 is
# also reported as a practical incremental-effect target, but it is an
# investigator choice, not a literature-standard cutoff.
MIN_POSITIVE_ORIGIN_FRACTION = 0.75
PRACTICAL_R2_TARGET = 0.05

DMD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DMD_ROOT.parent
DATA_PATH = REPO_ROOT / "data" / "raw" / "mouse02_sleep.mat"
OUTPUT_DIR = DMD_ROOT / "results" / "06_window_parameter_tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCRIPT_START_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
RUN_STARTED_AT = time.time()
RUN_STATUS_PATH = OUTPUT_DIR / "run_status.json"
with RUN_STATUS_PATH.open("w", encoding="utf-8") as stream:
    json.dump(
        {
            "status": "running",
            "run_started_unix_seconds": RUN_STARTED_AT,
            "script": Path(__file__).name,
            "script_sha256": SCRIPT_START_SHA256,
        },
        stream,
        indent=2,
    )

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
    """Save every intermediate figure and show it in an interactive run."""
    path = OUTPUT_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()
    plt.close(fig)


WINDOW_BINS = {
    seconds: int(round(seconds / DT_SECONDS)) for seconds in WINDOW_SECONDS
}

print("LOCKED GRID")
for seconds, bins in WINDOW_BINS.items():
    print(f"  W={seconds:3d} s -> {bins:3d} bins = {bins * DT_SECONDS:7.3f} s")
print(
    "  d histories: "
    + ", ".join(
        f"d={delay}: {(delay - 1) * DT_SECONDS:.3f} s"
        for delay in DELAY_CANDIDATES
    )
)
print(f"  r candidates: {RANK_CANDIDATES}")
print(f"  PyDMD version: {version('pydmd')}")


# %% Step 2 — Discover state runs and acquisition boundaries from the file
#
# No paper-specific ``used_frame`` mask is used. Every selected interval is
# defined only by the raw state code and acquisition boundary. This prevents a
# delay vector or a four-frame rate bin from crossing a state transition or a
# recording restart.


def state_runs(
    state_values: np.ndarray,
    segment_starts: np.ndarray,
    segment_stops: np.ndarray,
) -> list[tuple[int, int, float, int]]:
    """Return constant-label runs, split at every acquisition boundary."""
    runs: list[tuple[int, int, float, int]] = []
    for segment, (segment_start, segment_stop) in enumerate(
        zip(segment_starts, segment_stops, strict=True)
    ):
        values = state_values[segment_start:segment_stop]
        changes = np.r_[0, np.flatnonzero(np.diff(values) != 0) + 1, values.size]
        for local_start, local_stop in zip(changes[:-1], changes[1:], strict=True):
            start = int(segment_start + local_start)
            stop = int(segment_start + local_stop)
            runs.append((start, stop, float(state_values[start]), segment))
    return runs


if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)

with h5py.File(DATA_PATH, "r") as mat:
    state = np.asarray(mat["state"]).ravel(order="F").astype(float)
    raw_boundaries = np.asarray(mat["frame/boundary_ind"]).ravel(order="F")
    signal_shape = tuple(mat["spike_deconv"].shape)

n_frames = int(state.size)
segment_stops = np.unique(np.rint(raw_boundaries).astype(int))
segment_stops = segment_stops[(segment_stops > 0) & (segment_stops <= n_frames)]
if segment_stops.size == 0 or segment_stops[-1] != n_frames:
    segment_stops = np.r_[segment_stops, n_frames]
segment_starts = np.r_[0, segment_stops[:-1]]

all_runs = state_runs(state, segment_starts, segment_stops)
nrem_runs = [run for run in all_runs if np.isclose(run[2], STATE_CODE)]
representation_runs = [run for run in nrem_runs if run[3] == 0]
acquisition_two_runs = [run for run in nrem_runs if run[3] == 1]
if not representation_runs or not acquisition_two_runs:
    raise RuntimeError("Both acquisitions need raw-code-1 NREM bouts")

target_run = max(acquisition_two_runs, key=lambda run: run[1] - run[0])
target_start, target_stop, _, target_segment = target_run
target_complete_bins = (target_stop - target_start) // BIN_FRAMES

if target_complete_bins < OUTER_ORIGINS[-1] + FORECAST_HORIZON_BINS:
    raise RuntimeError("The longest NREM bout cannot support the locked outer targets")
if DEVELOPMENT_ORIGINS[-1] + FORECAST_HORIZON_BINS > OUTER_START_BIN:
    raise RuntimeError("A development forecast enters the outer block")

run_rows = []
for run_id, (start, stop, code, segment) in enumerate(nrem_runs, start=1):
    run_rows.append(
        {
            "run_id": run_id,
            "acquisition": segment + 1,
            "start_frame": start,
            "stop_frame": stop,
            "state_code": code,
            "duration_seconds": (stop - start) / FS_HZ,
            "complete_rate_bins": (stop - start) // BIN_FRAMES,
            "representation_training": segment == 0,
            "tuning_target_run": (start, stop) == (target_start, target_stop),
        }
    )
run_table = pd.DataFrame(run_rows)
run_table.to_csv(OUTPUT_DIR / "nrem_bout_geometry.csv", index=False)

print("DATA GEOMETRY")
print(f"  recording: {DATA_PATH.name}")
print(f"  signal dataset shape: {signal_shape}")
print(f"  acquisition stops: {segment_stops.tolist()}")
print(f"  NREM bouts: {len(nrem_runs)}")
print(f"  representation bouts (acquisition 1): {len(representation_runs)}")
print(
    f"  tuning/outer bout: [{target_start}, {target_stop}), "
    f"{(target_stop - target_start) / FS_HZ:.2f} s, "
    f"{target_complete_bins} complete bins"
)
print(
    f"  development targets end before bin {OUTER_START_BIN}; "
    f"outer duration = {(target_complete_bins - OUTER_START_BIN) * DT_SECONDS:.2f} s"
)

time_minutes = np.arange(n_frames) / FS_HZ / 60
fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
axes[0].step(time_minutes, state, where="post", color="#4C78A8", linewidth=0.8)
for start, stop, _, _ in representation_runs:
    axes[0].axvspan(
        start / FS_HZ / 60,
        stop / FS_HZ / 60,
        color="#009E73",
        alpha=0.18,
    )
axes[0].axvspan(
    target_start / FS_HZ / 60,
    target_stop / FS_HZ / 60,
    color="#E69F00",
    alpha=0.35,
)
for boundary in segment_stops[:-1]:
    axes[0].axvline(boundary / FS_HZ / 60, color="0.25", linestyle="--")
axes[0].set(
    xlabel="recording time (min)",
    ylabel="raw state code",
    title="Representation bouts (green) and the longest acquisition-2 NREM bout (orange)",
)
axes[0].set_yticks([0, 0.5, 1, 2], ["awake", "quiet awake", "NREM", "REM"])

target_time = np.arange(target_complete_bins) * DT_SECONDS
axes[1].axvspan(
    0,
    OUTER_START_BIN * DT_SECONDS,
    color="#56B4E9",
    alpha=0.18,
    label="development context/targets",
)
axes[1].axvspan(
    OUTER_START_BIN * DT_SECONDS,
    target_complete_bins * DT_SECONDS,
    color="#D55E00",
    alpha=0.24,
    label="score-locked outer block",
)
for origin in DEVELOPMENT_ORIGINS:
    axes[1].hlines(
        1,
        target_time[origin],
        (origin + FORECAST_HORIZON_BINS) * DT_SECONDS,
        color="#0072B2",
        linewidth=5,
    )
for origin in OUTER_ORIGINS:
    axes[1].hlines(
        0,
        target_time[origin],
        (origin + FORECAST_HORIZON_BINS) * DT_SECONDS,
        color="#D55E00",
        linewidth=5,
    )
axes[1].set(
    xlim=(0, target_complete_bins * DT_SECONDS),
    ylim=(-0.7, 1.7),
    xlabel="time within longest NREM bout (s)",
    ylabel="forecast blocks",
    title="Common, non-overlapping target blocks; fitting contexts end at each block",
    yticks=[0, 1],
    yticklabels=["outer", "development"],
)
axes[1].legend(frameon=False, loc="upper left")
fig.suptitle("Step 2 — Chronological geometry fixed before model fitting")
save_and_show(fig, "00_chronological_validation_geometry.png")


# %% Step 3 — Convert deconvolved events to a rate proxy bout by bout
#
# Four native frames are summed and divided by their duration. No bin crosses
# a state or acquisition boundary. As before, this is event mass per second,
# not calibrated spikes/s. The temporal preprocessing is held fixed while W,d,r
# are tuned; changing bin width in the same sweep would create another major
# confound.


def read_binned_rate(
    dataset: h5py.Dataset,
    start: int,
    stop: int,
    total_frames: int,
) -> np.ndarray:
    """Read one interval as neurons x frames and bin only complete groups."""
    if dataset.shape[0] == total_frames:
        values = np.asarray(dataset[start:stop, :], dtype=np.float64).T
    elif dataset.shape[1] == total_frames:
        values = np.asarray(dataset[:, start:stop], dtype=np.float64)
    else:
        raise ValueError(f"Unexpected spike_deconv shape: {dataset.shape}")
    n_bins = values.shape[1] // BIN_FRAMES
    values = values[:, : n_bins * BIN_FRAMES]
    return (
        values.reshape(values.shape[0], n_bins, BIN_FRAMES).sum(axis=2)
        / DT_SECONDS
    )


with h5py.File(DATA_PATH, "r") as mat:
    event_dataset = mat["spike_deconv"]
    representation_chunks = [
        read_binned_rate(event_dataset, start, stop, n_frames)
        for start, stop, _, _ in representation_runs
    ]
    target_rate = read_binned_rate(
        event_dataset, target_start, target_stop, n_frames
    )

representation_rate = np.concatenate(representation_chunks, axis=1)
n_neurons = int(representation_rate.shape[0])
if target_rate.shape != (n_neurons, target_complete_bins):
    raise RuntimeError(f"Unexpected target matrix shape: {target_rate.shape}")

representation_mean = np.mean(representation_rate, axis=1)
representation_rms = np.sqrt(
    np.mean(
        (representation_rate - representation_mean[:, None]) ** 2,
        axis=1,
    )
)
eligible = np.isfinite(representation_rms) & (representation_rms > 1e-12)
eligible_rows = np.flatnonzero(eligible)
scale = representation_rms[eligible] + 1e-3

representation_standardized = (
    representation_rate[eligible] - representation_mean[eligible, None]
) / scale[:, None]
target_standardized = (
    target_rate[eligible] - representation_mean[eligible, None]
) / scale[:, None]

if not np.all(np.isfinite(representation_standardized)):
    raise RuntimeError("Non-finite representation value after scaling")
if not np.all(np.isfinite(target_standardized)):
    raise RuntimeError("Non-finite target value after scaling")

print("PREPROCESSING")
print(f"  representation rate: {representation_rate.shape}")
print(f"  target rate:         {target_rate.shape}")
print(f"  eligible neurons:    {eligible_rows.size:,} / {n_neurons:,}")
print(f"  bin width:           {DT_SECONDS:.6f} s")
print(
    "  IMPORTANT: scaling is learned only from acquisition 1 and is fixed "
    "across every candidate window."
)

# A fixed POD basis prevents the huge neuron x delay dictionary from forcing a
# separate, incomparable spatial basis in every window. It is learned only from
# acquisition 1. DMD predictions are scored over all eligible neurons, so
# variance discarded by POD still counts as prediction error.
#
# Before fitting DMD, we must check whether the fixed observation space can in
# principle reach the declared practical target. For each q below, the
# ``projection oracle'' is allowed to see the development future and projects
# that true future snapshot into the q-dimensional POD space. It is therefore
# an instantaneous representation ceiling, not a forecast and not a model
# score. Outer samples are never used. Nested prefixes of one q=256 SVD make
# this a fair capacity comparison.
pod_start = time.perf_counter()
maximum_pod_dimension = max(POD_CAPACITY_DIMENSIONS)
if maximum_pod_dimension >= min(representation_standardized.shape):
    raise RuntimeError("Requested POD capacity exceeds the training matrix rank")
pod_basis_max, pod_singular_values_max, _ = svds(
    representation_standardized,
    k=maximum_pod_dimension,
    which="LM",
    solver="propack",
    rng=np.random.default_rng(SEED),
)
descending = np.argsort(pod_singular_values_max)[::-1]
pod_basis_max = pod_basis_max[:, descending]
pod_singular_values_max = pod_singular_values_max[descending]
pod_dimension = POD_DIMENSION
pod_basis = pod_basis_max[:, :pod_dimension]
pod_singular_values = pod_singular_values_max[:pod_dimension]
pod_retained_energy = float(
    np.sum(pod_singular_values**2)
    / np.sum(representation_standardized**2)
)
pod_orthogonality_error = float(
    np.linalg.norm(pod_basis.T @ pod_basis - np.eye(pod_dimension), ord=2)
)

print("FIXED POD OBSERVATION SPACE")
print(f"  dimension:          {pod_dimension}")
print(f"  retained energy:    {pod_retained_energy:.3%}")
print(f"  orthogonality error:{pod_orthogonality_error:.3e}")
print(f"  computation time:   {time.perf_counter() - pod_start:.2f} s")
print(
    "LIMITATION — neuronwise RMS scaling changes the physical inner product; "
    "raw-centered rate must later be checked as a sensitivity analysis."
)

# Project the complete tuning bout only after the acquisition-1 basis is
# frozen. Keeping this uncentered projected series also lets every later local
# window estimate its own past-only mean without repeatedly multiplying the
# all-neuron matrix.
target_latent_max = pod_basis_max.T @ target_standardized
target_latent = target_latent_max[:pod_dimension]

pod_oracle_rows: list[dict[str, float | int]] = []
representation_energy = float(np.sum(representation_standardized**2))
for capacity_dimension in POD_CAPACITY_DIMENSIONS:
    retained_energy = float(
        np.sum(pod_singular_values_max[:capacity_dimension] ** 2)
        / representation_energy
    )
    latent_values = target_latent_max[:capacity_dimension]
    for window_seconds, window_bins in WINDOW_BINS.items():
        for origin in DEVELOPMENT_ORIGINS:
            local_mean = np.mean(
                target_standardized[:, origin - window_bins : origin],
                axis=1,
                keepdims=True,
            )
            local_latent_mean = np.mean(
                latent_values[:, origin - window_bins : origin],
                axis=1,
                keepdims=True,
            )
            common_normalized_errors = []
            local_normalized_errors = []
            for horizon in SCORED_HORIZONS:
                truth = target_standardized[:, origin + horizon - 1]
                truth_centered = (
                    truth - local_mean[:, 0]
                )
                latent_truth_centered = (
                    latent_values[:, origin + horizon - 1]
                    - local_latent_mean[:, 0]
                )
                common_mean_sse = float(np.sum(truth**2))
                local_mean_sse = float(np.sum(truth_centered**2))
                oracle_sse = max(
                    0.0,
                    local_mean_sse - float(np.sum(latent_truth_centered**2)),
                )
                common_normalized_errors.append(oracle_sse / common_mean_sse)
                local_normalized_errors.append(oracle_sse / local_mean_sse)
            oracle_common_loss = float(np.mean(common_normalized_errors))
            oracle_local_loss = float(np.mean(local_normalized_errors))
            pod_oracle_rows.append(
                {
                    "pod_dimension": capacity_dimension,
                    "pod_retained_energy": retained_energy,
                    "window_nominal_seconds": window_seconds,
                    "window_bins": window_bins,
                    "origin_bin": origin,
                    "oracle_primary_loss": oracle_common_loss,
                    "oracle_common_mean_r2": 1 - oracle_common_loss,
                    "oracle_local_mean_r2": 1 - oracle_local_loss,
                }
            )

pod_oracle_table = pd.DataFrame(pod_oracle_rows)
pod_oracle_summary = (
    pod_oracle_table.groupby(
        [
            "pod_dimension",
            "pod_retained_energy",
            "window_nominal_seconds",
            "window_bins",
        ],
        as_index=False,
    )
    .agg(
        oracle_common_mean_development_r2=("oracle_common_mean_r2", "mean"),
        oracle_development_r2=("oracle_local_mean_r2", "mean"),
        minimum_origin_oracle_r2=("oracle_local_mean_r2", "min"),
    )
)
pod_oracle_table.to_csv(OUTPUT_DIR / "pod_oracle_origin_scores.csv", index=False)
pod_oracle_summary.to_csv(OUTPUT_DIR / "pod_oracle_capacity.csv", index=False)

capacity_maxima = pod_oracle_summary.groupby("pod_dimension")[
    "oracle_development_r2"
].max()
attainable_dimensions = capacity_maxima.index[
    capacity_maxima >= PRACTICAL_R2_TARGET
].to_list()
if pod_dimension not in attainable_dimensions:
    raise RuntimeError(
        "The selected POD dimension still makes the practical R² target unreachable"
    )
print("POD PROJECTION-CAPACITY AUDIT (development truth; not a forecast)")
for capacity_dimension, maximum_oracle_r2 in capacity_maxima.items():
    print(
        f"  q={int(capacity_dimension):3d}: maximum mean oracle R² across W "
        f"= {maximum_oracle_r2:.4f}"
    )
print(
    f"  selected q={pod_dimension}; smallest tested q reaching "
    f"R²={PRACTICAL_R2_TARGET:.2f} is q={min(attainable_dimensions)}"
)

fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
axes[0, 0].plot(
    np.arange(1, pod_dimension + 1),
    pod_singular_values,
    marker=".",
    linewidth=1,
)
axes[0, 0].set(
    xlabel="fixed POD component",
    ylabel="singular value",
    yscale="log",
    title="Acquisition-1 NREM singular spectrum",
)
axes[0, 1].plot(
    np.arange(1, pod_dimension + 1),
    np.cumsum(pod_singular_values**2)
    / np.sum(representation_standardized**2),
)
axes[0, 1].axhline(pod_retained_energy, color="#D55E00", linestyle="--")
axes[0, 1].set(
    xlabel="number of components",
    ylabel="fraction of all-neuron energy",
    title="POD energy retained before DMD",
)
axes[1, 0].hist(
    representation_rms[representation_rms > 0],
    bins=80,
    color="#009E73",
    log=True,
)
axes[1, 0].set(
    xlabel="acquisition-1 NREM RMS event-rate deviation",
    ylabel="neurons (log count)",
    title="Fixed neuron scaling",
)
axes[1, 1].plot(
    np.arange(OUTER_START_BIN) * DT_SECONDS,
    np.mean(target_rate[:, :OUTER_START_BIN], axis=0),
    color="#4C78A8",
    linewidth=0.9,
)
axes[1, 1].axvspan(
    OUTER_START_BIN * DT_SECONDS,
    target_complete_bins * DT_SECONDS,
    color="#D55E00",
    alpha=0.2,
)
axes[1, 1].set(
    xlim=(0, target_complete_bins * DT_SECONDS),
    xlabel="time within target NREM bout (s)",
    ylabel="mean event mass / s",
    title="Development activity; score-locked outer activity is hidden",
)
fig.suptitle("Step 3 — Fixed preprocessing and spatial coordinate system")
save_and_show(fig, "01_preprocessing_and_fixed_pod.png")

fig, axis = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
for capacity_dimension, group in pod_oracle_summary.groupby("pod_dimension"):
    axis.plot(
        group["window_nominal_seconds"],
        group["oracle_development_r2"],
        marker="o",
        label=f"q={int(capacity_dimension)}",
    )
axis.axhline(
    PRACTICAL_R2_TARGET,
    color="#D55E00",
    linestyle="--",
    label=f"practical target R²={PRACTICAL_R2_TARGET:.2f}",
)
axis.set(
    xlabel="local-mean window W (s)",
    ylabel="instantaneous projection-oracle R²",
    title="Future-informed capacity ceiling (not DMD performance)",
)
axis.legend(frameon=False)
fig.suptitle("Step 3 — The observation space must not predetermine failure")
save_and_show(fig, "01b_pod_projection_capacity.png")


# %% Step 4 — Define and unit-test autonomous forecasting
#
# For n latent observables and delay depth d, a Hankel column is
#
#     z_j = [x_j; x_(j+1); ...; x_(j+d-1)].
#
# PyDMD's HankelDMD eigenvalues are discrete-time multipliers. Starting from
# the final observed delay vector, the h-step autonomous embedded prediction is
#
#     z_hat(h) = Phi diag(lambda**h) pinv(Phi) z_last.
#
# Only the final n-row block is the genuinely new physical snapshot. No true
# future sample is inserted after initialization. This is stricter than the old
# teacher-forced one-step calibration calculation.


def relative_imaginary_leakage(values: np.ndarray) -> float:
    """Return imaginary-to-real Frobenius norm before coercing a forecast."""
    values = np.asarray(values)
    real_norm = float(np.linalg.norm(values.real))
    imaginary_norm = float(np.linalg.norm(values.imag))
    return imaginary_norm / max(real_norm, np.finfo(float).eps)


def forecast_hankeldmd(
    model: HankelDMD,
    context: np.ndarray,
    delay: int,
    steps: int,
) -> tuple[np.ndarray, float]:
    """Autonomously forecast the last physical block from one endpoint."""
    context = np.asarray(context)
    if context.ndim != 2 or context.shape[1] < delay or steps < 1:
        raise ValueError("context, delay, and steps are incompatible")
    n_observables = context.shape[0]
    modes = np.asarray(model.modes)
    eigenvalues = np.asarray(model.eigs)
    if modes.shape[0] != n_observables * delay:
        raise ValueError("PyDMD mode dimension disagrees with delay embedding")
    if not np.all(np.isfinite(modes)) or not np.all(np.isfinite(eigenvalues)):
        raise FloatingPointError("Non-finite PyDMD mode or eigenvalue")

    # Equivalent to the last column of pseudo_hankel_matrix(context, delay),
    # written explicitly to make the block order visible to students.
    last_delay_vector = context[:, -delay:].reshape(-1, order="F")
    endpoint_amplitudes = np.linalg.lstsq(
        modes, last_delay_vector, rcond=None
    )[0]
    horizons = np.arange(1, steps + 1)
    embedded_forecast = modes @ (
        endpoint_amplitudes[:, None]
        * eigenvalues[:, None] ** horizons[None, :]
    )
    forecast = embedded_forecast.reshape(
        delay, n_observables, steps
    )[-1]
    leakage = relative_imaginary_leakage(forecast)
    return np.asarray(forecast.real, dtype=np.float64), leakage


def forecast_bopdmd_from_endpoint(
    model: object,
    context: np.ndarray,
    delay: int,
    steps: int,
    dt_seconds: float,
) -> tuple[np.ndarray, float]:
    """Forecast BOPDMD modes after re-anchoring amplitudes at the endpoint.

    BOPDMD eigenvalues are continuous-time rates omega, so evolution is
    exp(omega * h * dt), not lambda**h. Re-anchoring makes this comparison
    condition on the same last observed delay vector as HankelDMD.
    """
    context = np.asarray(context)
    n_observables = context.shape[0]
    modes = np.asarray(model.modes)
    rates = np.asarray(model.eigs)
    if modes.shape[0] != n_observables * delay:
        raise ValueError("BOPDMD mode dimension disagrees with delay embedding")
    last_delay_vector = context[:, -delay:].reshape(-1, order="F")
    endpoint_amplitudes = np.linalg.lstsq(
        modes, last_delay_vector, rcond=None
    )[0]
    future_time = np.arange(1, steps + 1) * dt_seconds
    embedded_forecast = modes @ (
        endpoint_amplitudes[:, None]
        * np.exp(rates[:, None] * future_time[None, :])
    )
    forecast = embedded_forecast.reshape(
        delay, n_observables, steps
    )[-1]
    leakage = relative_imaginary_leakage(forecast)
    return np.asarray(forecast.real, dtype=np.float64), leakage


# Noise-free damped rotation: delay-DMD should forecast it to round-off.
synthetic_radius = 0.97
synthetic_angle = 0.23
synthetic_operator = synthetic_radius * np.array(
    [
        [np.cos(synthetic_angle), -np.sin(synthetic_angle)],
        [np.sin(synthetic_angle), np.cos(synthetic_angle)],
    ]
)
synthetic = np.empty((2, 80), dtype=float)
synthetic[:, 0] = [1.0, -0.25]
for synthetic_index in range(1, synthetic.shape[1]):
    synthetic[:, synthetic_index] = synthetic_operator @ synthetic[:, synthetic_index - 1]

synthetic_fit_stop = 55
synthetic_delay = 3
synthetic_model = HankelDMD(
    svd_rank=2,
    d=synthetic_delay,
    exact=True,
    opt=False,
    reconstruction_method="mean",
)
with warnings.catch_warnings(record=True) as synthetic_warnings:
    warnings.simplefilter("always")
    synthetic_model.fit(synthetic[:, :synthetic_fit_stop])
synthetic_prediction, synthetic_leakage = forecast_hankeldmd(
    synthetic_model,
    synthetic[:, :synthetic_fit_stop],
    synthetic_delay,
    synthetic.shape[1] - synthetic_fit_stop,
)
synthetic_max_error = float(
    np.max(np.abs(synthetic_prediction - synthetic[:, synthetic_fit_stop:]))
)
if synthetic_max_error > 1e-10:
    raise RuntimeError("The autonomous HankelDMD forecast unit check failed")

print("SUCCESS — autonomous forecast unit check")
print(f"  maximum 25-step error: {synthetic_max_error:.3e}")
print(f"  imaginary leakage:     {synthetic_leakage:.3e}")
print(f"  captured PyDMD warnings: {len(synthetic_warnings)}")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
synthetic_time = np.arange(synthetic.shape[1])
for row, color in zip(range(2), ("#0072B2", "#D55E00"), strict=True):
    axes[row].plot(
        synthetic_time,
        synthetic[row],
        color=color,
        label="truth",
    )
    axes[row].plot(
        synthetic_time[synthetic_fit_stop:],
        synthetic_prediction[row],
        color="black",
        linestyle="--",
        label="autonomous forecast",
    )
    axes[row].axvline(synthetic_fit_stop - 1, color="0.5", linestyle=":")
    axes[row].set(
        xlabel="sample",
        ylabel=f"state {row + 1}",
        title=f"Synthetic coordinate {row + 1}",
    )
    axes[row].legend(frameon=False)
fig.suptitle("Step 4 — Forecast implementation recovers a known linear system")
save_and_show(fig, "02_autonomous_forecast_unit_check.png")


# %% Step 5 — Build the explicit W,d,r grid and overparameterization guard
#
# The delayed ambient row count is q*d, but the fitted operator rank is r.
# The useful regression sample count is W-d transitions. We display their
# ratio instead of describing the method vaguely as overparameterized.

grid_rows: list[dict[str, float | int | bool | str]] = []
for window_seconds, window_bins in WINDOW_BINS.items():
    for delay in DELAY_CANDIDATES:
        transitions = window_bins - delay
        for rank in RANK_CANDIDATES:
            ratio = transitions / rank
            allowed = (
                transitions > rank
                and ratio >= MIN_TRANSITIONS_PER_RANK
                and rank <= pod_dimension * delay
            )
            reason = "allowed" if allowed else "insufficient transitions per rank"
            grid_rows.append(
                {
                    "window_nominal_seconds": window_seconds,
                    "window_bins": window_bins,
                    "window_actual_seconds": window_bins * DT_SECONDS,
                    "delay": delay,
                    "history_seconds": (delay - 1) * DT_SECONDS,
                    "rank": rank,
                    "ambient_hankel_rows": pod_dimension * delay,
                    "snapshot_transitions": transitions,
                    "transitions_per_rank": ratio,
                    "selection_allowed": allowed,
                    "guard_reason": reason,
                    "rough_frequency_resolution_hz": 1
                    / (transitions * DT_SECONDS),
                    "three_cycle_frequency_hz": 3 / (transitions * DT_SECONDS),
                }
            )

grid_table = pd.DataFrame(grid_rows)
grid_table.to_csv(OUTPUT_DIR / "candidate_grid.csv", index=False)
allowed_grid = grid_table.loc[grid_table["selection_allowed"]].copy()
if allowed_grid.empty:
    raise RuntimeError("Every W,d,r combination failed the rank guard")

print("GRID SUPPORT")
print(f"  all combinations shown: {len(grid_table)}")
print(f"  combinations eligible: {len(allowed_grid)}")
support_summary = (
    allowed_grid.groupby("window_nominal_seconds", as_index=False)
    .agg(
        eligible_configurations=("rank", "size"),
        maximum_allowed_rank=("rank", "max"),
        minimum_three_cycle_frequency_hz=("three_cycle_frequency_hz", "first"),
    )
)
print(support_summary.to_string(index=False, float_format=lambda value: f"{value:.4g}"))
print(
    "HEURISTIC — a mode with fewer than three observed cycles is not called "
    "a resolved oscillation in this audit."
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
for delay in DELAY_CANDIDATES:
    subset = allowed_grid.loc[allowed_grid["delay"] == delay]
    maximum_rank = subset.groupby("window_nominal_seconds")["rank"].max()
    axes[0].plot(
        maximum_rank.index,
        maximum_rank.values,
        marker="o",
        label=f"d={delay}",
    )
axes[0].set(
    xlabel="fit window W (s)",
    ylabel="maximum eligible r",
    title="Rank support after the 8-transitions/rank guard",
)
axes[0].legend(frameon=False, ncol=2)
axes[1].plot(
    support_summary["window_nominal_seconds"],
    support_summary["minimum_three_cycle_frequency_hz"],
    marker="o",
    color="#D55E00",
)
axes[1].set(
    xlabel="fit window W (s)",
    ylabel="3 / actual W (Hz)",
    yscale="log",
    title="Low-frequency cost of a short local window",
)
fig.suptitle("Step 5 — Shorter windows buy locality but lose regression and frequency support")
save_and_show(fig, "03_grid_support_and_frequency_tradeoff.png")


# %% Step 6 — Define scores and baselines before running the sweep
#
# For each origin, a model sees exactly W samples and forecasts the next eight
# autonomously. All-neuron errors are evaluated after mapping out of POD.
# The ranking denominator must not change with W. At horizon h,
#
#     R²_fixed(h) = 1 - SSE_DMD(h) / SSE_acquisition-1-mean(h).
#
# Because the standardized acquisition-1 mean is zero, the fixed denominator
# is identical for every W,d,r at a given origin and horizon. This makes window
# ranking fair. We separately report
#
#     R²_local(h) = 1 - SSE_DMD(h) / SSE_past-window-mean(h),
#
# which asks whether dynamics improve on the local level but is not used to
# rank W because its denominator changes with W. These are prediction
# coefficients, not claims that an exponential trajectory reconstructs all
# activity. Persistence and neuronwise AR(1) use the same model SSE.


@dataclass(frozen=True)
class OriginForecast:
    primary_loss: float
    common_mean_r2: float
    local_mean_r2: float
    skill_vs_persistence: float
    skill_vs_ar1: float
    h4_common_mean_r2: float
    h4_local_mean_r2: float
    imaginary_leakage: float
    maximum_eigenvalue_magnitude: float
    warning_count: int
    unexpected_warning_count: int


@dataclass(frozen=True)
class PreparedOrigin:
    """Past-only latent context and full-neuron baseline errors for one origin."""

    latent_context: np.ndarray
    truth_centered_latent: np.ndarray
    common_mean_sse: np.ndarray
    local_mean_sse: np.ndarray
    persistence_sse: np.ndarray
    ar1_sse: np.ndarray


def neuronwise_ar1_forecast(
    context: np.ndarray,
    local_mean: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Stable neuronwise AR(1) baseline fitted only inside the local window."""
    centered = context - local_mean
    x = centered[:, :-1]
    y = centered[:, 1:]
    denominator = np.sum(x**2, axis=1) + 1e-8
    coefficient = np.sum(x * y, axis=1) / denominator
    coefficient = np.clip(coefficient, -1.0, 1.0)
    state_value = centered[:, -1].copy()
    predictions = np.empty((context.shape[0], steps), dtype=np.float64)
    for step in range(steps):
        state_value = coefficient * state_value
        predictions[:, step] = state_value + local_mean[:, 0]
    return predictions


def prepare_origin(
    target_values: np.ndarray,
    projected_values: np.ndarray,
    window_bins: int,
    origin: int,
    fixed_reference_mean: np.ndarray,
) -> PreparedOrigin:
    """Prepare one local window without looking beyond its forecast block."""
    context = target_values[:, origin - window_bins : origin]
    truth = target_values[:, origin : origin + FORECAST_HORIZON_BINS]
    local_mean = np.mean(context, axis=1, keepdims=True)
    projected_context = projected_values[:, origin - window_bins : origin]
    projected_mean = np.mean(projected_context, axis=1, keepdims=True)
    latent_context = projected_context - projected_mean
    truth_centered_latent = (
        projected_values[:, origin : origin + FORECAST_HORIZON_BINS]
        - projected_mean
    )
    persistence = np.repeat(context[:, -1:], FORECAST_HORIZON_BINS, axis=1)
    ar1 = neuronwise_ar1_forecast(context, local_mean, FORECAST_HORIZON_BINS)
    fixed_reference_mean = np.asarray(fixed_reference_mean, dtype=np.float64)
    if fixed_reference_mean.shape != (target_values.shape[0],):
        raise ValueError("fixed_reference_mean must have one value per neuron")
    common_mean_sse = np.sum(
        (truth - fixed_reference_mean[:, None]) ** 2, axis=0
    )
    local_mean_sse = np.sum((truth - local_mean) ** 2, axis=0)
    persistence_sse = np.sum((truth - persistence) ** 2, axis=0)
    ar1_sse = np.sum((truth - ar1) ** 2, axis=0)
    if min(
        float(np.min(common_mean_sse)),
        float(np.min(local_mean_sse)),
        float(np.min(persistence_sse)),
        float(np.min(ar1_sse)),
    ) <= 0:
        raise RuntimeError("A prepared forecast baseline has zero SSE")
    return PreparedOrigin(
        latent_context=latent_context,
        truth_centered_latent=truth_centered_latent,
        common_mean_sse=np.asarray(common_mean_sse, dtype=np.float64),
        local_mean_sse=np.asarray(local_mean_sse, dtype=np.float64),
        persistence_sse=np.asarray(persistence_sse, dtype=np.float64),
        ar1_sse=np.asarray(ar1_sse, dtype=np.float64),
    )


def score_latent_forecast(
    prepared: PreparedOrigin,
    latent_prediction: np.ndarray,
    imaginary_leakage: float,
    maximum_eigenvalue_magnitude: float,
    warning_count: int,
    unexpected_warning_count: int,
) -> tuple[OriginForecast, list[dict[str, float | int]]]:
    """Score a POD-space prediction exactly in the full-neuron norm.

    For orthonormal B, ||y - B z||² = ||y||² + ||z||² - 2 z'B'y.
    This avoids constructing a 6,570-neuron prediction for every grid fit;
    discarded POD energy remains in the error through ||y||².
    """
    horizon_rows: list[dict[str, float | int]] = []
    common_normalized_errors: list[float] = []
    local_normalized_errors: list[float] = []
    persistence_skills: list[float] = []
    ar1_skills: list[float] = []
    common_r2_by_horizon: dict[int, float] = {}
    local_r2_by_horizon: dict[int, float] = {}
    for horizon in SCORED_HORIZONS:
        column = horizon - 1
        prediction = latent_prediction[:, column]
        truth_projection = prepared.truth_centered_latent[:, column]
        model_sse = max(
            0.0,
            float(
                prepared.local_mean_sse[column]
                + np.sum(prediction**2)
                - 2 * np.dot(prediction, truth_projection)
            ),
        )
        common_mean_sse = float(prepared.common_mean_sse[column])
        local_mean_sse = float(prepared.local_mean_sse[column])
        persistence_sse = float(prepared.persistence_sse[column])
        ar1_sse = float(prepared.ar1_sse[column])
        common_normalized_error = model_sse / common_mean_sse
        local_normalized_error = model_sse / local_mean_sse
        common_mean_r2 = 1 - common_normalized_error
        local_mean_r2 = 1 - local_normalized_error
        persistence_skill = 1 - model_sse / persistence_sse
        ar1_skill = 1 - model_sse / ar1_sse
        common_normalized_errors.append(common_normalized_error)
        local_normalized_errors.append(local_normalized_error)
        persistence_skills.append(persistence_skill)
        ar1_skills.append(ar1_skill)
        common_r2_by_horizon[horizon] = common_mean_r2
        local_r2_by_horizon[horizon] = local_mean_r2
        horizon_rows.append(
            {
                "horizon_bins": horizon,
                "horizon_seconds": horizon * DT_SECONDS,
                "model_sse": model_sse,
                "common_mean_sse": common_mean_sse,
                "local_mean_sse": local_mean_sse,
                "persistence_sse": persistence_sse,
                "ar1_sse": ar1_sse,
                "normalized_sse_vs_common_mean": common_normalized_error,
                "normalized_sse_vs_local_mean": local_normalized_error,
                "r2_vs_common_mean": common_mean_r2,
                "r2_vs_local_mean": local_mean_r2,
                "skill_vs_persistence": persistence_skill,
                "skill_vs_ar1": ar1_skill,
            }
        )
    primary_loss = float(np.mean(common_normalized_errors))
    return (
        OriginForecast(
            primary_loss=primary_loss,
            common_mean_r2=1 - primary_loss,
            local_mean_r2=1 - float(np.mean(local_normalized_errors)),
            skill_vs_persistence=float(np.mean(persistence_skills)),
            skill_vs_ar1=float(np.mean(ar1_skills)),
            h4_common_mean_r2=common_r2_by_horizon[4],
            h4_local_mean_r2=local_r2_by_horizon[4],
            imaginary_leakage=imaginary_leakage,
            maximum_eigenvalue_magnitude=maximum_eigenvalue_magnitude,
            warning_count=warning_count,
            unexpected_warning_count=unexpected_warning_count,
        ),
        horizon_rows,
    )


def unique_warning_records(
    caught: list[warnings.WarningMessage],
) -> list[dict[str, bool | str]]:
    """Deduplicate PyDMD wrapper warnings and classify expected conditioning."""
    records = []
    seen: set[tuple[str, str]] = set()
    for item in caught:
        category = item.category.__name__
        message = str(item.message)
        key = (category, message)
        if key in seen:
            continue
        seen.add(key)
        expected_conditioning = "condition number" in message.lower()
        records.append(
            {
                "warning_category": category,
                "warning_message": message,
                "expected_rank_truncation_warning": expected_conditioning,
                "unexpected_warning": not expected_conditioning,
            }
        )
    return records


standardized_origin_cache: dict[tuple[int, int], PreparedOrigin] = {}


def standardized_origin(window_bins: int, origin: int) -> PreparedOrigin:
    """Return a cached standardized-data origin preparation."""
    key = (window_bins, origin)
    if key not in standardized_origin_cache:
        standardized_origin_cache[key] = prepare_origin(
            target_standardized,
            target_latent,
            window_bins,
            origin,
            np.zeros(target_standardized.shape[0], dtype=np.float64),
        )
    return standardized_origin_cache[key]


# Numerical identity check for the fast all-neuron score. This verifies that
# discarded POD energy remains in the loss even though the sweep never builds
# a 6,570-neuron prediction matrix for every fit.
score_unit_window = WINDOW_BINS[20]
score_unit_origin = DEVELOPMENT_ORIGINS[0]
score_unit_prepared = standardized_origin(score_unit_window, score_unit_origin)
score_unit_latent_prediction = np.random.default_rng(SEED + 3).standard_normal(
    (pod_dimension, FORECAST_HORIZON_BINS)
) * 0.05
score_unit_context = target_standardized[
    :, score_unit_origin - score_unit_window : score_unit_origin
]
score_unit_truth = target_standardized[
    :, score_unit_origin : score_unit_origin + FORECAST_HORIZON_BINS
]
score_unit_mean = np.mean(score_unit_context, axis=1, keepdims=True)
score_unit_direct_prediction = (
    pod_basis @ score_unit_latent_prediction + score_unit_mean
)
score_unit_direct_sse = np.sum(
    (score_unit_truth - score_unit_direct_prediction) ** 2,
    axis=0,
)
score_unit_identity_sse = score_unit_prepared.local_mean_sse + np.sum(
    score_unit_latent_prediction**2, axis=0
) - 2 * np.sum(
    score_unit_latent_prediction
    * score_unit_prepared.truth_centered_latent,
    axis=0,
)
score_unit_relative_error = float(
    np.max(
        np.abs(score_unit_direct_sse - score_unit_identity_sse)
        / np.maximum(score_unit_direct_sse, np.finfo(float).eps)
    )
)
if score_unit_relative_error > 1e-10:
    raise RuntimeError("The fast all-neuron POD score identity failed")
print(
    "SUCCESS — exact all-neuron POD scoring identity; maximum relative error "
    f"{score_unit_relative_error:.3e}"
)

# The common-reference denominator must be numerically identical across W for
# every origin/horizon. The local-mean denominator is expected to vary and is
# therefore retained only as an incremental-effect diagnostic.
common_denominators = np.stack(
    [
        np.stack(
            [
                standardized_origin(window_bins, origin).common_mean_sse
                for origin in DEVELOPMENT_ORIGINS
            ]
        )
        for window_bins in WINDOW_BINS.values()
    ]
)
local_denominators = np.stack(
    [
        np.stack(
            [
                standardized_origin(window_bins, origin).local_mean_sse
                for origin in DEVELOPMENT_ORIGINS
            ]
        )
        for window_bins in WINDOW_BINS.values()
    ]
)
common_denominator_relative_range = float(
    np.max(np.ptp(common_denominators, axis=0))
    / np.mean(common_denominators)
)
local_denominator_relative_range = float(
    np.max(np.ptp(local_denominators, axis=0))
    / np.mean(local_denominators)
)
if common_denominator_relative_range > 1e-14:
    raise RuntimeError("The supposedly common selection denominator varies with W")
print(
    "SUCCESS — common-reference denominator is W-invariant; maximum relative "
    f"range={common_denominator_relative_range:.3e}"
)
print(
    "DIAGNOSTIC — local-mean denominator varies across W; maximum relative "
    f"range={local_denominator_relative_range:.3f}"
)


def fit_hankel_at_origin(
    window_bins: int,
    delay: int,
    rank: int,
    origin: int,
) -> tuple[
    OriginForecast,
    list[dict[str, float | int]],
    HankelDMD,
    list[dict[str, bool | str]],
]:
    """Fit one full local window and forecast its following target block."""
    prepared = standardized_origin(window_bins, origin)
    latent_context = prepared.latent_context

    model = HankelDMD(
        svd_rank=rank,
        tlsq_rank=0,
        exact=True,
        opt=False,
        rescale_mode=None,
        forward_backward=False,
        d=delay,
        sorted_eigs=False,
        reconstruction_method="mean",
        tikhonov_regularization=None,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(latent_context)
    warning_records = unique_warning_records(caught)

    latent_prediction, imaginary_leakage = forecast_hankeldmd(
        model,
        latent_context,
        delay,
        FORECAST_HORIZON_BINS,
    )
    forecast_score, horizon_rows = score_latent_forecast(
        prepared,
        latent_prediction,
        imaginary_leakage,
        float(np.max(np.abs(model.eigs))),
        len(warning_records),
        sum(bool(record["unexpected_warning"]) for record in warning_records),
    )
    return forecast_score, horizon_rows, model, warning_records


print("PRIMARY DEVELOPMENT LOSS")
print(
    "  equal-weight mean over autonomous horizons "
    f"{SCORED_HORIZONS} of SSE_DMD / SSE_fixed-acquisition-1-mean"
)
print("  this common denominator is identical across W at each target")
print("  local-window-mean incremental R² is reported separately")
print("  each origin receives equal weight; neuron x time entries are not replicates")


# %% Step 7 — Run the development-only PyDMD HankelDMD sweep
#
# Every allowed configuration is refitted at all eight common origins. The
# model family is held fixed in this first screen: exact HankelDMD, no TLSQ,
# no forward-backward correction, and no Tikhonov regularization. Adding every
# noise correction at once would make it impossible to learn which change
# helped. PyDMD's documented noise-aware optimized DMD is assessed only after
# this fast screen, on a locked shortlist.

development_rows: list[dict[str, float | int | bool | str]] = []
development_horizon_rows: list[dict[str, float | int | str]] = []
failure_rows: list[dict[str, float | int | str]] = []
warning_rows: list[dict[str, float | int | bool | str]] = []

sweep_start = time.perf_counter()
progress = tqdm(
    total=len(allowed_grid) * len(DEVELOPMENT_ORIGINS),
    desc="development HankelDMD fits",
    disable=not sys.stderr.isatty(),
)
for configuration in allowed_grid.itertuples(index=False):
    configuration_key = {
        "window_nominal_seconds": int(configuration.window_nominal_seconds),
        "window_bins": int(configuration.window_bins),
        "window_actual_seconds": float(configuration.window_actual_seconds),
        "delay": int(configuration.delay),
        "history_seconds": float(configuration.history_seconds),
        "rank": int(configuration.rank),
        "snapshot_transitions": int(configuration.snapshot_transitions),
        "transitions_per_rank": float(configuration.transitions_per_rank),
    }
    for origin in DEVELOPMENT_ORIGINS:
        try:
            origin_score, horizon_scores, _, origin_warnings = fit_hankel_at_origin(
                int(configuration.window_bins),
                int(configuration.delay),
                int(configuration.rank),
                origin,
            )
            development_rows.append(
                {
                    **configuration_key,
                    "origin_bin": origin,
                    "origin_seconds": origin * DT_SECONDS,
                    **asdict(origin_score),
                }
            )
            for horizon_score in horizon_scores:
                development_horizon_rows.append(
                    {
                        **configuration_key,
                        "origin_bin": origin,
                        "origin_seconds": origin * DT_SECONDS,
                        **horizon_score,
                    }
                )
            for warning_record in origin_warnings:
                warning_rows.append(
                    {
                        **configuration_key,
                        "origin_bin": origin,
                        **warning_record,
                    }
                )
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            failure_rows.append(
                {
                    **configuration_key,
                    "origin_bin": origin,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        finally:
            progress.update(1)
progress.close()

development_table = pd.DataFrame(development_rows)
development_horizon_table = pd.DataFrame(development_horizon_rows)
failure_columns = [
    "window_nominal_seconds",
    "window_bins",
    "window_actual_seconds",
    "delay",
    "history_seconds",
    "rank",
    "snapshot_transitions",
    "transitions_per_rank",
    "origin_bin",
    "error_type",
    "error",
]
warning_columns = [
    "window_nominal_seconds",
    "window_bins",
    "window_actual_seconds",
    "delay",
    "history_seconds",
    "rank",
    "snapshot_transitions",
    "transitions_per_rank",
    "origin_bin",
    "warning_category",
    "warning_message",
    "expected_rank_truncation_warning",
    "unexpected_warning",
]
failure_table = pd.DataFrame(failure_rows, columns=failure_columns)
warning_table = pd.DataFrame(warning_rows, columns=warning_columns)
development_table.to_csv(OUTPUT_DIR / "development_origin_scores.csv", index=False)
development_horizon_table.to_csv(
    OUTPUT_DIR / "development_horizon_scores.csv", index=False
)
failure_table.to_csv(OUTPUT_DIR / "development_fit_failures.csv", index=False)
warning_table.to_csv(OUTPUT_DIR / "development_fit_warnings.csv", index=False)

if development_table.empty:
    raise RuntimeError("Every development model fit failed")

configuration_columns = [
    "window_nominal_seconds",
    "window_bins",
    "window_actual_seconds",
    "delay",
    "history_seconds",
    "rank",
    "snapshot_transitions",
    "transitions_per_rank",
]

development_summary = (
    development_table.groupby(configuration_columns, as_index=False)
    .agg(
        successful_origins=("origin_bin", "nunique"),
        primary_loss_mean=("primary_loss", "mean"),
        primary_loss_sd=("primary_loss", "std"),
        development_r2=("common_mean_r2", "mean"),
        local_mean_development_r2=("local_mean_r2", "mean"),
        positive_r2_fraction=(
            "common_mean_r2", lambda values: float(np.mean(values > 0))
        ),
        positive_local_mean_r2_fraction=(
            "local_mean_r2", lambda values: float(np.mean(values > 0))
        ),
        skill_vs_persistence=("skill_vs_persistence", "mean"),
        skill_vs_ar1=("skill_vs_ar1", "mean"),
        h4_r2=("h4_common_mean_r2", "mean"),
        h4_local_mean_r2=("h4_local_mean_r2", "mean"),
        imaginary_leakage_max=("imaginary_leakage", "max"),
        maximum_eigenvalue_magnitude=("maximum_eigenvalue_magnitude", "max"),
        warning_count=("warning_count", "sum"),
        unexpected_warning_count=("unexpected_warning_count", "sum"),
    )
)

# Add horizon-specific gates. These prevent an average score from hiding a
# one-step or longest-horizon failure.
horizon_summary = (
    development_horizon_table.groupby(
        configuration_columns + ["horizon_bins"], as_index=False
    )
    .agg(
        horizon_r2=("r2_vs_common_mean", "mean"),
        horizon_local_mean_r2=("r2_vs_local_mean", "mean"),
        horizon_positive_r2_fraction=(
            "r2_vs_common_mean", lambda values: float(np.mean(values > 0))
        ),
        horizon_positive_local_mean_r2_fraction=(
            "r2_vs_local_mean", lambda values: float(np.mean(values > 0))
        ),
        horizon_skill_persistence=("skill_vs_persistence", "mean"),
        horizon_skill_ar1=("skill_vs_ar1", "mean"),
    )
)
for horizon in (1, FORECAST_HORIZON_BINS):
    horizon_values = horizon_summary.loc[
        horizon_summary["horizon_bins"] == horizon,
        configuration_columns
        + [
            "horizon_r2",
            "horizon_local_mean_r2",
            "horizon_positive_r2_fraction",
            "horizon_positive_local_mean_r2_fraction",
            "horizon_skill_persistence",
            "horizon_skill_ar1",
        ],
    ].rename(
        columns={
            "horizon_r2": f"h{horizon}_r2",
            "horizon_local_mean_r2": f"h{horizon}_local_mean_r2",
            "horizon_positive_r2_fraction": f"h{horizon}_positive_r2_fraction",
            "horizon_positive_local_mean_r2_fraction": (
                f"h{horizon}_positive_local_mean_r2_fraction"
            ),
            "horizon_skill_persistence": f"h{horizon}_skill_persistence",
            "horizon_skill_ar1": f"h{horizon}_skill_ar1",
        }
    )
    development_summary = development_summary.merge(
        horizon_values,
        on=configuration_columns,
        how="left",
        validate="one_to_one",
    )

# Count origins where DMD beats both dynamic baselines on the integrated score.
strong_baseline_counts = (
    development_table.assign(
        beats_both_dynamic_baselines=lambda table: (
            (table["skill_vs_persistence"] > 0) & (table["skill_vs_ar1"] > 0)
        )
    )
    .groupby(configuration_columns, as_index=False)
    .agg(
        beats_both_fraction=(
            "beats_both_dynamic_baselines",
            lambda values: float(np.mean(values)),
        )
    )
)
development_summary = development_summary.merge(
    strong_baseline_counts,
    on=configuration_columns,
    how="left",
    validate="one_to_one",
)

# Four chronological pairs supply a delete-two-block stability heuristic. The
# contexts overlap, so this is not presented as a formal independent-sample SE.
origin_to_pair = {
    origin: pair_index
    for pair_index, pair in enumerate(
        np.asarray(DEVELOPMENT_ORIGINS).reshape(-1, 2)
    )
    for origin in pair
}
development_table["delete_pair"] = development_table["origin_bin"].map(origin_to_pair)

jackknife_rows: list[dict[str, float | int]] = []
for configuration_key, group in development_table.groupby(
    configuration_columns, sort=False
):
    leave_pair_out = []
    for pair_index in sorted(group["delete_pair"].unique()):
        leave_pair_out.append(
            float(group.loc[group["delete_pair"] != pair_index, "primary_loss"].mean())
        )
    leave_pair_out_array = np.asarray(leave_pair_out)
    leave_pair_out_mean = float(np.mean(leave_pair_out_array))
    n_groups = leave_pair_out_array.size
    jackknife_se = float(
        np.sqrt(
            (n_groups - 1)
            / n_groups
            * np.sum((leave_pair_out_array - leave_pair_out_mean) ** 2)
        )
    )
    if not isinstance(configuration_key, tuple):
        configuration_key = (configuration_key,)
    jackknife_rows.append(
        {
            **dict(zip(configuration_columns, configuration_key, strict=True)),
            "delete_pair_jackknife_se": jackknife_se,
            "delete_pair_loss_range": float(np.ptp(leave_pair_out_array)),
        }
    )
development_summary = development_summary.merge(
    pd.DataFrame(jackknife_rows),
    on=configuration_columns,
    how="left",
    validate="one_to_one",
)

development_summary["development_gate_pass"] = (
    (development_summary["successful_origins"] == len(DEVELOPMENT_ORIGINS))
    & (development_summary["development_r2"] > 0)
    & (
        development_summary["positive_r2_fraction"]
        >= MIN_POSITIVE_ORIGIN_FRACTION
    )
    & (development_summary["local_mean_development_r2"] > 0)
    & (
        development_summary["positive_local_mean_r2_fraction"]
        >= MIN_POSITIVE_ORIGIN_FRACTION
    )
    & (development_summary["skill_vs_persistence"] > 0)
    & (development_summary["skill_vs_ar1"] > 0)
    & (development_summary["h1_r2"] > 0)
    & (development_summary["h1_local_mean_r2"] > 0)
    & (
        development_summary["h1_positive_r2_fraction"]
        >= MIN_POSITIVE_ORIGIN_FRACTION
    )
    & (
        development_summary["h1_positive_local_mean_r2_fraction"]
        >= MIN_POSITIVE_ORIGIN_FRACTION
    )
    & (development_summary["h1_skill_persistence"] > 0)
    & (development_summary["h1_skill_ar1"] > 0)
    & (development_summary[f"h{FORECAST_HORIZON_BINS}_r2"] > 0)
    & (
        development_summary[f"h{FORECAST_HORIZON_BINS}_local_mean_r2"]
        > 0
    )
    & (
        development_summary[f"h{FORECAST_HORIZON_BINS}_positive_r2_fraction"]
        >= MIN_POSITIVE_ORIGIN_FRACTION
    )
    & (
        development_summary[
            f"h{FORECAST_HORIZON_BINS}_positive_local_mean_r2_fraction"
        ]
        >= MIN_POSITIVE_ORIGIN_FRACTION
    )
    & (
        development_summary[f"h{FORECAST_HORIZON_BINS}_skill_persistence"]
        > 0
    )
    & (development_summary[f"h{FORECAST_HORIZON_BINS}_skill_ar1"] > 0)
    & (development_summary["beats_both_fraction"] >= MIN_POSITIVE_ORIGIN_FRACTION)
    & np.isfinite(development_summary["imaginary_leakage_max"])
    & (development_summary["unexpected_warning_count"] == 0)
)

# Every DMD prediction is constrained to local_mean + col(pod_basis). Its
# development R² therefore cannot exceed the future-informed projection
# oracle at the same W. This invariant catches mistakes in map-back or scoring.
selected_oracle_ceiling = pod_oracle_summary.loc[
    pod_oracle_summary["pod_dimension"] == pod_dimension,
    [
        "window_nominal_seconds",
        "oracle_common_mean_development_r2",
        "oracle_development_r2",
    ],
]
development_summary = development_summary.merge(
    selected_oracle_ceiling,
    on="window_nominal_seconds",
    how="left",
    validate="many_to_one",
)
if np.any(
    development_summary["local_mean_development_r2"]
    > development_summary["oracle_development_r2"] + 1e-10
):
    raise RuntimeError("A local-mean DMD score exceeds its POD oracle ceiling")
if np.any(
    development_summary["development_r2"]
    > development_summary["oracle_common_mean_development_r2"] + 1e-10
):
    raise RuntimeError("A common-mean DMD score exceeds its POD oracle ceiling")

development_summary.to_csv(OUTPUT_DIR / "development_configuration_summary.csv", index=False)

passing_summary = development_summary.loc[
    development_summary["development_gate_pass"]
].copy()
selection_pool = passing_summary if not passing_summary.empty else development_summary
selection_pool_best = selection_pool.sort_values(
    ["primary_loss_mean", "window_nominal_seconds", "rank", "delay"],
    kind="stable",
).iloc[0]
one_se_threshold = float(
    selection_pool_best["primary_loss_mean"]
    + selection_pool_best["delete_pair_jackknife_se"]
)
one_se_set = selection_pool.loc[
    selection_pool["primary_loss_mean"] <= one_se_threshold
].copy()
selected = one_se_set.sort_values(
    ["window_nominal_seconds", "rank", "delay", "primary_loss_mean"],
    kind="stable",
).iloc[0]

selected_window_seconds = int(selected["window_nominal_seconds"])
selected_window_bins = int(selected["window_bins"])
selected_delay = int(selected["delay"])
selected_rank = int(selected["rank"])
development_pass = bool(selected["development_gate_pass"])
print("DEVELOPMENT SWEEP COMPLETE")
print(f"  elapsed: {time.perf_counter() - sweep_start:.1f} s")
print(f"  successful fits: {len(development_table):,}")
print(f"  failed fits:     {len(failure_table):,}")
print(f"  configurations passing the predeclared gate: {len(passing_summary)}")
print("RAW LOWEST-LOSS CONFIGURATION IN THE SELECTION POOL")
print(
    f"  W={int(selection_pool_best['window_nominal_seconds'])} s, "
    f"d={int(selection_pool_best['delay'])}, "
    f"r={int(selection_pool_best['rank'])}, "
    f"fixed-mean R²={float(selection_pool_best['development_r2']):.4f}, "
    f"local-mean R²={float(selection_pool_best['local_mean_development_r2']):.4f}"
)
print(
    "ONE-DELETE-PAIR-SE PARSIMONY CHOICE"
    if not passing_summary.empty
    else "FALLBACK ONE-DELETE-PAIR-SE DIAGNOSTIC (no gate-passing model)"
)
print(
    f"  W={selected_window_seconds} s ({selected_window_bins * DT_SECONDS:.3f} actual), "
    f"d={selected_delay} ({(selected_delay - 1) * DT_SECONDS:.3f} s history), "
    f"r={selected_rank}"
)
print(f"  fixed-mean R²:              {float(selected['development_r2']):.4f}")
print(
    "  local-mean incremental R²:  "
    f"{float(selected['local_mean_development_r2']):.4f}"
)
print(f"  skill vs persistence:       {float(selected['skill_vs_persistence']):.4f}")
print(f"  skill vs neuronwise AR(1):  {float(selected['skill_vs_ar1']):.4f}")
print(f"  fixed-mean positive origins:{float(selected['positive_r2_fraction']):.3f}")
print(
    "  local-mean positive origins:"
    f"{float(selected['positive_local_mean_r2_fraction']):.3f}"
)
print(f"  passes development gate:    {development_pass}")

# Selection stability is recomputed after deleting each consecutive origin
# pair. Each six-origin screen repeats the complete technical gate and one-SE
# parsimony rule rather than auditing a different raw-minimum selection rule.
delete_pair_winners: list[dict[str, float | int]] = []
full_selected_tuple = (selected_window_seconds, selected_delay, selected_rank)
for deleted_pair in sorted(development_table["delete_pair"].unique()):
    remaining_origins = [
        origin
        for origin in DEVELOPMENT_ORIGINS
        if origin_to_pair[origin] != deleted_pair
    ]
    subset = development_table.loc[
        development_table["origin_bin"].isin(remaining_origins)
    ]
    subset_horizons = development_horizon_table.loc[
        development_horizon_table["origin_bin"].isin(remaining_origins)
    ]
    subset_summary = (
        subset.groupby(configuration_columns, as_index=False)
        .agg(
            successful_origins=("origin_bin", "nunique"),
            primary_loss_mean=("primary_loss", "mean"),
            development_r2=("common_mean_r2", "mean"),
            local_mean_development_r2=("local_mean_r2", "mean"),
            positive_r2_fraction=(
                "common_mean_r2", lambda values: float(np.mean(values > 0))
            ),
            positive_local_mean_r2_fraction=(
                "local_mean_r2", lambda values: float(np.mean(values > 0))
            ),
            skill_vs_persistence=("skill_vs_persistence", "mean"),
            skill_vs_ar1=("skill_vs_ar1", "mean"),
            imaginary_leakage_max=("imaginary_leakage", "max"),
            unexpected_warning_count=("unexpected_warning_count", "sum"),
        )
    )
    subset_summary = subset_summary.loc[
        subset_summary["successful_origins"] == len(remaining_origins)
    ]
    if subset_summary.empty:
        raise RuntimeError("No complete configuration in a delete-pair screen")

    subset_horizon_summary = (
        subset_horizons.groupby(
            configuration_columns + ["horizon_bins"], as_index=False
        )
        .agg(
            horizon_r2=("r2_vs_common_mean", "mean"),
            horizon_local_mean_r2=("r2_vs_local_mean", "mean"),
            horizon_positive_r2_fraction=(
                "r2_vs_common_mean", lambda values: float(np.mean(values > 0))
            ),
            horizon_positive_local_mean_r2_fraction=(
                "r2_vs_local_mean", lambda values: float(np.mean(values > 0))
            ),
            horizon_skill_persistence=("skill_vs_persistence", "mean"),
            horizon_skill_ar1=("skill_vs_ar1", "mean"),
        )
    )
    for horizon in (1, FORECAST_HORIZON_BINS):
        horizon_values = subset_horizon_summary.loc[
            subset_horizon_summary["horizon_bins"] == horizon,
            configuration_columns
            + [
                "horizon_r2",
                "horizon_local_mean_r2",
                "horizon_positive_r2_fraction",
                "horizon_positive_local_mean_r2_fraction",
                "horizon_skill_persistence",
                "horizon_skill_ar1",
            ],
        ].rename(
            columns={
                "horizon_r2": f"h{horizon}_r2",
                "horizon_local_mean_r2": f"h{horizon}_local_mean_r2",
                "horizon_positive_r2_fraction": (
                    f"h{horizon}_positive_r2_fraction"
                ),
                "horizon_positive_local_mean_r2_fraction": (
                    f"h{horizon}_positive_local_mean_r2_fraction"
                ),
                "horizon_skill_persistence": f"h{horizon}_skill_persistence",
                "horizon_skill_ar1": f"h{horizon}_skill_ar1",
            }
        )
        subset_summary = subset_summary.merge(
            horizon_values,
            on=configuration_columns,
            how="left",
            validate="one_to_one",
        )

    subset_both = (
        subset.assign(
            beats_both=lambda table: (
                (table["skill_vs_persistence"] > 0)
                & (table["skill_vs_ar1"] > 0)
            )
        )
        .groupby(configuration_columns, as_index=False)
        .agg(beats_both_fraction=("beats_both", lambda x: float(np.mean(x))))
    )
    subset_summary = subset_summary.merge(
        subset_both,
        on=configuration_columns,
        how="left",
        validate="one_to_one",
    )
    subset_summary["development_gate_pass"] = (
        (subset_summary["development_r2"] > 0)
        & (subset_summary["local_mean_development_r2"] > 0)
        & (subset_summary["positive_r2_fraction"] >= MIN_POSITIVE_ORIGIN_FRACTION)
        & (
            subset_summary["positive_local_mean_r2_fraction"]
            >= MIN_POSITIVE_ORIGIN_FRACTION
        )
        & (subset_summary["skill_vs_persistence"] > 0)
        & (subset_summary["skill_vs_ar1"] > 0)
        & (subset_summary["h1_r2"] > 0)
        & (subset_summary["h1_local_mean_r2"] > 0)
        & (
            subset_summary["h1_positive_r2_fraction"]
            >= MIN_POSITIVE_ORIGIN_FRACTION
        )
        & (
            subset_summary["h1_positive_local_mean_r2_fraction"]
            >= MIN_POSITIVE_ORIGIN_FRACTION
        )
        & (subset_summary["h1_skill_persistence"] > 0)
        & (subset_summary["h1_skill_ar1"] > 0)
        & (subset_summary[f"h{FORECAST_HORIZON_BINS}_r2"] > 0)
        & (subset_summary[f"h{FORECAST_HORIZON_BINS}_local_mean_r2"] > 0)
        & (
            subset_summary[f"h{FORECAST_HORIZON_BINS}_positive_r2_fraction"]
            >= MIN_POSITIVE_ORIGIN_FRACTION
        )
        & (
            subset_summary[
                f"h{FORECAST_HORIZON_BINS}_positive_local_mean_r2_fraction"
            ]
            >= MIN_POSITIVE_ORIGIN_FRACTION
        )
        & (subset_summary[f"h{FORECAST_HORIZON_BINS}_skill_persistence"] > 0)
        & (subset_summary[f"h{FORECAST_HORIZON_BINS}_skill_ar1"] > 0)
        & (subset_summary["beats_both_fraction"] >= MIN_POSITIVE_ORIGIN_FRACTION)
        & np.isfinite(subset_summary["imaginary_leakage_max"])
        & (subset_summary["unexpected_warning_count"] == 0)
    )

    subset_se_rows: list[dict[str, float | int]] = []
    for configuration_key, group in subset.groupby(
        configuration_columns, sort=False
    ):
        group_pair = group["origin_bin"].map(origin_to_pair)
        leave_pair_losses = np.asarray(
            [
                group.loc[group_pair != pair, "primary_loss"].mean()
                for pair in sorted(group_pair.unique())
            ],
            dtype=float,
        )
        n_groups = leave_pair_losses.size
        se = float(
            np.sqrt(
                (n_groups - 1)
                / n_groups
                * np.sum((leave_pair_losses - leave_pair_losses.mean()) ** 2)
            )
        )
        if not isinstance(configuration_key, tuple):
            configuration_key = (configuration_key,)
        subset_se_rows.append(
            {
                **dict(zip(configuration_columns, configuration_key, strict=True)),
                "delete_pair_jackknife_se": se,
            }
        )
    subset_summary = subset_summary.merge(
        pd.DataFrame(subset_se_rows),
        on=configuration_columns,
        how="left",
        validate="one_to_one",
    )
    subset_passing = subset_summary.loc[subset_summary["development_gate_pass"]]
    subset_pool = subset_passing if not subset_passing.empty else subset_summary
    subset_best = subset_pool.sort_values(
        ["primary_loss_mean", "window_nominal_seconds", "rank", "delay"],
        kind="stable",
    ).iloc[0]
    subset_threshold = float(
        subset_best["primary_loss_mean"] + subset_best["delete_pair_jackknife_se"]
    )
    winner = subset_pool.loc[
        subset_pool["primary_loss_mean"] <= subset_threshold
    ].sort_values(
        ["window_nominal_seconds", "rank", "delay", "primary_loss_mean"],
        kind="stable",
    ).iloc[0]
    winner_tuple = (
        int(winner["window_nominal_seconds"]),
        int(winner["delay"]),
        int(winner["rank"]),
    )
    delete_pair_winners.append(
        {
            "deleted_pair": int(deleted_pair),
            "window_nominal_seconds": int(winner["window_nominal_seconds"]),
            "delay": int(winner["delay"]),
            "rank": int(winner["rank"]),
            "primary_loss_mean": float(winner["primary_loss_mean"]),
            "gate_passing_pool_available": bool(not subset_passing.empty),
            "matches_full_one_se_choice": winner_tuple == full_selected_tuple,
        }
    )
delete_pair_winner_table = pd.DataFrame(delete_pair_winners)
delete_pair_winner_table.to_csv(
    OUTPUT_DIR / "development_delete_pair_winners.csv", index=False
)
print("DELETE-TWO-ORIGIN ONE-SE SELECTIONS (stability heuristic)")
print(delete_pair_winner_table.to_string(index=False))

# Two views are necessary. Rank 2 is feasible throughout the entire W,d grid
# and isolates the duration/delay comparison. The second panel shows the best
# rank allowed by the regularized joint search.
rank_two = development_summary.loc[development_summary["rank"] == 2]
best_over_rank = (
    development_summary.sort_values("primary_loss_mean", kind="stable")
    .groupby(["window_nominal_seconds", "delay"], as_index=False)
    .first()
)

fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), constrained_layout=True)
for column, (table, title) in enumerate(
    (
        (rank_two, "Common capacity: fixed r=2"),
        (best_over_rank, "Best r for each W,d by common-reference loss"),
    )
):
    for row, (metric, metric_label) in enumerate(
        (
            ("development_r2", "fixed acquisition-1 mean R²"),
            ("local_mean_development_r2", "local-mean incremental R²"),
        )
    ):
        axis = axes[row, column]
        pivot = table.pivot(
            index="delay",
            columns="window_nominal_seconds",
            values=metric,
        )
        color_limit = max(0.05, float(np.nanmax(np.abs(pivot.to_numpy()))))
        image = axis.imshow(
            pivot.to_numpy(),
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            vmin=-color_limit,
            vmax=color_limit,
        )
        axis.set(
            xticks=np.arange(pivot.shape[1]),
            xticklabels=pivot.columns.astype(int),
            yticks=np.arange(pivot.shape[0]),
            yticklabels=pivot.index.astype(int),
            xlabel="fit window W (s)",
            ylabel="delay depth d",
            title=f"{title}\n{metric_label}",
        )
        for row_index in range(pivot.shape[0]):
            for column_index in range(pivot.shape[1]):
                value = pivot.iloc[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        fig.colorbar(image, ax=axis, label=metric_label)
fig.suptitle("Step 7 — Fair selection score and local incremental effect are distinct")
save_and_show(fig, "04_development_window_delay_heatmaps.png")

selected_origins = development_table.loc[
    (development_table["window_bins"] == selected_window_bins)
    & (development_table["delay"] == selected_delay)
    & (development_table["rank"] == selected_rank)
].sort_values("origin_bin")
selected_horizons = development_horizon_table.loc[
    (development_horizon_table["window_bins"] == selected_window_bins)
    & (development_horizon_table["delay"] == selected_delay)
    & (development_horizon_table["rank"] == selected_rank)
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
axes[0].plot(
    selected_origins["origin_seconds"],
    selected_origins["common_mean_r2"],
    marker="o",
    label="DMD vs fixed acquisition-1 mean",
)
axes[0].plot(
    selected_origins["origin_seconds"],
    selected_origins["local_mean_r2"],
    marker="o",
    label="DMD vs local past mean",
)
axes[0].plot(
    selected_origins["origin_seconds"],
    selected_origins["skill_vs_persistence"],
    marker="o",
    label="skill vs persistence",
)
axes[0].plot(
    selected_origins["origin_seconds"],
    selected_origins["skill_vs_ar1"],
    marker="o",
    label="skill vs AR(1)",
)
axes[0].axhline(0, color="0.5", linestyle="--")
axes[0].set(
    xlabel="development target origin (s)",
    ylabel="score (positive is better)",
    title="Every development target block",
)
axes[0].legend(frameon=False, fontsize=8)

horizon_plot = (
    selected_horizons.groupby("horizon_seconds", as_index=False)
    .agg(
        common_r2=("r2_vs_common_mean", "mean"),
        local_r2=("r2_vs_local_mean", "mean"),
        persistence=("skill_vs_persistence", "mean"),
        ar1=("skill_vs_ar1", "mean"),
    )
)
axes[1].plot(
    horizon_plot["horizon_seconds"],
    horizon_plot["common_r2"],
    marker="o",
    label="R² vs fixed mean",
)
axes[1].plot(
    horizon_plot["horizon_seconds"],
    horizon_plot["local_r2"],
    marker="o",
    label="R² vs local mean",
)
axes[1].plot(
    horizon_plot["horizon_seconds"],
    horizon_plot["persistence"],
    marker="o",
    label="skill vs persistence",
)
axes[1].plot(
    horizon_plot["horizon_seconds"],
    horizon_plot["ar1"],
    marker="o",
    label="skill vs AR(1)",
)
axes[1].axhline(0, color="0.5", linestyle="--")
axes[1].set(
    xlabel="autonomous forecast horizon (s)",
    ylabel="mean development score",
    title="Error growth without teacher forcing",
)
axes[1].legend(frameon=False, fontsize=8)
fig.suptitle(
    f"Step 7 — Selected development model: W={selected_window_seconds}s, "
    f"d={selected_delay}, r={selected_rank}"
)
save_and_show(fig, "05_selected_development_scores.png")


# %% Step 8 — Noise-aware PyDMD optimized-DMD shortlist on development only
#
# PyDMD's real-data tutorial recommends delay embedding plus BOP-DMD for noisy
# measurements. A full second Cartesian sweep would multiply the winner's
# curse, so the shortlist is deterministic: the best exact-Hankel configuration
# for each d, the one-SE parsimony choice, and the best current-duration
# (120-s) anchor. Two physically interpretable eigenvalue constraints are
# compared. ``num_trials=0`` means optimized DMD without bagging; bagging is
# reserved for a finalist and its uncertainty is algorithmic, not a time-series
# confidence interval.


def fit_bop_at_origin(
    window_bins: int,
    delay: int,
    rank: int,
    origin: int,
    stable: bool,
) -> tuple[
    OriginForecast,
    list[dict[str, float | int]],
    object,
    int,
    list[dict[str, bool | str]],
]:
    """Fit endpoint-conditioned, Hankel-preprocessed optimized DMD."""
    prepared = standardized_origin(window_bins, origin)
    latent_context = prepared.latent_context

    constraints = {"conjugate_pairs"}
    if stable:
        constraints.add("stable")
    model = hankel_preprocessing(
        BOPDMD(
            svd_rank=rank,
            compute_A=False,
            use_proj=True,
            num_trials=0,
            eig_sort="auto",
            eig_constraints=constraints,
            varpro_opts_dict={"maxiter": 100, "tol": 1e-6},
        ),
        d=delay,
        reconstruction_method="mean",
    )
    embedded_columns = window_bins - delay + 1
    embedded_time = np.arange(embedded_columns) * DT_SECONDS
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(latent_context, embedded_time)

    warning_records = unique_warning_records(caught)
    convergence_warning_count = sum(
        "failed to converge" in str(record["warning_message"]).lower()
        for record in warning_records
    )
    latent_prediction, imaginary_leakage = forecast_bopdmd_from_endpoint(
        model,
        latent_context,
        delay,
        FORECAST_HORIZON_BINS,
        DT_SECONDS,
    )
    one_step_growth_multiplier = float(
        np.exp(np.max(np.real(model.eigs)) * DT_SECONDS)
    )
    forecast_score, horizon_rows = score_latent_forecast(
        prepared,
        latent_prediction,
        imaginary_leakage,
        one_step_growth_multiplier,
        len(warning_records),
        sum(bool(record["unexpected_warning"]) for record in warning_records),
    )
    return (
        forecast_score,
        horizon_rows,
        model,
        convergence_warning_count,
        warning_records,
    )


best_by_delay = (
    development_summary.sort_values("primary_loss_mean", kind="stable")
    .groupby("delay", as_index=False)
    .first()
)
best_by_window = (
    development_summary.sort_values("primary_loss_mean", kind="stable")
    .groupby("window_nominal_seconds", as_index=False)
    .first()
)
best_local_by_window = (
    development_summary.sort_values(
        "local_mean_development_r2", ascending=False, kind="stable"
    )
    .groupby("window_nominal_seconds", as_index=False)
    .first()
)
best_local_global = best_local_by_window.sort_values(
    "local_mean_development_r2", ascending=False, kind="stable"
).head(1)
current_duration_anchor = (
    development_summary.loc[
        development_summary["window_nominal_seconds"] == 120
    ]
    .sort_values("primary_loss_mean", kind="stable")
    .head(1)
)
shortlist = pd.concat(
    [
        best_by_delay,
        pd.DataFrame([selected]),
        best_local_global,
        current_duration_anchor,
    ],
    ignore_index=True,
).drop_duplicates(
    subset=["window_nominal_seconds", "delay", "rank"],
    keep="first",
)
shortlist.to_csv(OUTPUT_DIR / "bopdmd_shortlist.csv", index=False)

# Spectral adequacy is a mandatory second-stage criterion, so every
# technically gate-passing exact-Hankel configuration must receive that audit;
# limiting it to the prediction minimum could miss a slightly less predictive
# but trackable mode family. Best-by-delay, best-by-window under both the common
# selection loss and local incremental effect, and the old-duration anchor
# remain included to diagnose why nonpassing alternatives failed.
spectral_shortlist = pd.concat(
    [
        passing_summary,
        best_by_delay,
        best_by_window,
        best_local_by_window,
        pd.DataFrame([selected]),
        current_duration_anchor,
    ],
    ignore_index=True,
).drop_duplicates(
    subset=["window_nominal_seconds", "delay", "rank"],
    keep="first",
).sort_values(
    ["primary_loss_mean", "window_nominal_seconds", "rank", "delay"],
    kind="stable",
)
spectral_shortlist.to_csv(
    OUTPUT_DIR / "exact_spectral_shortlist.csv", index=False
)

bop_rows: list[dict[str, float | int | bool | str]] = []
bop_horizon_rows: list[dict[str, float | int | bool | str]] = []
bop_failures: list[dict[str, float | int | bool | str]] = []
bop_warning_rows: list[dict[str, float | int | bool | str]] = []

print("OPTIMIZED-DMD SHORTLIST")
print(
    shortlist[
        ["window_nominal_seconds", "delay", "rank", "development_r2"]
    ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
)

for candidate in tqdm(
    list(shortlist.itertuples(index=False)),
    desc="optimized-DMD shortlist",
    disable=not sys.stderr.isatty(),
):
    for stable_constraint in (False, True):
        constraint_name = (
            "stable+conjugate_pairs" if stable_constraint else "conjugate_pairs"
        )
        for origin in DEVELOPMENT_ORIGINS:
            key = {
                "window_nominal_seconds": int(candidate.window_nominal_seconds),
                "window_bins": int(candidate.window_bins),
                "window_actual_seconds": float(candidate.window_actual_seconds),
                "delay": int(candidate.delay),
                "history_seconds": float(candidate.history_seconds),
                "rank": int(candidate.rank),
                "constraint": constraint_name,
            }
            try:
                (
                    origin_score,
                    horizon_scores,
                    _,
                    convergence_warnings,
                    origin_warnings,
                ) = fit_bop_at_origin(
                    int(candidate.window_bins),
                    int(candidate.delay),
                    int(candidate.rank),
                    origin,
                    stable_constraint,
                )
                bop_rows.append(
                    {
                        **key,
                        "origin_bin": origin,
                        "origin_seconds": origin * DT_SECONDS,
                        **asdict(origin_score),
                        "convergence_warning_count": convergence_warnings,
                    }
                )
                for horizon_score in horizon_scores:
                    bop_horizon_rows.append(
                        {
                            **key,
                            "origin_bin": origin,
                            "origin_seconds": origin * DT_SECONDS,
                            **horizon_score,
                        }
                    )
                for warning_record in origin_warnings:
                    bop_warning_rows.append(
                        {
                            **key,
                            "origin_bin": origin,
                            **warning_record,
                        }
                    )
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                bop_failures.append(
                    {
                        **key,
                        "origin_bin": origin,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )

bop_table = pd.DataFrame(bop_rows)
bop_horizon_table = pd.DataFrame(bop_horizon_rows)
bop_failure_columns = [
    "window_nominal_seconds",
    "window_bins",
    "window_actual_seconds",
    "delay",
    "history_seconds",
    "rank",
    "constraint",
    "origin_bin",
    "error_type",
    "error",
]
bop_warning_columns = [
    "window_nominal_seconds",
    "window_bins",
    "window_actual_seconds",
    "delay",
    "history_seconds",
    "rank",
    "constraint",
    "origin_bin",
    "warning_category",
    "warning_message",
    "expected_rank_truncation_warning",
    "unexpected_warning",
]
bop_failure_table = pd.DataFrame(bop_failures, columns=bop_failure_columns)
bop_warning_table = pd.DataFrame(bop_warning_rows, columns=bop_warning_columns)
bop_table.to_csv(OUTPUT_DIR / "bopdmd_development_origin_scores.csv", index=False)
bop_horizon_table.to_csv(
    OUTPUT_DIR / "bopdmd_development_horizon_scores.csv", index=False
)
bop_failure_table.to_csv(
    OUTPUT_DIR / "bopdmd_development_failures.csv", index=False
)
bop_warning_table.to_csv(
    OUTPUT_DIR / "bopdmd_development_warnings.csv", index=False
)

bop_configuration_columns = [
    "window_nominal_seconds",
    "window_bins",
    "window_actual_seconds",
    "delay",
    "history_seconds",
    "rank",
    "constraint",
]
if bop_table.empty:
    bop_summary = pd.DataFrame()
else:
    bop_summary = (
        bop_table.groupby(bop_configuration_columns, as_index=False)
        .agg(
            successful_origins=("origin_bin", "nunique"),
            primary_loss_mean=("primary_loss", "mean"),
            development_r2=("common_mean_r2", "mean"),
            local_mean_development_r2=("local_mean_r2", "mean"),
            positive_r2_fraction=(
                "common_mean_r2", lambda values: float(np.mean(values > 0))
            ),
            positive_local_mean_r2_fraction=(
                "local_mean_r2", lambda values: float(np.mean(values > 0))
            ),
            skill_vs_persistence=("skill_vs_persistence", "mean"),
            skill_vs_ar1=("skill_vs_ar1", "mean"),
            h4_r2=("h4_common_mean_r2", "mean"),
            h4_local_mean_r2=("h4_local_mean_r2", "mean"),
            imaginary_leakage_max=("imaginary_leakage", "max"),
            warning_count=("warning_count", "sum"),
            unexpected_warning_count=("unexpected_warning_count", "sum"),
            convergence_warning_count=("convergence_warning_count", "sum"),
        )
    )
    bop_horizon_summary = (
        bop_horizon_table.groupby(
            bop_configuration_columns + ["horizon_bins"], as_index=False
        )
        .agg(
            horizon_r2=("r2_vs_common_mean", "mean"),
            horizon_local_mean_r2=("r2_vs_local_mean", "mean"),
            horizon_positive_r2_fraction=(
                "r2_vs_common_mean", lambda values: float(np.mean(values > 0))
            ),
            horizon_positive_local_mean_r2_fraction=(
                "r2_vs_local_mean", lambda values: float(np.mean(values > 0))
            ),
            horizon_skill_persistence=("skill_vs_persistence", "mean"),
            horizon_skill_ar1=("skill_vs_ar1", "mean"),
        )
    )
    for horizon in (1, FORECAST_HORIZON_BINS):
        horizon_values = bop_horizon_summary.loc[
            bop_horizon_summary["horizon_bins"] == horizon,
            bop_configuration_columns
            + [
                "horizon_r2",
                "horizon_local_mean_r2",
                "horizon_positive_r2_fraction",
                "horizon_positive_local_mean_r2_fraction",
                "horizon_skill_persistence",
                "horizon_skill_ar1",
            ],
        ].rename(
            columns={
                "horizon_r2": f"h{horizon}_r2",
                "horizon_local_mean_r2": f"h{horizon}_local_mean_r2",
                "horizon_positive_r2_fraction": (
                    f"h{horizon}_positive_r2_fraction"
                ),
                "horizon_positive_local_mean_r2_fraction": (
                    f"h{horizon}_positive_local_mean_r2_fraction"
                ),
                "horizon_skill_persistence": f"h{horizon}_skill_persistence",
                "horizon_skill_ar1": f"h{horizon}_skill_ar1",
            }
        )
        bop_summary = bop_summary.merge(
            horizon_values,
            on=bop_configuration_columns,
            how="left",
            validate="one_to_one",
        )

    bop_both_counts = (
        bop_table.assign(
            beats_both=lambda table: (
                (table["skill_vs_persistence"] > 0)
                & (table["skill_vs_ar1"] > 0)
            )
        )
        .groupby(bop_configuration_columns, as_index=False)
        .agg(beats_both_fraction=("beats_both", lambda values: float(np.mean(values))))
    )
    bop_summary = bop_summary.merge(
        bop_both_counts,
        on=bop_configuration_columns,
        how="left",
        validate="one_to_one",
    )
    bop_summary["development_gate_pass"] = (
        (bop_summary["successful_origins"] == len(DEVELOPMENT_ORIGINS))
        & (bop_summary["development_r2"] > 0)
        & (bop_summary["local_mean_development_r2"] > 0)
        & (bop_summary["positive_r2_fraction"] >= MIN_POSITIVE_ORIGIN_FRACTION)
        & (
            bop_summary["positive_local_mean_r2_fraction"]
            >= MIN_POSITIVE_ORIGIN_FRACTION
        )
        & (bop_summary["skill_vs_persistence"] > 0)
        & (bop_summary["skill_vs_ar1"] > 0)
        & (bop_summary["h1_r2"] > 0)
        & (bop_summary["h1_local_mean_r2"] > 0)
        & (bop_summary["h1_positive_r2_fraction"] >= MIN_POSITIVE_ORIGIN_FRACTION)
        & (
            bop_summary["h1_positive_local_mean_r2_fraction"]
            >= MIN_POSITIVE_ORIGIN_FRACTION
        )
        & (bop_summary["h1_skill_persistence"] > 0)
        & (bop_summary["h1_skill_ar1"] > 0)
        & (bop_summary[f"h{FORECAST_HORIZON_BINS}_r2"] > 0)
        & (bop_summary[f"h{FORECAST_HORIZON_BINS}_local_mean_r2"] > 0)
        & (
            bop_summary[f"h{FORECAST_HORIZON_BINS}_positive_r2_fraction"]
            >= MIN_POSITIVE_ORIGIN_FRACTION
        )
        & (
            bop_summary[
                f"h{FORECAST_HORIZON_BINS}_positive_local_mean_r2_fraction"
            ]
            >= MIN_POSITIVE_ORIGIN_FRACTION
        )
        & (bop_summary[f"h{FORECAST_HORIZON_BINS}_skill_persistence"] > 0)
        & (bop_summary[f"h{FORECAST_HORIZON_BINS}_skill_ar1"] > 0)
        & (bop_summary["beats_both_fraction"] >= MIN_POSITIVE_ORIGIN_FRACTION)
        & (bop_summary["convergence_warning_count"] == 0)
        & (bop_summary["unexpected_warning_count"] == 0)
    )

bop_summary.to_csv(OUTPUT_DIR / "bopdmd_development_summary.csv", index=False)

final_family = "HankelDMD"
final_window_seconds = selected_window_seconds
final_window_bins = selected_window_bins
final_delay = selected_delay
final_rank = selected_rank
final_constraint = "none"
final_development_common_r2 = float(selected["development_r2"])
final_development_local_r2 = float(selected["local_mean_development_r2"])
final_technical_pass = development_pass
bop_clear_improvement = False
bop_paired_improvement = np.nan
bop_paired_jackknife_se = np.nan

eligible_bop = (
    bop_summary.loc[bop_summary["development_gate_pass"]].copy()
    if not bop_summary.empty
    else pd.DataFrame()
)
if not eligible_bop.empty:
    best_bop = eligible_bop.sort_values(
        ["primary_loss_mean", "window_nominal_seconds", "rank", "delay"],
        kind="stable",
    ).iloc[0]
    best_bop_origins = bop_table.loc[
        (bop_table["window_bins"] == int(best_bop["window_bins"]))
        & (bop_table["delay"] == int(best_bop["delay"]))
        & (bop_table["rank"] == int(best_bop["rank"]))
        & (bop_table["constraint"] == best_bop["constraint"])
    ].sort_values("origin_bin")
    exact_bop_comparator = development_table.loc[
        (development_table["window_bins"] == int(best_bop["window_bins"]))
        & (development_table["delay"] == int(best_bop["delay"]))
        & (development_table["rank"] == int(best_bop["rank"]))
    ].sort_values("origin_bin")
    paired = exact_bop_comparator[["origin_bin", "primary_loss"]].merge(
        best_bop_origins[["origin_bin", "primary_loss"]],
        on="origin_bin",
        suffixes=("_hankel", "_bop"),
        validate="one_to_one",
    )
    paired["improvement"] = paired["primary_loss_hankel"] - paired["primary_loss_bop"]
    bop_paired_improvement = float(paired["improvement"].mean())
    leave_pair_improvement = []
    for pair_index, pair_origins in enumerate(
        np.asarray(DEVELOPMENT_ORIGINS).reshape(-1, 2)
    ):
        del pair_index
        leave_pair_improvement.append(
            float(
                paired.loc[
                    ~paired["origin_bin"].isin(pair_origins), "improvement"
                ].mean()
            )
        )
    leave_pair_improvement = np.asarray(leave_pair_improvement)
    bop_paired_jackknife_se = float(
        np.sqrt(
            3
            / 4
            * np.sum(
                (leave_pair_improvement - leave_pair_improvement.mean()) ** 2
            )
        )
    )
    bop_clear_improvement = bool(
        bop_paired_improvement > bop_paired_jackknife_se
    )

print("OPTIMIZED-DMD DEVELOPMENT RESULT")
print(f"  configurations returning values: {len(bop_summary)}")
print(f"  gate-passing, convergence-clean configurations: {len(eligible_bop)}")
if not eligible_bop.empty:
    print(
        "  best optimized-DMD: "
        f"W={int(best_bop['window_nominal_seconds'])} s, "
        f"d={int(best_bop['delay'])}, r={int(best_bop['rank'])}, "
        f"constraint={best_bop['constraint']}, "
        f"fixed-mean R²={float(best_bop['development_r2']):.4f}, "
        f"local-mean R²={float(best_bop['local_mean_development_r2']):.4f}"
    )
    print(
        "  paired loss gain over same-configuration exact HankelDMD: "
        f"{bop_paired_improvement:.5f}"
    )
    print(f"  delete-pair jackknife SE of gain:    {bop_paired_jackknife_se:.5f}")
print(f"  clear improvement beyond heuristic SE: {bop_clear_improvement}")
print(
    "  estimator role: sensitivity only; it cannot replace the exact-Hankel "
    "finalist without its own spectral audit"
)
print(
    "FROZEN DEVELOPMENT FINALIST"
    if development_pass
    else "FROZEN DEVELOPMENT FALLBACK CANDIDATE"
)
print(
    f"  family={final_family}, W={final_window_seconds} s, "
    f"d={final_delay}, r={final_rank}, constraint={final_constraint}"
)
print(f"  fixed-mean R²={final_development_common_r2:.4f}")
print(f"  local-mean incremental R²={final_development_local_r2:.4f}")

fig, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
plot_rows = development_summary.loc[
    development_summary.set_index(
        ["window_nominal_seconds", "delay", "rank"]
    ).index.isin(
        shortlist.set_index(["window_nominal_seconds", "delay", "rank"]).index
    )
].copy()
plot_rows["label"] = plot_rows.apply(
    lambda row: f"H W{int(row.window_nominal_seconds)} d{int(row.delay)} r{int(row['rank'])}",
    axis=1,
)
axis.scatter(
    plot_rows["label"],
    plot_rows["local_mean_development_r2"],
    marker="o",
    s=55,
    label="exact HankelDMD",
)
clean_bop_plot = (
    bop_summary.loc[
        (bop_summary["convergence_warning_count"] == 0)
        & (bop_summary["unexpected_warning_count"] == 0)
    ]
    if not bop_summary.empty
    else pd.DataFrame()
)
if not clean_bop_plot.empty:
    for constraint, marker in (
        ("conjugate_pairs", "s"),
        ("stable+conjugate_pairs", "^"),
    ):
        subset = clean_bop_plot.loc[
            clean_bop_plot["constraint"] == constraint
        ].copy()
        subset["label"] = subset.apply(
            lambda row: f"H W{int(row.window_nominal_seconds)} d{int(row.delay)} r{int(row['rank'])}",
            axis=1,
        )
        axis.scatter(
            subset["label"],
            subset["local_mean_development_r2"],
            marker=marker,
            s=50,
            label=f"optimized: {constraint}",
        )
else:
    axis.text(
        0.02,
        0.04,
        "All optimized-DMD fits emitted convergence warnings;\n"
        "their returned scores are omitted.",
        transform=axis.transAxes,
        fontsize=9,
        color="#D55E00",
        va="bottom",
    )
axis.axhline(0, color="0.5", linestyle="--")
axis.axhline(
    PRACTICAL_R2_TARGET,
    color="#D55E00",
    linestyle=":",
    label=f"predeclared practical target R²={PRACTICAL_R2_TARGET:.2f}",
)
axis.tick_params(axis="x", rotation=45)
axis.set(
    xlabel="deterministic shortlist configuration",
    ylabel="local-mean incremental development R²",
    title="Only convergence-clean optimized-DMD scores are interpretable",
)
axis.legend(frameon=False, fontsize=8)
fig.suptitle("Step 8 — PyDMD optimized-DMD sensitivity before opening outer data")
save_and_show(fig, "06_bopdmd_shortlist_comparison.png")


# %% Step 9 — Audit recurrent modes across a deterministic exact-Hankel shortlist
#
# Forecasting and spectral verification are separate requirements. Inspecting
# only the rank-2 parsimony choice previously created a false conclusion: that
# particular model could not contain an oscillatory conjugate pair, while a
# rank-4 prediction candidate did. We therefore audit the deterministic
# shortlist saved in Step 8: every technically passing configuration plus the
# prediction winner for each delay, the one-SE choice, and the 120-s anchor.
# BOP-DMD cannot be promoted because the audit below is for exact HankelDMD
# modes.
#
# Adjacent windows are not independent. For W=30 s they share about 79% of
# their samples, and the overlap is even larger for longer W. A track may use
# adjacent matches for continuity, but it can pass only when the same
# frequency/spatial-mode plane also recurs across at least two origin pairs
# sharing no more than 25% of their fitting samples. Windows whose geometry
# offers fewer than two such pairs are labelled inconclusive, not failures.
#
# Pilot operational thresholds (not DMD theorems): one representative from a
# conjugate pair must complete >=3 cycles over the actual (W-d) transition
# span, account for >=1% of a descriptive fitted-component Frobenius-power
# proxy, and retain >=10% amplitude over one cycle. Matches must lie within one
# Rayleigh bin, have real-plane spatial affinity >=0.70, and have per-cycle
# damping multipliers within one natural-log unit. A recurrent component must
# cover >=6/8 origins and directly match at least 75% of its available
# low-overlap pairs.

MIN_MODE_CYCLES = 3.0
MIN_MODE_PAIR_CONTRIBUTION = 0.01
MIN_AMPLITUDE_MULTIPLIER_PER_CYCLE = 0.10
MAX_LOG_CYCLE_MULTIPLIER_DIFFERENCE = 1.0
MAX_FREQUENCY_BIN_DISTANCE = 1.0
MIN_MODE_PLANE_AFFINITY = 0.70
MAX_LOW_OVERLAP_FRACTION = 0.25
MIN_MODE_TRACK_COVERAGE = 0.75
MIN_LOW_OVERLAP_MATCH_FRACTION = 0.75
MIN_LOW_OVERLAP_PAIRS = 2
N_RESDMD_SURROGATES = 199


def physical_hankel_modes(model: HankelDMD, delay: int) -> np.ndarray:
    """Undo delay phase before mapping latent modes to fixed neuron space."""
    embedded_modes = np.asarray(model.modes)
    mode_count = embedded_modes.shape[1]
    blocks = embedded_modes.reshape(delay, pod_dimension, mode_count)
    aligned = np.empty_like(blocks, dtype=np.complex128)
    for mode_index, eigenvalue in enumerate(model.eigs):
        if abs(eigenvalue) <= 1e-10:
            aligned[:, :, mode_index] = blocks[:, :, mode_index]
        else:
            for lag in range(delay):
                aligned[lag, :, mode_index] = (
                    blocks[lag, :, mode_index] / eigenvalue**lag
                )
    latent_modes = np.mean(aligned, axis=0)
    neuron_modes = pod_basis @ latent_modes
    norms = np.linalg.norm(neuron_modes, axis=0)
    valid = norms > np.finfo(float).eps
    neuron_modes[:, valid] /= norms[valid]
    return neuron_modes


def subspace_affinity(left: np.ndarray, right: np.ndarray) -> float:
    """Mean cosine of principal angles; invariant to phase and mode order."""
    left_q, _ = scipy.linalg.qr(left, mode="economic")
    right_q, _ = scipy.linalg.qr(right, mode="economic")
    singular_values = scipy.linalg.svdvals(left_q.conj().T @ right_q)
    return float(np.mean(np.clip(singular_values, 0, 1)))


def oscillatory_mode_pairs(
    model: HankelDMD,
    latent_context: np.ndarray,
    window_seconds: int,
    window_bins: int,
    delay: int,
    rank: int,
    origin: int,
) -> tuple[list[dict[str, float | int | bool | str]], list[dict[str, object]]]:
    """Extract positive-frequency pairs passing resolution and damping gates."""
    eigenvalues = np.asarray(model.eigs, dtype=np.complex128)
    continuous_rates = np.log(eigenvalues) / DT_SECONDS
    embedded_modes = np.asarray(model.modes, dtype=np.complex128)
    hankel = pseudo_hankel_matrix(latent_context, delay)
    coefficients = np.linalg.lstsq(embedded_modes, hankel, rcond=None)[0]
    component_power = np.asarray(
        [
            np.linalg.norm(
                embedded_modes[:, mode_index, None]
                * coefficients[mode_index][None, :],
                ord="fro",
            )
            ** 2
            for mode_index in range(eigenvalues.size)
        ]
    )
    total_power = max(float(np.sum(component_power)), np.finfo(float).eps)
    physical_modes = physical_hankel_modes(model, delay)
    rows: list[dict[str, float | int | bool | str]] = []
    objects: list[dict[str, object]] = []

    for mode_index, (eigenvalue, rate) in enumerate(
        zip(eigenvalues, continuous_rates, strict=True)
    ):
        signed_frequency = float(np.imag(rate) / (2 * np.pi))
        if signed_frequency <= 1e-8:
            continue
        conjugate_index = int(
            np.argmin(np.abs(eigenvalues - np.conj(eigenvalue)))
        )
        conjugate_found = bool(
            conjugate_index != mode_index
            and np.isclose(
                eigenvalues[conjugate_index],
                np.conj(eigenvalue),
                rtol=1e-6,
                atol=1e-8,
            )
        )
        frequency = signed_frequency
        evidence_seconds = (window_bins - delay) * DT_SECONDS
        cycles = frequency * evidence_seconds
        log_cycle_multiplier = float(np.real(rate) / frequency)
        amplitude_multiplier_per_cycle = float(
            np.exp(np.clip(log_cycle_multiplier, -700, 700))
        )
        pair_contribution = float(
            (
                component_power[mode_index]
                + (
                    component_power[conjugate_index]
                    if conjugate_found
                    else 0.0
                )
            )
            / total_power
        )
        resolved = bool(cycles >= MIN_MODE_CYCLES)
        salient = bool(pair_contribution >= MIN_MODE_PAIR_CONTRIBUTION)
        persistent_enough = bool(
            amplitude_multiplier_per_cycle
            >= MIN_AMPLITUDE_MULTIPLIER_PER_CYCLE
        )
        eligible_pair = bool(
            conjugate_found and resolved and salient and persistent_enough
        )
        node_id = f"{origin}:{mode_index}"
        row = {
            "window_nominal_seconds": window_seconds,
            "window_bins": window_bins,
            "delay": delay,
            "rank": rank,
            "origin_bin": origin,
            "origin_seconds": origin * DT_SECONDS,
            "node_id": node_id,
            "positive_mode_index": mode_index,
            "conjugate_mode_index": conjugate_index,
            "conjugate_found": conjugate_found,
            "eigenvalue_real": float(np.real(eigenvalue)),
            "eigenvalue_imag": float(np.imag(eigenvalue)),
            "eigenvalue_magnitude": float(abs(eigenvalue)),
            "growth_rate_per_second": float(np.real(rate)),
            "frequency_hz": frequency,
            "evidence_seconds": evidence_seconds,
            "cycles_in_evidence_span": float(cycles),
            "pair_fit_proxy_fraction": pair_contribution,
            "log_amplitude_multiplier_per_cycle": log_cycle_multiplier,
            "amplitude_multiplier_per_cycle": amplitude_multiplier_per_cycle,
            "three_cycle_resolved": resolved,
            "one_percent_salient": salient,
            "cycle_persistence_pass": persistent_enough,
            "eligible_mode_pair": eligible_pair,
        }
        rows.append(row)
        if eligible_pair:
            spatial_mode = physical_modes[:, mode_index]
            plane = scipy.linalg.orth(
                np.column_stack([spatial_mode.real, spatial_mode.imag])
            )
            objects.append(
                {
                    **row,
                    "plane": plane,
                }
            )
    return rows, objects


def fit_overlap_fraction(
    left_origin: int,
    right_origin: int,
    window_bins: int,
) -> float:
    """Fraction of fitting samples shared by two same-length windows."""
    return max(0.0, window_bins - abs(right_origin - left_origin)) / window_bins


def match_mode_pairs(
    origin_modes: dict[int, list[dict[str, object]]],
    window_seconds: int,
    window_bins: int,
    delay: int,
    rank: int,
) -> tuple[list[dict[str, float | int | bool | str]], set[frozenset[str]]]:
    """Hungarian-match eligible pairs for every pair of development origins."""
    match_rows: list[dict[str, float | int | bool | str]] = []
    accepted_edges: set[frozenset[str]] = set()
    for left_origin, right_origin in combinations(DEVELOPMENT_ORIGINS, 2):
        left_modes = origin_modes[left_origin]
        right_modes = origin_modes[right_origin]
        if not left_modes or not right_modes:
            continue
        cost = np.full((len(left_modes), len(right_modes)), 1e6, dtype=float)
        metrics: dict[tuple[int, int], tuple[float, float, float, bool]] = {}
        for left_index, left_mode in enumerate(left_modes):
            for right_index, right_mode in enumerate(right_modes):
                frequency_bin_distance = abs(
                    float(left_mode["frequency_hz"])
                    - float(right_mode["frequency_hz"])
                ) * (window_bins - delay) * DT_SECONDS
                plane_affinity = subspace_affinity(
                    np.asarray(left_mode["plane"]),
                    np.asarray(right_mode["plane"]),
                )
                log_cycle_multiplier_difference = abs(
                    float(left_mode["log_amplitude_multiplier_per_cycle"])
                    - float(right_mode["log_amplitude_multiplier_per_cycle"])
                )
                valid = bool(
                    frequency_bin_distance <= MAX_FREQUENCY_BIN_DISTANCE
                    and plane_affinity >= MIN_MODE_PLANE_AFFINITY
                    and log_cycle_multiplier_difference
                    <= MAX_LOG_CYCLE_MULTIPLIER_DIFFERENCE
                )
                metrics[(left_index, right_index)] = (
                    frequency_bin_distance,
                    plane_affinity,
                    log_cycle_multiplier_difference,
                    valid,
                )
                if valid:
                    cost[left_index, right_index] = (
                        frequency_bin_distance
                        + (1 - plane_affinity) / (1 - MIN_MODE_PLANE_AFFINITY)
                        + log_cycle_multiplier_difference
                    )
        left_assignment, right_assignment = linear_sum_assignment(cost)
        overlap_fraction = fit_overlap_fraction(
            left_origin, right_origin, window_bins
        )
        for left_index, right_index in zip(
            left_assignment, right_assignment, strict=True
        ):
            left_mode = left_modes[left_index]
            right_mode = right_modes[right_index]
            (
                frequency_bin_distance,
                plane_affinity,
                log_cycle_multiplier_difference,
                valid,
            ) = metrics[
                (left_index, right_index)
            ]
            accepted = bool(valid and cost[left_index, right_index] < 1e6)
            left_node = str(left_mode["node_id"])
            right_node = str(right_mode["node_id"])
            if accepted:
                accepted_edges.add(frozenset((left_node, right_node)))
            match_rows.append(
                {
                    "window_nominal_seconds": window_seconds,
                    "window_bins": window_bins,
                    "delay": delay,
                    "rank": rank,
                    "left_origin_bin": left_origin,
                    "right_origin_bin": right_origin,
                    "left_node_id": left_node,
                    "right_node_id": right_node,
                    "left_frequency_hz": float(left_mode["frequency_hz"]),
                    "right_frequency_hz": float(right_mode["frequency_hz"]),
                    "frequency_bin_distance": frequency_bin_distance,
                    "spatial_plane_affinity": plane_affinity,
                    "log_cycle_multiplier_difference": (
                        log_cycle_multiplier_difference
                    ),
                    "growth_rate_difference_per_second": abs(
                        float(left_mode["growth_rate_per_second"])
                        - float(right_mode["growth_rate_per_second"])
                    ),
                    "fit_overlap_fraction": overlap_fraction,
                    "low_overlap_pair": bool(
                        overlap_fraction <= MAX_LOW_OVERLAP_FRACTION
                    ),
                    "accepted_match": accepted,
                }
            )
    return match_rows, accepted_edges


def summarize_mode_tracks(
    origin_modes: dict[int, list[dict[str, object]]],
    accepted_edges: set[frozenset[str]],
    window_seconds: int,
    window_bins: int,
    delay: int,
    rank: int,
) -> list[dict[str, float | int | bool]]:
    """Build graph components and test recurrence using low-overlap edges."""
    node_lookup = {
        str(mode["node_id"]): mode
        for modes in origin_modes.values()
        for mode in modes
    }
    parent = {node_id: node_id for node_id in node_lookup}

    def find(node_id: str) -> str:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(left_node: str, right_node: str) -> None:
        left_root = find(left_node)
        right_root = find(right_node)
        if left_root != right_root:
            parent[right_root] = left_root

    for edge in accepted_edges:
        left_node, right_node = tuple(edge)
        union(left_node, right_node)

    components: dict[str, list[str]] = {}
    for node_id in node_lookup:
        components.setdefault(find(node_id), []).append(node_id)

    track_rows: list[dict[str, float | int | bool]] = []
    for track_index, component_nodes in enumerate(components.values(), start=1):
        modes = [node_lookup[node_id] for node_id in component_nodes]
        origins = [int(mode["origin_bin"]) for mode in modes]
        unique_origins = sorted(set(origins))
        ambiguous = len(origins) != len(unique_origins)
        node_by_origin = {
            int(mode["origin_bin"]): str(mode["node_id"]) for mode in modes
        }
        low_overlap_possible = 0
        low_overlap_accepted = 0
        if not ambiguous:
            for left_origin, right_origin in combinations(unique_origins, 2):
                if (
                    fit_overlap_fraction(left_origin, right_origin, window_bins)
                    <= MAX_LOW_OVERLAP_FRACTION
                ):
                    low_overlap_possible += 1
                    edge = frozenset(
                        (node_by_origin[left_origin], node_by_origin[right_origin])
                    )
                    low_overlap_accepted += int(edge in accepted_edges)
        low_overlap_fraction = (
            low_overlap_accepted / low_overlap_possible
            if low_overlap_possible
            else 0.0
        )
        coverage = len(unique_origins) / len(DEVELOPMENT_ORIGINS)
        track_pass = bool(
            not ambiguous
            and coverage >= MIN_MODE_TRACK_COVERAGE
            and low_overlap_possible >= MIN_LOW_OVERLAP_PAIRS
            and low_overlap_fraction >= MIN_LOW_OVERLAP_MATCH_FRACTION
        )
        frequencies = np.asarray([float(mode["frequency_hz"]) for mode in modes])
        growth_rates = np.asarray(
            [float(mode["growth_rate_per_second"]) for mode in modes]
        )
        contributions = np.asarray(
            [float(mode["pair_fit_proxy_fraction"]) for mode in modes]
        )
        cycle_multipliers = np.asarray(
            [float(mode["amplitude_multiplier_per_cycle"]) for mode in modes]
        )
        track_rows.append(
            {
                "window_nominal_seconds": window_seconds,
                "window_bins": window_bins,
                "delay": delay,
                "rank": rank,
                "track_id": track_index,
                "mode_nodes": len(component_nodes),
                "origin_count": len(unique_origins),
                "coverage_fraction": coverage,
                "ambiguous_multiple_modes_per_origin": ambiguous,
                "median_frequency_hz": float(np.median(frequencies)),
                "frequency_range_hz": float(np.ptp(frequencies)),
                "median_growth_rate_per_second": float(np.median(growth_rates)),
                "growth_rate_iqr_per_second": float(
                    np.subtract(*np.percentile(growth_rates, [75, 25]))
                ),
                "median_pair_fit_proxy_fraction": float(
                    np.median(contributions)
                ),
                "median_amplitude_multiplier_per_cycle": float(
                    np.median(cycle_multipliers)
                ),
                "low_overlap_pairs_possible": low_overlap_possible,
                "low_overlap_pairs_accepted": low_overlap_accepted,
                "low_overlap_match_fraction": low_overlap_fraction,
                "track_pass": track_pass,
            }
        )
    return track_rows


modal_mode_rows: list[dict[str, float | int | bool | str]] = []
modal_match_rows: list[dict[str, float | int | bool | str]] = []
modal_track_rows: list[dict[str, float | int | bool]] = []
modal_summary_rows: list[dict[str, float | int | bool | str]] = []

for candidate in spectral_shortlist.itertuples(index=False):
    window_seconds = int(candidate.window_nominal_seconds)
    window_bins = int(candidate.window_bins)
    delay = int(candidate.delay)
    rank = int(candidate.rank)
    origin_modes: dict[int, list[dict[str, object]]] = {}
    for origin in DEVELOPMENT_ORIGINS:
        _, _, model, _ = fit_hankel_at_origin(
            window_bins,
            delay,
            rank,
            origin,
        )
        rows, objects = oscillatory_mode_pairs(
            model,
            standardized_origin(window_bins, origin).latent_context,
            window_seconds,
            window_bins,
            delay,
            rank,
            origin,
        )
        modal_mode_rows.extend(rows)
        origin_modes[origin] = objects

    match_rows, accepted_edges = match_mode_pairs(
        origin_modes,
        window_seconds,
        window_bins,
        delay,
        rank,
    )
    track_rows = summarize_mode_tracks(
        origin_modes,
        accepted_edges,
        window_seconds,
        window_bins,
        delay,
        rank,
    )
    modal_match_rows.extend(match_rows)
    modal_track_rows.extend(track_rows)
    eligible_origin_fraction = float(
        np.mean([bool(origin_modes[origin]) for origin in DEVELOPMENT_ORIGINS])
    )
    low_overlap_geometry_pairs = sum(
        fit_overlap_fraction(left, right, window_bins)
        <= MAX_LOW_OVERLAP_FRACTION
        for left, right in combinations(DEVELOPMENT_ORIGINS, 2)
    )
    passing_tracks = [row for row in track_rows if bool(row["track_pass"])]
    best_track = max(
        track_rows,
        key=lambda row: (
            bool(row["track_pass"]),
            float(row["coverage_fraction"]),
            float(row["low_overlap_match_fraction"]),
        ),
        default=None,
    )
    mode_recurrence_pass = bool(passing_tracks)
    if low_overlap_geometry_pairs < MIN_LOW_OVERLAP_PAIRS:
        status = "inconclusive_geometry"
    elif eligible_origin_fraction == 0:
        status = "no_pair_passing_resolution_salience_damping"
    elif mode_recurrence_pass:
        status = "pass"
    else:
        status = "no_recurrent_track"
    modal_summary_rows.append(
        {
            "window_nominal_seconds": window_seconds,
            "window_bins": window_bins,
            "delay": delay,
            "rank": rank,
            "development_r2": float(candidate.development_r2),
            "local_mean_development_r2": float(
                candidate.local_mean_development_r2
            ),
            "primary_loss_mean": float(candidate.primary_loss_mean),
            "development_gate_pass": bool(candidate.development_gate_pass),
            "eligible_origin_fraction": eligible_origin_fraction,
            "adjacent_fit_overlap_fraction": fit_overlap_fraction(
                DEVELOPMENT_ORIGINS[0], DEVELOPMENT_ORIGINS[1], window_bins
            ),
            "low_overlap_geometry_pairs": low_overlap_geometry_pairs,
            "track_count": len(track_rows),
            "best_track_coverage_fraction": (
                float(best_track["coverage_fraction"])
                if best_track is not None
                else 0.0
            ),
            "best_track_low_overlap_pairs_possible": (
                int(best_track["low_overlap_pairs_possible"])
                if best_track is not None
                else 0
            ),
            "best_track_low_overlap_match_fraction": (
                float(best_track["low_overlap_match_fraction"])
                if best_track is not None
                else 0.0
            ),
            "mode_recurrence_pass": mode_recurrence_pass,
            "status": status,
        }
    )

modal_mode_table = pd.DataFrame(modal_mode_rows)
modal_match_table = pd.DataFrame(modal_match_rows)
modal_track_table = pd.DataFrame(modal_track_rows)
modal_summary_table = pd.DataFrame(modal_summary_rows)
modal_mode_table.to_csv(OUTPUT_DIR / "modal_shortlist_origin_modes.csv", index=False)
modal_match_table.to_csv(OUTPUT_DIR / "modal_shortlist_pair_matches.csv", index=False)
modal_track_table.to_csv(OUTPUT_DIR / "modal_shortlist_tracks.csv", index=False)
modal_summary_table.to_csv(OUTPUT_DIR / "modal_shortlist_summary.csv", index=False)

spectral_qualified = modal_summary_table.loc[
    modal_summary_table["mode_recurrence_pass"]
    & modal_summary_table["development_gate_pass"]
].sort_values(
    ["primary_loss_mean", "window_nominal_seconds", "rank", "delay"],
    kind="stable",
)
spectral_finalist_available = not spectral_qualified.empty
if spectral_finalist_available:
    spectral_finalist = spectral_qualified.iloc[0]
    final_window_seconds = int(spectral_finalist["window_nominal_seconds"])
    final_window_bins = int(spectral_finalist["window_bins"])
    final_delay = int(spectral_finalist["delay"])
    final_rank = int(spectral_finalist["rank"])
    final_development_common_r2 = float(spectral_finalist["development_r2"])
    final_development_local_r2 = float(
        spectral_finalist["local_mean_development_r2"]
    )
    final_technical_pass = bool(spectral_finalist["development_gate_pass"])
    mode_recurrence_pass = True
else:
    mode_recurrence_pass = False

# The internal residual diagnostic uses the spectrally qualified finalist when
# one exists. Otherwise it audits the already selected prediction candidate;
# it does not choose a different W,d,r because that candidate happened to have
# a favorable mode or residual. This reduces, but does not eliminate,
# post-selection inference because the prediction candidate is data-selected.
if spectral_finalist_available:
    residual_candidate = spectral_finalist
else:
    residual_candidate = modal_summary_table.loc[
        (modal_summary_table["window_bins"] == selected_window_bins)
        & (modal_summary_table["delay"] == selected_delay)
        & (modal_summary_table["rank"] == selected_rank)
    ].iloc[0]
residual_window_seconds = int(residual_candidate["window_nominal_seconds"])
residual_window_bins = int(residual_candidate["window_bins"])
residual_delay = int(residual_candidate["delay"])
residual_rank = int(residual_candidate["rank"])

print("EXACT-HANKEL MODE-RECURRENCE SHORTLIST")
print(
    modal_summary_table[
        [
            "window_nominal_seconds",
            "delay",
            "rank",
            "development_r2",
            "eligible_origin_fraction",
            "low_overlap_geometry_pairs",
            "best_track_coverage_fraction",
            "best_track_low_overlap_match_fraction",
            "status",
        ]
    ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
)
print(f"  spectrally qualified prediction finalist: {spectral_finalist_available}")
print(
    "  development-selected internal residual-audit candidate: "
    f"W={residual_window_seconds}s, d={residual_delay}, r={residual_rank}"
)


# %% Step 9b — Internal chronological G,A,L diagnostic
#
# PyDMD does not implement a Residual-DMD fitter (its RDMD class is Randomized
# DMD). The calculation below therefore refits a separate rank-r EDMD
# dictionary on the first part of each context and evaluates its G,A,L form on
# an internal chronological tail. That tail was already part of development
# model selection, so it is an internal split diagnostic—not held-out
# confirmation and not a residual attached to the full-window PyDMD modes.
# For this stochastic trajectory the statistic contains dynamical variance as
# well as projection error. A fixed-POD-coordinate circular-shift null retains
# each coordinate's circular autocorrelation while destroying cross-coordinate
# timing. One global median statistic avoids selecting among eight origin-level
# p-values. However, W,d,r is chosen from development outcomes, so these
# p-values remain exploratory and are not adjusted for candidate selection.


def resdmd_matrices(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Form uniformly weighted G, A, L for coordinate x/y pair columns."""
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("x and y must be aligned coordinate-by-pair matrices")
    pair_count = x.shape[1]
    g = (x.conj() @ x.T) / pair_count
    a = (x.conj() @ y.T) / pair_count
    l_matrix = (y.conj() @ y.T) / pair_count
    return (g + g.conj().T) / 2, a, (l_matrix + l_matrix.conj().T) / 2


def resdmd_residuals(
    eigenvalues: np.ndarray,
    eigenfunctions: np.ndarray,
    g: np.ndarray,
    a: np.ndarray,
    l_matrix: np.ndarray,
) -> np.ndarray:
    """Evaluate the published relative ResDMD residual quadratic form."""
    residuals = []
    for eigenvalue, vector in zip(
        eigenvalues, eigenfunctions.T, strict=True
    ):
        form = (
            l_matrix
            - eigenvalue * a.conj().T
            - np.conj(eigenvalue) * a
            + abs(eigenvalue) ** 2 * g
        )
        form = (form + form.conj().T) / 2
        numerator = float(np.real(vector.conj() @ form @ vector))
        denominator = float(np.real(vector.conj() @ g @ vector))
        if denominator <= 100 * np.finfo(float).eps:
            residuals.append(np.nan)
        else:
            residuals.append(np.sqrt(max(0.0, numerator) / denominator))
    return np.asarray(residuals)


@dataclass(frozen=True)
class ResDMDDiagnostic:
    training_residuals: np.ndarray
    quadrature_residuals: np.ndarray
    eigenvalues: np.ndarray
    fit_bins: int
    quadrature_bins: int


def chronological_resdmd_diagnostic(
    uncentered_latent_context: np.ndarray,
    delay: int,
    rank: int,
) -> ResDMDDiagnostic:
    """Fit-center candidates first, then score G,A,L on an internal tail."""
    total_bins = uncentered_latent_context.shape[1]
    quadrature_bins = max(delay + rank + 2, int(np.ceil(0.25 * total_bins)))
    fit_bins = total_bins - quadrature_bins
    if fit_bins - delay < rank or quadrature_bins - delay < rank:
        raise ValueError("Not enough independent chronological pairs for ResDMD")

    # The quadrature tail must not influence candidate centering. The mean is
    # learned from the candidate-fit prefix and applied unchanged to both
    # blocks. Projection has already been frozen on acquisition 1.
    fit_mean = np.mean(
        uncentered_latent_context[:, :fit_bins], axis=1, keepdims=True
    )
    centered_latent_context = uncentered_latent_context - fit_mean

    fit_hankel = pseudo_hankel_matrix(
        centered_latent_context[:, :fit_bins], delay
    )
    quadrature_hankel = pseudo_hankel_matrix(
        centered_latent_context[:, fit_bins:], delay
    )
    fit_x, fit_y = fit_hankel[:, :-1], fit_hankel[:, 1:]
    quadrature_x = quadrature_hankel[:, :-1]
    quadrature_y = quadrature_hankel[:, 1:]

    # The dictionary is the rank-r POD coordinate basis learned only from the
    # candidate-fit pairs. The quadrature block never changes this basis.
    try:
        dictionary_basis, dictionary_singular_values, _ = svds(
            fit_x,
            k=rank,
            which="LM",
            solver="propack",
            maxiter=200,
            tol=1e-10,
            rng=np.random.default_rng(SEED + 4),
        )
        dictionary_basis = dictionary_basis[
            :, np.argsort(dictionary_singular_values)[::-1]
        ]
    except np.linalg.LinAlgError:
        dictionary_basis = np.linalg.svd(
            fit_x,
            full_matrices=False,
        )[0][:, :rank]
    fit_x_reduced = dictionary_basis.conj().T @ fit_x
    fit_y_reduced = dictionary_basis.conj().T @ fit_y
    quadrature_x_reduced = dictionary_basis.conj().T @ quadrature_x
    quadrature_y_reduced = dictionary_basis.conj().T @ quadrature_y

    g_fit, a_fit, l_fit = resdmd_matrices(fit_x_reduced, fit_y_reduced)
    g_quad, a_quad, l_quad = resdmd_matrices(
        quadrature_x_reduced, quadrature_y_reduced
    )
    eigenvalues, eigenfunctions = scipy.linalg.eig(a_fit, g_fit)
    for mode_index in range(eigenfunctions.shape[1]):
        vector = eigenfunctions[:, mode_index]
        norm = float(np.sqrt(np.real(vector.conj() @ g_fit @ vector)))
        if norm > 0:
            eigenfunctions[:, mode_index] = vector / norm
    return ResDMDDiagnostic(
        training_residuals=resdmd_residuals(
            eigenvalues, eigenfunctions, g_fit, a_fit, l_fit
        ),
        quadrature_residuals=resdmd_residuals(
            eigenvalues, eigenfunctions, g_quad, a_quad, l_quad
        ),
        eigenvalues=eigenvalues,
        fit_bins=fit_bins,
        quadrature_bins=quadrature_bins,
    )


# Independent random pairs verify the G,A,L orientation and residual formula
# before those helpers are applied to the neural trajectory.
residual_unit_rng = np.random.default_rng(SEED + 2)
residual_unit_x = residual_unit_rng.standard_normal((2, 900))
residual_unit_operator = 0.93 * np.array(
    [[np.cos(0.27), -np.sin(0.27)], [np.sin(0.27), np.cos(0.27)]]
)
residual_unit_y = residual_unit_operator @ residual_unit_x
unit_g, unit_a, unit_l = resdmd_matrices(
    residual_unit_x[:, :600], residual_unit_y[:, :600]
)
unit_g_test, unit_a_test, unit_l_test = resdmd_matrices(
    residual_unit_x[:, 600:], residual_unit_y[:, 600:]
)
unit_eigenvalues, unit_eigenfunctions = scipy.linalg.eig(unit_a, unit_g)
unit_residuals = resdmd_residuals(
    unit_eigenvalues,
    unit_eigenfunctions,
    unit_g_test,
    unit_a_test,
    unit_l_test,
)
unit_maximum_residual = float(np.nanmax(unit_residuals))
if unit_maximum_residual > 1e-7:
    raise RuntimeError("The exact ResDMD residual unit check failed")
print(
    "SUCCESS — independent-pair G,A,L unit check; maximum residual "
    f"{unit_maximum_residual:.3e}"
)


residual_rows: list[dict[str, float | int]] = []
surrogate_rows: list[dict[str, float | int]] = []
for origin in DEVELOPMENT_ORIGINS:
    uncentered_latent_context = target_latent[
        :, origin - residual_window_bins : origin
    ]
    observed_resdmd = chronological_resdmd_diagnostic(
        uncentered_latent_context,
        residual_delay,
        residual_rank,
    )
    observed_median = float(np.nanmedian(observed_resdmd.quadrature_residuals))
    residual_rows.append(
        {
            "origin_bin": origin,
            "training_residual_median": float(
                np.nanmedian(observed_resdmd.training_residuals)
            ),
            "quadrature_residual_median": observed_median,
            "candidate_fit_bins": observed_resdmd.fit_bins,
            "quadrature_bins": observed_resdmd.quadrature_bins,
        }
    )

# Generate each surrogate once over the full accessible development prefix,
# then slice every overlapping origin from that same surrogate. This preserves
# the overlap-induced dependence among origin statistics under the null. Each
# coordinate receives one independent circular shift; every sliced diagnostic
# still learns its own fit-prefix mean without quadrature-tail leakage.
surrogate_source_start = min(
    origin - residual_window_bins for origin in DEVELOPMENT_ORIGINS
)
surrogate_source_stop = max(DEVELOPMENT_ORIGINS)
surrogate_source = target_latent[:, surrogate_source_start:surrogate_source_stop]
surrogate_source_bins = surrogate_source.shape[1]
surrogate_time_index = np.arange(surrogate_source_bins)[None, :]
surrogate_coordinate_index = np.arange(surrogate_source.shape[0])[:, None]
for surrogate in range(N_RESDMD_SURROGATES):
    offsets = rng.integers(
        1,
        surrogate_source_bins,
        size=(surrogate_source.shape[0], 1),
    )
    shifted_index = (surrogate_time_index - offsets) % surrogate_source_bins
    shifted_source = surrogate_source[
        surrogate_coordinate_index, shifted_index
    ]
    for origin in DEVELOPMENT_ORIGINS:
        local_start = origin - residual_window_bins - surrogate_source_start
        local_stop = origin - surrogate_source_start
        shifted_latent = shifted_source[:, local_start:local_stop]
        surrogate_resdmd = chronological_resdmd_diagnostic(
            shifted_latent,
            residual_delay,
            residual_rank,
        )
        surrogate_rows.append(
            {
                "origin_bin": origin,
                "surrogate": surrogate,
                "full_prefix_shift": True,
                "quadrature_residual_median": float(
                    np.nanmedian(surrogate_resdmd.quadrature_residuals)
                ),
            }
        )

residual_table = pd.DataFrame(residual_rows)
surrogate_table = pd.DataFrame(surrogate_rows)

null_percentiles: list[dict[str, float | int | bool]] = []
for origin in DEVELOPMENT_ORIGINS:
    observed_value = float(
        residual_table.loc[
            residual_table["origin_bin"] == origin,
            "quadrature_residual_median",
        ].iloc[0]
    )
    null_values = surrogate_table.loc[
        surrogate_table["origin_bin"] == origin,
        "quadrature_residual_median",
    ].to_numpy()
    null_at_or_below = int(np.sum(null_values <= observed_value))
    monte_carlo_p_lower = (null_at_or_below + 1) / (null_values.size + 1)
    null_percentiles.append(
        {
            "origin_bin": origin,
            "observed_quadrature_residual": observed_value,
            "null_at_or_below_count": null_at_or_below,
            "n_surrogates": int(null_values.size),
            "monte_carlo_p_lower": monte_carlo_p_lower,
            "passes_nominal_0_05": bool(monte_carlo_p_lower <= 0.05),
        }
    )
null_percentile_table = pd.DataFrame(null_percentiles)
residual_table = residual_table.merge(
    null_percentile_table,
    on="origin_bin",
    how="left",
    validate="one_to_one",
)

residual_table.to_csv(OUTPUT_DIR / "development_resdmd_diagnostics.csv", index=False)
surrogate_table.to_csv(OUTPUT_DIR / "development_resdmd_surrogates.csv", index=False)

observed_global_residual = float(
    np.median(residual_table["quadrature_residual_median"])
)
global_null_values = (
    surrogate_table.pivot(
        index="surrogate",
        columns="origin_bin",
        values="quadrature_residual_median",
    )
    .median(axis=1)
    .to_numpy()
)
global_null_at_or_below = int(
    np.sum(global_null_values <= observed_global_residual)
)
global_residual_p_lower = (
    global_null_at_or_below + 1
) / (global_null_values.size + 1)
global_residual_pass = bool(global_residual_p_lower <= 0.05)
residual_inference_confirmatory = False
global_residual_table = pd.DataFrame(
    {
        "statistic": ["median across eight origins"],
        "observed_quadrature_residual": [observed_global_residual],
        "null_at_or_below_count": [global_null_at_or_below],
        "n_surrogates": [global_null_values.size],
        "monte_carlo_p_lower": [global_residual_p_lower],
        "passes_nominal_0_05": [global_residual_pass],
        "candidate_selection_adjusted": [False],
        "interpretation": [
            "exploratory internal development diagnostic; not selection-adjusted"
        ],
    }
)
global_residual_table.to_csv(
    OUTPUT_DIR / "development_resdmd_global_null.csv", index=False
)

audit_summary = modal_summary_table.loc[
    (modal_summary_table["window_bins"] == residual_window_bins)
    & (modal_summary_table["delay"] == residual_delay)
    & (modal_summary_table["rank"] == residual_rank)
].iloc[0]
eligible_mode_origin_fraction = float(audit_summary["eligible_origin_fraction"])
audit_matches = modal_match_table.loc[
    (modal_match_table["window_bins"] == residual_window_bins)
    & (modal_match_table["delay"] == residual_delay)
    & (modal_match_table["rank"] == residual_rank)
]
adjacent_matches = audit_matches.loc[
    (audit_matches["right_origin_bin"] - audit_matches["left_origin_bin"])
    == (DEVELOPMENT_ORIGINS[1] - DEVELOPMENT_ORIGINS[0])
]
median_affinity = (
    float(adjacent_matches["spatial_plane_affinity"].median())
    if not adjacent_matches.empty
    else np.nan
)
median_adjacent_overlap = fit_overlap_fraction(
    DEVELOPMENT_ORIGINS[0],
    DEVELOPMENT_ORIGINS[1],
    residual_window_bins,
)
residual_null_win_fraction = float(
    residual_table["passes_nominal_0_05"].mean()
)
modal_gate_pass = bool(
    mode_recurrence_pass
    and global_residual_pass
    and residual_inference_confirmatory
)

print("MODE-RECURRENCE / INTERNAL RESDMD-STYLE DEVELOPMENT CHECK")
print(
    "  audit-candidate origins with a resolution/salience/damping-qualified "
    f"pair: {eligible_mode_origin_fraction:.3f}"
)
print(
    "  median adjacent mode-plane affinity: "
    f"{median_affinity:.3f} (windows share {median_adjacent_overlap:.1%})"
)
print(
    "  origins with descriptive lower-tail p<=0.05: "
    f"{residual_null_win_fraction:.3f}"
)
print(f"  global median residual Monte Carlo p: {global_residual_p_lower:.4f}")
print("  selection-adjusted confirmatory inference: False")
print(f"  recurrent low-overlap mode track: {mode_recurrence_pass}")
print(f"  modal gate passes: {modal_gate_pass}")
print(
    "  caution: the G,A,L statistic is variance-containing and belongs to a "
    "separate tail-split EDMD diagnostic, not the displayed PyDMD modes."
)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
audit_modes = modal_mode_table.loc[
    (modal_mode_table["window_bins"] == residual_window_bins)
    & (modal_mode_table["delay"] == residual_delay)
    & (modal_mode_table["rank"] == residual_rank)
    & modal_mode_table["eligible_mode_pair"]
]
if audit_modes.empty:
    axes[0].text(
        0.5,
        0.5,
        "No pair passed all three gates",
        transform=axes[0].transAxes,
        ha="center",
        va="center",
        color="#D55E00",
    )
    axes[0].set_xlim(
        DEVELOPMENT_ORIGINS[0] * DT_SECONDS,
        DEVELOPMENT_ORIGINS[-1] * DT_SECONDS,
    )
    axes[0].set_ylim(0, 0.5 / DT_SECONDS)
else:
    axes[0].scatter(
        audit_modes["origin_seconds"],
        audit_modes["frequency_hz"],
        s=35,
        color="#0072B2",
        label="eligible conjugate pair",
    )
for match in audit_matches.loc[audit_matches["accepted_match"]].itertuples(
    index=False
):
    axes[0].plot(
        [match.left_origin_bin * DT_SECONDS, match.right_origin_bin * DT_SECONDS],
        [match.left_frequency_hz, match.right_frequency_hz],
        color="0.65",
        linewidth=0.7,
        alpha=0.5,
    )
axes[0].set(
    xlabel="development origin (s)",
    ylabel="frequency (Hz)",
    title="Resolution/salience/damping-qualified signatures",
)
if not audit_modes.empty:
    axes[0].legend(frameon=False, fontsize=7)

if not audit_matches.empty:
    axes[1].scatter(
        audit_matches["fit_overlap_fraction"],
        audit_matches["spatial_plane_affinity"],
        c=np.where(audit_matches["accepted_match"], "#0072B2", "0.7"),
        s=28,
    )
else:
    axes[1].text(
        0.5,
        0.5,
        "No eligible mode pairs to match",
        transform=axes[1].transAxes,
        ha="center",
        va="center",
        color="#D55E00",
    )
axes[1].axhline(MIN_MODE_PLANE_AFFINITY, color="#D55E00", linestyle="--")
axes[1].axvline(MAX_LOW_OVERLAP_FRACTION, color="#009E73", linestyle=":")
axes[1].set(
    ylim=(0, 1.02),
    xlabel="fit-window overlap fraction",
    ylabel="real mode-plane affinity",
    title="Low-overlap evidence is required",
)

surrogate_positions = np.arange(len(DEVELOPMENT_ORIGINS))
surrogate_groups = [
    surrogate_table.loc[
        surrogate_table["origin_bin"] == origin,
        "quadrature_residual_median",
    ].to_numpy()
    for origin in DEVELOPMENT_ORIGINS
]
axes[2].boxplot(
    surrogate_groups,
    positions=surrogate_positions,
    widths=0.55,
    showfliers=False,
)
axes[2].scatter(
    surrogate_positions,
    residual_table["quadrature_residual_median"],
    color="#D55E00",
    zorder=3,
    label="observed",
)
axes[2].set(
    xticks=surrogate_positions,
    xticklabels=[f"{origin * DT_SECONDS:.0f}" for origin in DEVELOPMENT_ORIGINS],
    xlabel="origin (s)",
    ylabel="median internal-tail variance-containing residual",
    title="Observed vs POD-coordinate-shift null",
)
axes[2].legend(frameon=False)
fig.suptitle(
    f"Step 9 — Spectral audit candidate W={residual_window_seconds}s, "
    f"d={residual_delay}, r={residual_rank}"
)
save_and_show(fig, "07_modal_stability_and_resdmd_null.png")

# Explicit-rank conditioning is disclosed for every distinct candidate that
# could be read as a finalist: the common-loss best, one-SE choice, final
# development candidate, and internal residual-audit candidate.
# Local centering makes the untruncated snapshot matrix rank-deficient whenever
# rows >= columns; PyDMD's large full condition-number warning is therefore
# expected. The retained condition number sigma_1/sigma_r is the relevant
# numerical diagnostic after explicit rank truncation.
conditioning_candidate_roles: dict[tuple[int, int, int], list[str]] = {}
for role, candidate_tuple in (
    (
        "common_loss_best",
        (
            int(selection_pool_best["window_bins"]),
            int(selection_pool_best["delay"]),
            int(selection_pool_best["rank"]),
        ),
    ),
    ("one_se_prediction", (selected_window_bins, selected_delay, selected_rank)),
    (
        "best_local_increment",
        (
            int(best_local_global.iloc[0]["window_bins"]),
            int(best_local_global.iloc[0]["delay"]),
            int(best_local_global.iloc[0]["rank"]),
        ),
    ),
    ("final_development", (final_window_bins, final_delay, final_rank)),
    (
        "internal_residual_audit",
        (residual_window_bins, residual_delay, residual_rank),
    ),
):
    conditioning_candidate_roles.setdefault(candidate_tuple, []).append(role)

conditioning_rows: list[dict[str, float | int | str]] = []
for (candidate_window_bins, candidate_delay, candidate_rank), roles in (
    conditioning_candidate_roles.items()
):
    candidate_window_seconds = next(
        seconds
        for seconds, bins in WINDOW_BINS.items()
        if bins == candidate_window_bins
    )
    for origin in DEVELOPMENT_ORIGINS:
        latent_context = standardized_origin(
            candidate_window_bins, origin
        ).latent_context
        full_hankel_input = pseudo_hankel_matrix(
            latent_context, candidate_delay
        )
        snapshot_pairs = full_hankel_input[:, :-1]
        singular_values = scipy.linalg.svdvals(snapshot_pairs)
        full_input_singular_values = scipy.linalg.svdvals(full_hankel_input)
        retained_condition_number = float(
            singular_values[0] / singular_values[candidate_rank - 1]
        )
        rank_boundary_ratio = (
            float(
                singular_values[candidate_rank - 1]
                / singular_values[candidate_rank]
            )
            if candidate_rank < singular_values.size
            and singular_values[candidate_rank] > np.finfo(float).eps
            else np.inf
        )
        operator_pair_condition_number = (
            float(singular_values[0] / singular_values[-1])
            if singular_values[-1] > np.finfo(float).eps
            else np.inf
        )
        centered_input_condition_number = (
            float(full_input_singular_values[0] / full_input_singular_values[-1])
            if full_input_singular_values[-1] > np.finfo(float).eps
            else np.inf
        )
        conditioning_rows.append(
            {
                "candidate_roles": "+".join(roles),
                "origin_bin": origin,
                "window_nominal_seconds": candidate_window_seconds,
                "window_bins": candidate_window_bins,
                "delay": candidate_delay,
                "rank": candidate_rank,
                "retained_condition_number_sigma1_over_sigmar": (
                    retained_condition_number
                ),
                "rank_boundary_ratio_sigmar_over_sigma_next": rank_boundary_ratio,
                "operator_pair_full_condition_number": (
                    operator_pair_condition_number
                ),
                "centered_input_full_condition_number": (
                    centered_input_condition_number
                ),
            }
        )
conditioning_table = pd.DataFrame(conditioning_rows)
conditioning_table.to_csv(
    OUTPUT_DIR / "finalist_conditioning.csv", index=False
)
candidate_warnings = warning_table.loc[
    (warning_table["window_bins"] == residual_window_bins)
    & (warning_table["delay"] == residual_delay)
    & (warning_table["rank"] == residual_rank)
]
print("EXPLICIT-RANK CONDITIONING AUDIT")
print(
    conditioning_table.groupby(
        ["candidate_roles", "window_nominal_seconds", "delay", "rank"],
        as_index=False,
    )["retained_condition_number_sigma1_over_sigmar"]
    .median()
    .to_string(index=False, float_format=lambda value: f"{value:.3g}")
)
print(
    "  unique PyDMD warnings for this candidate: "
    f"{candidate_warnings['warning_message'].nunique()} "
    f"({int(candidate_warnings['unexpected_warning'].sum())} unexpected records)"
)

# Show the all-neuron spatial weights of the final development-origin modes.
with h5py.File(DATA_PATH, "r") as mat:
    centroid_raw = np.asarray(mat["ROIs/Centroid"], dtype=np.float64)
if centroid_raw.shape == (2, n_neurons):
    centroid_px = centroid_raw.T
elif centroid_raw.shape == (n_neurons, 2):
    centroid_px = centroid_raw
else:
    raise ValueError(f"Unexpected centroid shape: {centroid_raw.shape}")

_, _, spatial_model, _ = fit_hankel_at_origin(
    residual_window_bins,
    residual_delay,
    residual_rank,
    DEVELOPMENT_ORIGINS[-1],
)
spatial_modes_standardized = physical_hankel_modes(
    spatial_model, residual_delay
)
spatial_modes_rate = scale[:, None] * spatial_modes_standardized
spatial_rates = np.log(
    np.asarray(spatial_model.eigs, dtype=np.complex128)
) / DT_SECONDS
representative_mode_indices = [
    mode_index
    for mode_index, rate in enumerate(spatial_rates)
    if np.imag(rate) > 1e-8 or abs(np.imag(rate)) <= 1e-8
]
n_spatial_panels = min(4, len(representative_mode_indices))
fig, axes = plt.subplots(
    1,
    n_spatial_panels,
    figsize=(4.2 * n_spatial_panels, 4),
    constrained_layout=True,
)
axes_array = np.atleast_1d(axes)
for panel_index, axis in enumerate(axes_array):
    mode_index = representative_mode_indices[panel_index]
    mode = spatial_modes_rate[:, mode_index].copy()
    pivot = int(np.argmax(np.abs(mode)))
    mode *= np.exp(-1j * np.angle(mode[pivot]))
    signed_weight = mode.real
    color_limit = float(np.quantile(np.abs(signed_weight), 0.99))
    scatter = axis.scatter(
        centroid_px[eligible_rows, 0],
        centroid_px[eligible_rows, 1],
        c=signed_weight,
        s=4,
        cmap="coolwarm",
        vmin=-color_limit,
        vmax=color_limit,
        linewidths=0,
    )
    axis.set(
        xlabel="x (pixels)",
        ylabel="y (pixels)",
        title=(
            f"mode {mode_index + 1}: "
            f"f={abs(np.imag(spatial_rates[mode_index])) / (2 * np.pi):.3f} Hz\n"
            f"growth={np.real(spatial_rates[mode_index]):.3f} /s"
        ),
        aspect="equal",
    )
    axis.invert_yaxis()
    fig.colorbar(scatter, ax=axis, shrink=0.75, label="phase-aligned rate weight")
fig.suptitle(
    "Step 9 — All-neuron audit-candidate modes at the last development origin; "
    "no neuron sampling"
)
save_and_show(fig, "08_audit_candidate_spatial_modes.png")


# %% Step 9c — Descriptive training reconstruction and spectral adequacy
#
# Forecasting remains the selection criterion. Still, a claimed mode family
# should at least describe the transformed activity it was fitted to. For the
# common-loss best, one-SE candidate, and best local-increment candidate, we
# therefore compute (i) in-sample reconstruction R² in the exact all-neuron
# norm, including POD-discarded variance, and (ii) overlap between normalized
# observed and reconstructed aggregate POD power spectra. These are optimistic
# descriptive diagnostics because the same window fits and scores amplitudes.

reconstruction_candidate_roles: dict[tuple[int, int, int], list[str]] = {}
for role, candidate_tuple in (
    (
        "common_loss_best",
        (
            int(selection_pool_best["window_bins"]),
            int(selection_pool_best["delay"]),
            int(selection_pool_best["rank"]),
        ),
    ),
    ("one_se_prediction", (selected_window_bins, selected_delay, selected_rank)),
    (
        "best_local_increment",
        (
            int(best_local_global.iloc[0]["window_bins"]),
            int(best_local_global.iloc[0]["delay"]),
            int(best_local_global.iloc[0]["rank"]),
        ),
    ),
):
    reconstruction_candidate_roles.setdefault(candidate_tuple, []).append(role)

reconstruction_rows: list[dict[str, float | int | str]] = []
for (candidate_window_bins, candidate_delay, candidate_rank), roles in (
    reconstruction_candidate_roles.items()
):
    candidate_window_seconds = next(
        seconds
        for seconds, bins in WINDOW_BINS.items()
        if bins == candidate_window_bins
    )
    for origin in DEVELOPMENT_ORIGINS:
        _, _, reconstruction_model, _ = fit_hankel_at_origin(
            candidate_window_bins,
            candidate_delay,
            candidate_rank,
            origin,
        )
        prepared = standardized_origin(candidate_window_bins, origin)
        latent_reconstruction_complex = np.asarray(
            reconstruction_model.reconstructed_data
        )
        if latent_reconstruction_complex.shape != prepared.latent_context.shape:
            raise RuntimeError("Unexpected HankelDMD reconstruction shape")
        reconstruction_imaginary_leakage = relative_imaginary_leakage(
            latent_reconstruction_complex
        )
        latent_reconstruction = np.real(latent_reconstruction_complex)
        full_context = target_standardized[
            :, origin - candidate_window_bins : origin
        ]
        full_context_centered = full_context - np.mean(
            full_context, axis=1, keepdims=True
        )
        total_centered_energy = float(np.sum(full_context_centered**2))
        retained_observed_energy = float(np.sum(prepared.latent_context**2))
        discarded_energy = max(
            0.0, total_centered_energy - retained_observed_energy
        )
        reconstruction_sse = discarded_energy + float(
            np.sum((prepared.latent_context - latent_reconstruction) ** 2)
        )
        reconstruction_r2 = 1 - reconstruction_sse / total_centered_energy

        observed_power = np.sum(
            np.abs(np.fft.rfft(prepared.latent_context, axis=1)) ** 2,
            axis=0,
        )[1:]
        reconstructed_power = np.sum(
            np.abs(np.fft.rfft(latent_reconstruction, axis=1)) ** 2,
            axis=0,
        )[1:]
        frequency_axis = np.fft.rfftfreq(
            candidate_window_bins, d=DT_SECONDS
        )[1:]
        observed_power /= max(float(np.sum(observed_power)), np.finfo(float).eps)
        reconstructed_power /= max(
            float(np.sum(reconstructed_power)), np.finfo(float).eps
        )
        spectral_overlap = float(
            np.sum(np.minimum(observed_power, reconstructed_power))
        )
        reconstruction_rows.append(
            {
                "candidate_roles": "+".join(roles),
                "window_nominal_seconds": candidate_window_seconds,
                "window_bins": candidate_window_bins,
                "delay": candidate_delay,
                "rank": candidate_rank,
                "origin_bin": origin,
                "all_neuron_training_reconstruction_r2": reconstruction_r2,
                "normalized_power_spectral_overlap": spectral_overlap,
                "observed_peak_frequency_hz": float(
                    frequency_axis[int(np.argmax(observed_power))]
                ),
                "reconstructed_peak_frequency_hz": float(
                    frequency_axis[int(np.argmax(reconstructed_power))]
                ),
                "reconstruction_imaginary_leakage": (
                    reconstruction_imaginary_leakage
                ),
            }
        )

reconstruction_table = pd.DataFrame(reconstruction_rows)
reconstruction_table.to_csv(
    OUTPUT_DIR / "training_reconstruction_spectral_audit.csv", index=False
)
reconstruction_summary = (
    reconstruction_table.groupby(
        [
            "candidate_roles",
            "window_nominal_seconds",
            "delay",
            "rank",
        ],
        as_index=False,
    )
    .agg(
        median_training_reconstruction_r2=(
            "all_neuron_training_reconstruction_r2",
            "median",
        ),
        median_power_spectral_overlap=(
            "normalized_power_spectral_overlap",
            "median",
        ),
    )
)
print("DESCRIPTIVE TRAINING RECONSTRUCTION / SPECTRAL ADEQUACY")
print(
    reconstruction_summary.to_string(
        index=False, float_format=lambda value: f"{value:.4f}"
    )
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
reconstruction_labels = reconstruction_summary["candidate_roles"].to_list()
reconstruction_display_labels = {
    "best_local_increment": "best local\nincrement",
    "common_loss_best": "common-loss\nbest",
    "one_se_prediction": "one-SE fallback",
}
for axis, metric, ylabel, title in (
    (
        axes[0],
        "all_neuron_training_reconstruction_r2",
        "in-sample all-neuron R²",
        "Training reconstruction (optimistic)",
    ),
    (
        axes[1],
        "normalized_power_spectral_overlap",
        "normalized spectral overlap",
        "Aggregate POD power-spectrum recovery",
    ),
):
    groups = [
        reconstruction_table.loc[
            reconstruction_table["candidate_roles"] == label, metric
        ].to_numpy()
        for label in reconstruction_labels
    ]
    axis.boxplot(
        groups,
        tick_labels=[
            reconstruction_display_labels.get(label, label)
            for label in reconstruction_labels
        ],
        showfliers=False,
    )
    axis.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    axis.set(ylabel=ylabel, title=title)
fig.suptitle("Step 9 — Fitted-mode adequacy is separate from forecast validation")
save_and_show(fig, "08b_training_reconstruction_spectral_audit.png")


# %% Step 10 — Prespecified inner-product sensitivity: raw-centered rate
#
# Neuronwise RMS scaling gives low-rate and high-rate neurons similar weight.
# That is useful for an all-neuron tutorial but is not a universal physical
# inner product. We therefore repeat the *same* exact-Hankel W,d,r grid using
# raw event-rate units. The acquisition-1 mean is removed for POD training;
# each local fitting window is centered by its own past-only mean exactly as in
# the primary analysis. Results are not pooled across the two estimands.

raw_representation_centered = (
    representation_rate[eligible] - representation_mean[eligible, None]
)
raw_target_centered = (
    target_rate[eligible] - representation_mean[eligible, None]
)
raw_pod_basis, raw_pod_singular_values, _ = svds(
    raw_representation_centered,
    k=pod_dimension,
    which="LM",
    solver="propack",
    rng=np.random.default_rng(SEED + 1),
)
raw_order = np.argsort(raw_pod_singular_values)[::-1]
raw_pod_basis = raw_pod_basis[:, raw_order]
raw_pod_singular_values = raw_pod_singular_values[raw_order]
raw_pod_energy = float(
    np.sum(raw_pod_singular_values**2)
    / np.sum(raw_representation_centered**2)
)
raw_target_latent = raw_pod_basis.T @ raw_target_centered
raw_origin_cache: dict[tuple[int, int], PreparedOrigin] = {}


def fit_hankel_in_coordinates(
    target_values: np.ndarray,
    projected_values: np.ndarray,
    window_bins: int,
    delay: int,
    rank: int,
    origin: int,
) -> tuple[OriginForecast, list[dict[str, float | int]]]:
    """Apply the locked exact-Hankel estimator in another fixed inner product."""
    cache_key = (window_bins, origin)
    if cache_key not in raw_origin_cache:
        raw_origin_cache[cache_key] = prepare_origin(
            target_values,
            projected_values,
            window_bins,
            origin,
            np.zeros(target_values.shape[0], dtype=np.float64),
        )
    prepared = raw_origin_cache[cache_key]
    latent_context = prepared.latent_context
    model = HankelDMD(
        svd_rank=rank,
        tlsq_rank=0,
        exact=True,
        opt=False,
        forward_backward=False,
        d=delay,
        reconstruction_method="mean",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(latent_context)
    warning_records = unique_warning_records(caught)
    latent_prediction, imaginary_leakage = forecast_hankeldmd(
        model,
        latent_context,
        delay,
        FORECAST_HORIZON_BINS,
    )
    return score_latent_forecast(
        prepared,
        latent_prediction,
        imaginary_leakage,
        float(np.max(np.abs(model.eigs))),
        len(warning_records),
        sum(bool(record["unexpected_warning"]) for record in warning_records),
    )


raw_rows: list[dict[str, float | int]] = []
raw_horizon_rows: list[dict[str, float | int]] = []
for configuration in tqdm(
    list(allowed_grid.itertuples(index=False)),
    desc="raw-rate sensitivity",
    disable=not sys.stderr.isatty(),
):
    key = {
        "window_nominal_seconds": int(configuration.window_nominal_seconds),
        "window_bins": int(configuration.window_bins),
        "window_actual_seconds": float(configuration.window_actual_seconds),
        "delay": int(configuration.delay),
        "history_seconds": float(configuration.history_seconds),
        "rank": int(configuration.rank),
    }
    for origin in DEVELOPMENT_ORIGINS:
        raw_score, raw_horizon_scores = fit_hankel_in_coordinates(
            raw_target_centered,
            raw_target_latent,
            int(configuration.window_bins),
            int(configuration.delay),
            int(configuration.rank),
            origin,
        )
        raw_rows.append(
            {
                **key,
                "origin_bin": origin,
                **asdict(raw_score),
            }
        )
        for horizon_score in raw_horizon_scores:
            raw_horizon_rows.append(
                {
                    **key,
                    "origin_bin": origin,
                    **horizon_score,
                }
            )

raw_table = pd.DataFrame(raw_rows)
raw_horizon_table = pd.DataFrame(raw_horizon_rows)
raw_table.to_csv(OUTPUT_DIR / "raw_rate_development_origin_scores.csv", index=False)
raw_horizon_table.to_csv(
    OUTPUT_DIR / "raw_rate_development_horizon_scores.csv", index=False
)
raw_configuration_columns = [
    "window_nominal_seconds",
    "window_bins",
    "window_actual_seconds",
    "delay",
    "history_seconds",
    "rank",
]
raw_summary = (
    raw_table.groupby(raw_configuration_columns, as_index=False)
    .agg(
        successful_origins=("origin_bin", "nunique"),
        primary_loss_mean=("primary_loss", "mean"),
        development_r2=("common_mean_r2", "mean"),
        local_mean_development_r2=("local_mean_r2", "mean"),
        positive_r2_fraction=(
            "common_mean_r2", lambda values: float(np.mean(values > 0))
        ),
        positive_local_mean_r2_fraction=(
            "local_mean_r2", lambda values: float(np.mean(values > 0))
        ),
        skill_vs_persistence=("skill_vs_persistence", "mean"),
        skill_vs_ar1=("skill_vs_ar1", "mean"),
        unexpected_warning_count=("unexpected_warning_count", "sum"),
    )
)
raw_both_counts = (
    raw_table.assign(
        beats_both=lambda table: (
            (table["skill_vs_persistence"] > 0)
            & (table["skill_vs_ar1"] > 0)
        )
    )
    .groupby(raw_configuration_columns, as_index=False)
    .agg(beats_both_fraction=("beats_both", lambda values: float(np.mean(values))))
)
raw_summary = raw_summary.merge(
    raw_both_counts,
    on=raw_configuration_columns,
    how="left",
    validate="one_to_one",
)
raw_horizon_summary = (
    raw_horizon_table.groupby(
        raw_configuration_columns + ["horizon_bins"], as_index=False
    )
    .agg(
        horizon_r2=("r2_vs_common_mean", "mean"),
        horizon_local_mean_r2=("r2_vs_local_mean", "mean"),
        horizon_positive_r2_fraction=(
            "r2_vs_common_mean", lambda values: float(np.mean(values > 0))
        ),
        horizon_positive_local_mean_r2_fraction=(
            "r2_vs_local_mean", lambda values: float(np.mean(values > 0))
        ),
        horizon_skill_persistence=("skill_vs_persistence", "mean"),
        horizon_skill_ar1=("skill_vs_ar1", "mean"),
    )
)
for horizon in (1, FORECAST_HORIZON_BINS):
    values = raw_horizon_summary.loc[
        raw_horizon_summary["horizon_bins"] == horizon,
        raw_configuration_columns
        + [
            "horizon_r2",
            "horizon_local_mean_r2",
            "horizon_positive_r2_fraction",
            "horizon_positive_local_mean_r2_fraction",
            "horizon_skill_persistence",
            "horizon_skill_ar1",
        ],
    ].rename(
        columns={
            "horizon_r2": f"h{horizon}_r2",
            "horizon_local_mean_r2": f"h{horizon}_local_mean_r2",
            "horizon_positive_r2_fraction": f"h{horizon}_positive_r2_fraction",
            "horizon_positive_local_mean_r2_fraction": (
                f"h{horizon}_positive_local_mean_r2_fraction"
            ),
            "horizon_skill_persistence": f"h{horizon}_skill_persistence",
            "horizon_skill_ar1": f"h{horizon}_skill_ar1",
        }
    )
    raw_summary = raw_summary.merge(
        values,
        on=raw_configuration_columns,
        how="left",
        validate="one_to_one",
    )

raw_summary["development_gate_pass"] = (
    (raw_summary["successful_origins"] == len(DEVELOPMENT_ORIGINS))
    & (raw_summary["development_r2"] > 0)
    & (raw_summary["local_mean_development_r2"] > 0)
    & (raw_summary["positive_r2_fraction"] >= MIN_POSITIVE_ORIGIN_FRACTION)
    & (
        raw_summary["positive_local_mean_r2_fraction"]
        >= MIN_POSITIVE_ORIGIN_FRACTION
    )
    & (raw_summary["skill_vs_persistence"] > 0)
    & (raw_summary["skill_vs_ar1"] > 0)
    & (raw_summary["h1_r2"] > 0)
    & (raw_summary["h1_local_mean_r2"] > 0)
    & (raw_summary["h1_positive_r2_fraction"] >= MIN_POSITIVE_ORIGIN_FRACTION)
    & (
        raw_summary["h1_positive_local_mean_r2_fraction"]
        >= MIN_POSITIVE_ORIGIN_FRACTION
    )
    & (raw_summary["h1_skill_persistence"] > 0)
    & (raw_summary["h1_skill_ar1"] > 0)
    & (raw_summary[f"h{FORECAST_HORIZON_BINS}_r2"] > 0)
    & (raw_summary[f"h{FORECAST_HORIZON_BINS}_local_mean_r2"] > 0)
    & (
        raw_summary[f"h{FORECAST_HORIZON_BINS}_positive_r2_fraction"]
        >= MIN_POSITIVE_ORIGIN_FRACTION
    )
    & (
        raw_summary[
            f"h{FORECAST_HORIZON_BINS}_positive_local_mean_r2_fraction"
        ]
        >= MIN_POSITIVE_ORIGIN_FRACTION
    )
    & (raw_summary[f"h{FORECAST_HORIZON_BINS}_skill_persistence"] > 0)
    & (raw_summary[f"h{FORECAST_HORIZON_BINS}_skill_ar1"] > 0)
    & (raw_summary["beats_both_fraction"] >= MIN_POSITIVE_ORIGIN_FRACTION)
    & (raw_summary["unexpected_warning_count"] == 0)
)

# Add the same delete-pair SE before using a parsimony tie break.
raw_table["delete_pair"] = raw_table["origin_bin"].map(origin_to_pair)
raw_jackknife_rows = []
for key_values, group in raw_table.groupby(raw_configuration_columns, sort=False):
    leave_pair_out = np.asarray(
        [
            group.loc[group["delete_pair"] != pair, "primary_loss"].mean()
            for pair in sorted(group["delete_pair"].unique())
        ]
    )
    jackknife_mean = leave_pair_out.mean()
    jackknife_se = float(
        np.sqrt(3 / 4 * np.sum((leave_pair_out - jackknife_mean) ** 2))
    )
    raw_jackknife_rows.append(
        {
            **dict(zip(raw_configuration_columns, key_values, strict=True)),
            "delete_pair_jackknife_se": jackknife_se,
        }
    )
raw_summary = raw_summary.merge(
    pd.DataFrame(raw_jackknife_rows),
    on=raw_configuration_columns,
    how="left",
    validate="one_to_one",
)
raw_summary.to_csv(OUTPUT_DIR / "raw_rate_development_summary.csv", index=False)

raw_passing = raw_summary.loc[raw_summary["development_gate_pass"]]
raw_pool = raw_passing if not raw_passing.empty else raw_summary
raw_rate_best = raw_pool.sort_values(
    ["primary_loss_mean", "window_nominal_seconds", "rank", "delay"],
    kind="stable",
).iloc[0]
raw_one_se_threshold = float(
    raw_rate_best["primary_loss_mean"]
    + raw_rate_best["delete_pair_jackknife_se"]
)
raw_rate_selected = raw_pool.loc[
    raw_pool["primary_loss_mean"] <= raw_one_se_threshold
].sort_values(
    ["window_nominal_seconds", "rank", "delay", "primary_loss_mean"],
    kind="stable",
).iloc[0]

print("RAW-CENTERED RATE SENSITIVITY")
print(f"  fixed POD retained energy: {raw_pod_energy:.3%}")
print(
    f"  raw best: W={int(raw_rate_best['window_nominal_seconds'])} s, "
    f"d={int(raw_rate_best['delay'])}, r={int(raw_rate_best['rank'])}, "
    f"fixed-mean R²={float(raw_rate_best['development_r2']):.4f}, "
    f"local-mean R²={float(raw_rate_best['local_mean_development_r2']):.4f}"
)
print(
    f"  one-SE choice: W={int(raw_rate_selected['window_nominal_seconds'])} s, "
    f"d={int(raw_rate_selected['delay'])}, r={int(raw_rate_selected['rank'])}, "
    f"fixed-mean R²={float(raw_rate_selected['development_r2']):.4f}, "
    f"local-mean R²={float(raw_rate_selected['local_mean_development_r2']):.4f}"
)
print(
    "  NOTE — raw-rate and RMS-standardized R² answer different weighted "
    "questions and are not pooled."
)

standardized_best_by_window = (
    development_summary.sort_values("primary_loss_mean", kind="stable")
    .groupby("window_nominal_seconds", as_index=False)
    .first()
)
raw_best_by_window = (
    raw_summary.sort_values("primary_loss_mean", kind="stable")
    .groupby("window_nominal_seconds", as_index=False)
    .first()
)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
for axis, metric, title in (
    (axes[0], "development_r2", "Common-reference selection score"),
    (axes[1], "local_mean_development_r2", "Local incremental effect"),
):
    axis.plot(
        standardized_best_by_window["window_nominal_seconds"],
        standardized_best_by_window[metric],
        marker="o",
        label="RMS-standardized",
    )
    axis.plot(
        raw_best_by_window["window_nominal_seconds"],
        raw_best_by_window[metric],
        marker="o",
        label="raw-rate",
    )
    axis.axhline(0, color="0.5", linestyle="--")
    axis.set(
        xlabel="fit window W (s)",
        ylabel="best guarded R² at each W",
        title=title,
    )
    axis.legend(frameon=False)
axes[1].axhline(
    PRACTICAL_R2_TARGET,
    color="#D55E00",
    linestyle=":",
    label=f"local target {PRACTICAL_R2_TARGET:.2f}",
)
axes[1].legend(frameon=False)
fig.suptitle("Step 10 — Inner-product sensitivity remains development-only")
save_and_show(fig, "09_raw_vs_standardized_window_sensitivity.png")


# %% Step 10b — Check within-window drift instead of equating one state label with stationarity
#
# NREM label homogeneity prevents an obvious state transition inside W, but it
# does not prove that one dynamical system generated the whole window. As a
# descriptive locality diagnostic, the first eight fixed POD coordinates are
# split into chronological halves. We compare their means, covariances, and
# separately fitted ridge-stabilized one-step operators. We also ask whether an
# operator fitted in one half transfers to the other half better than that
# half's training mean. These quantities are not used to tune hyperparameters
# in this pilot and have no universal pass threshold; a future confirmatory
# protocol should calibrate them with state-preserving block surrogates or
# explicit change-point exclusion.

DRIFT_DIAGNOSTIC_DIMENSION = 8


def ridge_one_step_operator(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a small, numerically regularized one-step operator and its mean."""
    local_mean = np.mean(values, axis=1, keepdims=True)
    centered = values - local_mean
    x = centered[:, :-1]
    y = centered[:, 1:]
    gram = x @ x.T
    ridge = 1e-6 * max(float(np.trace(gram)) / gram.shape[0], 1.0)
    operator = y @ x.T @ np.linalg.inv(gram + ridge * np.eye(gram.shape[0]))
    return local_mean, operator


def transferred_one_step_r2(
    training_mean: np.ndarray,
    operator: np.ndarray,
    target_half: np.ndarray,
) -> float:
    """Teacher-forced one-step transfer score relative to the training mean."""
    x = target_half[:, :-1] - training_mean
    truth = target_half[:, 1:] - training_mean
    prediction = operator @ x
    denominator = float(np.sum(truth**2))
    return 1 - float(np.sum((truth - prediction) ** 2)) / denominator


drift_rows: list[dict[str, float | int]] = []
for window_seconds, window_bins in WINDOW_BINS.items():
    for origin in DEVELOPMENT_ORIGINS:
        context = target_latent[:DRIFT_DIAGNOSTIC_DIMENSION, origin - window_bins : origin]
        split = window_bins // 2
        first_half = context[:, :split]
        second_half = context[:, split:]
        first_mean, first_operator = ridge_one_step_operator(first_half)
        second_mean, second_operator = ridge_one_step_operator(second_half)
        first_centered = first_half - first_mean
        second_centered = second_half - second_mean
        pooled_rms = np.sqrt(
            0.5
            * (
                np.mean(np.sum(first_centered**2, axis=0))
                + np.mean(np.sum(second_centered**2, axis=0))
            )
        )
        first_covariance = np.cov(first_centered, bias=False)
        second_covariance = np.cov(second_centered, bias=False)
        covariance_scale = max(
            0.5
            * (
                np.linalg.norm(first_covariance, ord="fro")
                + np.linalg.norm(second_covariance, ord="fro")
            ),
            np.finfo(float).eps,
        )
        operator_scale = max(
            0.5
            * (
                np.linalg.norm(first_operator, ord="fro")
                + np.linalg.norm(second_operator, ord="fro")
            ),
            np.finfo(float).eps,
        )
        forward_transfer_r2 = transferred_one_step_r2(
            first_mean, first_operator, second_half
        )
        reverse_transfer_r2 = transferred_one_step_r2(
            second_mean, second_operator, first_half
        )
        drift_rows.append(
            {
                "window_nominal_seconds": window_seconds,
                "window_bins": window_bins,
                "origin_bin": origin,
                "mean_shift_over_pooled_rms": float(
                    np.linalg.norm(second_mean - first_mean) / pooled_rms
                ),
                "relative_covariance_drift": float(
                    np.linalg.norm(
                        second_covariance - first_covariance, ord="fro"
                    )
                    / covariance_scale
                ),
                "relative_operator_drift": float(
                    np.linalg.norm(
                        second_operator - first_operator, ord="fro"
                    )
                    / operator_scale
                ),
                "forward_transfer_r2": forward_transfer_r2,
                "reverse_transfer_r2": reverse_transfer_r2,
                "symmetric_transfer_r2": 0.5
                * (forward_transfer_r2 + reverse_transfer_r2),
            }
        )

drift_table = pd.DataFrame(drift_rows)
drift_summary = (
    drift_table.groupby(["window_nominal_seconds", "window_bins"], as_index=False)
    .agg(
        mean_shift=("mean_shift_over_pooled_rms", "median"),
        covariance_drift=("relative_covariance_drift", "median"),
        operator_drift=("relative_operator_drift", "median"),
        symmetric_transfer_r2=("symmetric_transfer_r2", "mean"),
        positive_transfer_fraction=(
            "symmetric_transfer_r2",
            lambda values: float(np.mean(values > 0)),
        ),
    )
)
drift_table.to_csv(OUTPUT_DIR / "within_window_drift_origin_scores.csv", index=False)
drift_summary.to_csv(OUTPUT_DIR / "within_window_drift_summary.csv", index=False)

print("WITHIN-WINDOW LOCALITY DIAGNOSTIC (descriptive, not a stationarity proof)")
print(
    drift_summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), constrained_layout=True)
axes[0].plot(
    drift_summary["window_nominal_seconds"],
    drift_summary["mean_shift"],
    marker="o",
    label="mean shift / RMS",
)
axes[0].plot(
    drift_summary["window_nominal_seconds"],
    drift_summary["covariance_drift"],
    marker="o",
    label="relative covariance drift",
)
axes[0].set(xlabel="W (s)", ylabel="dimensionless drift", title="Moment drift")
axes[0].legend(frameon=False, fontsize=8)
axes[1].plot(
    drift_summary["window_nominal_seconds"],
    drift_summary["operator_drift"],
    marker="o",
    color="#D55E00",
)
axes[1].set(
    xlabel="W (s)",
    ylabel="relative operator difference",
    title="Split-half operator drift",
)
axes[2].plot(
    drift_summary["window_nominal_seconds"],
    drift_summary["symmetric_transfer_r2"],
    marker="o",
    color="#0072B2",
)
axes[2].axhline(0, color="0.5", linestyle="--")
axes[2].set(
    xlabel="W (s)",
    ylabel="cross-half one-step R²",
    title="Operator transfer between halves",
)
fig.suptitle("Step 10b — Same state label does not guarantee one stationary system")
save_and_show(fig, "09b_within_window_drift.png")


# %% Step 11 — Decide whether it is scientifically safe to open the outer tail
#
# A low positive prediction R² is a technical signal, not yet satisfactory
# evidence for trackable dynamics. The outer tail is opened only if all of the
# following development-only conditions hold:
# 1. the multi-baseline technical gate passes;
# 2. the investigator-declared practical R² target is reached;
# 3. the complete one-SE selection rule recurs in at least half of delete-pair
#    screens;
# 4. the separate modal/residual gate passes.
#
# These thresholds are conservative pilot decisions, not universal theorems.
# If they fail, inspecting outer outcomes would only spend confirmation data to
# tune the method. The correct result is to keep those samples score-locked.

delete_pair_selection_match_fraction = float(
    delete_pair_winner_table["matches_full_one_se_choice"].mean()
)
practical_prediction_pass = bool(
    final_development_local_r2 >= PRACTICAL_R2_TARGET
)
selection_stability_pass = bool(
    development_pass
    and (
        final_window_seconds,
        final_delay,
        final_rank,
    )
    == full_selected_tuple
    and delete_pair_winner_table["gate_passing_pool_available"].all()
    and delete_pair_selection_match_fraction >= 0.5
)
development_satisfactory = bool(
    final_technical_pass
    and practical_prediction_pass
    and selection_stability_pass
    and modal_gate_pass
)

outer_rows: list[dict[str, float | int | str]] = []
outer_horizon_rows: list[dict[str, float | int | str]] = []
outer_opened = development_satisfactory
outer_gate_pass = False

if outer_opened:
    print("DEVELOPMENT PASSED — opening the frozen within-bout outer targets once")
    for origin in OUTER_ORIGINS:
        if final_family == "HankelDMD":
            outer_score, outer_horizons, _, _ = fit_hankel_at_origin(
                final_window_bins,
                final_delay,
                final_rank,
                origin,
            )
        else:
            outer_score, outer_horizons, _, _, _ = fit_bop_at_origin(
                final_window_bins,
                final_delay,
                final_rank,
                origin,
                final_constraint == "stable+conjugate_pairs",
            )
        outer_rows.append(
            {
                "family": final_family,
                "window_nominal_seconds": final_window_seconds,
                "window_bins": final_window_bins,
                "delay": final_delay,
                "rank": final_rank,
                "origin_bin": origin,
                "origin_seconds": origin * DT_SECONDS,
                **asdict(outer_score),
            }
        )
        for horizon_score in outer_horizons:
            outer_horizon_rows.append(
                {
                    "family": final_family,
                    "window_nominal_seconds": final_window_seconds,
                    "window_bins": final_window_bins,
                    "delay": final_delay,
                    "rank": final_rank,
                    "origin_bin": origin,
                    "origin_seconds": origin * DT_SECONDS,
                    **horizon_score,
                }
            )
    outer_table = pd.DataFrame(outer_rows)
    outer_horizon_table = pd.DataFrame(outer_horizon_rows)
    outer_gate_horizons = (1, FORECAST_HORIZON_BINS)
    outer_gate_metrics = [
        "r2_vs_common_mean",
        "r2_vs_local_mean",
        "skill_vs_persistence",
        "skill_vs_ar1",
    ]
    outer_endpoint_means = (
        outer_horizon_table.loc[
            outer_horizon_table["horizon_bins"].isin(outer_gate_horizons),
            ["horizon_bins", *outer_gate_metrics],
        ]
        .groupby("horizon_bins", sort=True)[outer_gate_metrics]
        .mean()
        .reindex(outer_gate_horizons)
    )
    outer_endpoint_pass = bool(
        np.isfinite(outer_endpoint_means.to_numpy()).all()
        and outer_endpoint_means.gt(0).all().all()
    )
    outer_endpoint_positive_fractions = (
        outer_horizon_table.loc[
            outer_horizon_table["horizon_bins"].isin(outer_gate_horizons)
        ]
        .groupby("horizon_bins", sort=True)[
            ["r2_vs_common_mean", "r2_vs_local_mean"]
        ]
        .agg(lambda values: float(np.mean(values > 0)))
        .reindex(outer_gate_horizons)
    )
    outer_endpoint_fraction_pass = bool(
        np.isfinite(outer_endpoint_positive_fractions.to_numpy()).all()
        and outer_endpoint_positive_fractions.ge(0.8).all().all()
    )
    outer_gate_pass = bool(
        (outer_table["common_mean_r2"].mean() > 0)
        and (outer_table["local_mean_r2"].mean() > 0)
        and (outer_table["skill_vs_persistence"].mean() > 0)
        and (outer_table["skill_vs_ar1"].mean() > 0)
        and (np.mean(outer_table["common_mean_r2"] > 0) >= 0.8)
        and (np.mean(outer_table["local_mean_r2"] > 0) >= 0.8)
        and outer_endpoint_pass
        and outer_endpoint_fraction_pass
    )
else:
    outer_table = pd.DataFrame(
        columns=[
            "family",
            "window_nominal_seconds",
            "window_bins",
            "delay",
            "rank",
            "origin_bin",
            "origin_seconds",
            "primary_loss",
            "common_mean_r2",
            "local_mean_r2",
            "skill_vs_persistence",
            "skill_vs_ar1",
        ]
    )
    outer_horizon_table = pd.DataFrame(
        columns=[
            "family",
            "window_nominal_seconds",
            "window_bins",
            "delay",
            "rank",
            "origin_bin",
            "horizon_bins",
            "r2_vs_common_mean",
            "r2_vs_local_mean",
            "skill_vs_persistence",
            "skill_vs_ar1",
        ]
    )

outer_table.to_csv(OUTPUT_DIR / "outer_origin_scores.csv", index=False)
outer_horizon_table.to_csv(OUTPUT_DIR / "outer_horizon_scores.csv", index=False)

raw_sensitivity_practical_pass = bool(
    raw_rate_selected["development_gate_pass"]
    and float(raw_rate_selected["local_mean_development_r2"])
    >= PRACTICAL_R2_TARGET
)
final_drift_summary = drift_summary.loc[
    drift_summary["window_nominal_seconds"] == final_window_seconds
].iloc[0]
final_modal_summary = modal_summary_table.loc[
    (modal_summary_table["window_bins"] == final_window_bins)
    & (modal_summary_table["delay"] == final_delay)
    & (modal_summary_table["rank"] == final_rank)
].iloc[0]
decision = {
    "data": DATA_PATH.name,
    "old_selected_block_seconds": OLD_SELECTED_BLOCK_SECONDS,
    "old_actual_dmd_fit_seconds": OLD_DMD_FIT_SECONDS,
    "bin_frames": BIN_FRAMES,
    "dt_seconds": DT_SECONDS,
    "window_seconds_grid": list(WINDOW_SECONDS),
    "delay_grid": list(DELAY_CANDIDATES),
    "rank_grid": list(RANK_CANDIDATES),
    "minimum_transitions_per_rank_heuristic": MIN_TRANSITIONS_PER_RANK,
    "primary_selection_metric": (
        "equal-horizon SSE_DMD / SSE_fixed_acquisition1_mean"
    ),
    "local_increment_metric": "1 - SSE_DMD / SSE_past_window_mean",
    "practical_target_metric": "local_mean_development_r2",
    "pod_observation_space": {
        "dimension": pod_dimension,
        "retained_energy": pod_retained_energy,
        "capacity_dimensions": list(POD_CAPACITY_DIMENSIONS),
        "maximum_local_mean_oracle_r2_by_dimension": {
            str(int(dimension)): float(value)
            for dimension, value in capacity_maxima.items()
        },
    },
    "one_se_prediction_choice": {
        "status": (
            "gate_passing_selection"
            if not passing_summary.empty
            else "fallback_no_gate_passing_configuration"
        ),
        "family": "HankelDMD",
        "window_nominal_seconds": selected_window_seconds,
        "window_actual_seconds": selected_window_bins * DT_SECONDS,
        "delay": selected_delay,
        "history_seconds": (selected_delay - 1) * DT_SECONDS,
        "rank": selected_rank,
        "common_mean_development_r2": float(selected["development_r2"]),
        "local_mean_development_r2": float(
            selected["local_mean_development_r2"]
        ),
    },
    "final_development_candidate": {
        "status": (
            "gate_passing_finalist"
            if final_technical_pass
            else "fallback_no_gate_passing_configuration"
        ),
        "family": final_family,
        "window_nominal_seconds": final_window_seconds,
        "window_actual_seconds": final_window_bins * DT_SECONDS,
        "delay": final_delay,
        "history_seconds": (final_delay - 1) * DT_SECONDS,
        "rank": final_rank,
        "common_mean_development_r2": final_development_common_r2,
        "local_mean_development_r2": final_development_local_r2,
        "eligible_mode_origin_fraction": float(
            final_modal_summary["eligible_origin_fraction"]
        ),
        "mode_audit_status": str(final_modal_summary["status"]),
        "spectrally_qualified": spectral_finalist_available,
    },
    "internal_resdmd_style_audit_candidate": {
        "window_nominal_seconds": residual_window_seconds,
        "window_actual_seconds": residual_window_bins * DT_SECONDS,
        "delay": residual_delay,
        "rank": residual_rank,
        "common_mean_development_r2": float(
            residual_candidate["development_r2"]
        ),
        "local_mean_development_r2": float(
            residual_candidate["local_mean_development_r2"]
        ),
        "eligible_mode_origin_fraction": eligible_mode_origin_fraction,
        "median_subspace_affinity": (
            median_affinity if np.isfinite(median_affinity) else None
        ),
        "adjacent_fit_overlap_fraction": median_adjacent_overlap,
        "global_monte_carlo_p": global_residual_p_lower,
        "candidate_selection_adjusted": residual_inference_confirmatory,
        "interpretation": "exploratory internal development diagnostic",
    },
    "raw_rate_lowest_loss_choice": {
        "window_nominal_seconds": int(raw_rate_best["window_nominal_seconds"]),
        "delay": int(raw_rate_best["delay"]),
        "rank": int(raw_rate_best["rank"]),
        "common_mean_development_r2": float(raw_rate_best["development_r2"]),
        "local_mean_development_r2": float(
            raw_rate_best["local_mean_development_r2"]
        ),
    },
    "raw_rate_one_se_choice": {
        "status": (
            "gate_passing_selection"
            if not raw_passing.empty
            else "fallback_no_gate_passing_configuration"
        ),
        "window_nominal_seconds": int(
            raw_rate_selected["window_nominal_seconds"]
        ),
        "delay": int(raw_rate_selected["delay"]),
        "rank": int(raw_rate_selected["rank"]),
        "common_mean_development_r2": float(
            raw_rate_selected["development_r2"]
        ),
        "local_mean_development_r2": float(
            raw_rate_selected["local_mean_development_r2"]
        ),
    },
    "gates": {
        "technical_prediction": final_technical_pass,
        "practical_r2_target": PRACTICAL_R2_TARGET,
        "practical_prediction": practical_prediction_pass,
        "delete_pair_selection_match_fraction": (
            delete_pair_selection_match_fraction
        ),
        "selection_stability_tuple_matches_final": bool(
            (final_window_seconds, final_delay, final_rank)
            == full_selected_tuple
        ),
        "selection_stability": selection_stability_pass,
        "final_candidate_eligible_mode_origin_fraction": float(
            final_modal_summary["eligible_origin_fraction"]
        ),
        "residual_audit_nominal_win_fraction": residual_null_win_fraction,
        "residual_audit_selection_adjusted": residual_inference_confirmatory,
        "mode_recurrence": mode_recurrence_pass,
        "modal_residual": modal_gate_pass,
        "raw_rate_practical_sensitivity": raw_sensitivity_practical_pass,
        "development_satisfactory": development_satisfactory,
    },
    "within_window_locality_diagnostic": {
        "status": "descriptive; not a stationarity certificate or gate",
        "dimension": DRIFT_DIAGNOSTIC_DIMENSION,
        "final_window_mean_shift": float(final_drift_summary["mean_shift"]),
        "final_window_covariance_drift": float(
            final_drift_summary["covariance_drift"]
        ),
        "final_window_operator_drift": float(
            final_drift_summary["operator_drift"]
        ),
        "final_window_symmetric_transfer_r2": float(
            final_drift_summary["symmetric_transfer_r2"]
        ),
    },
    "optimized_dmd": {
        "shortlist_variants": len(bop_summary),
        "gate_passing_convergence_clean": len(eligible_bop),
        "clear_paired_improvement": bop_clear_improvement,
    },
    "outer": {
        "opened": outer_opened,
        "status": (
            "evaluated once"
            if outer_opened
            else "score-locked and unscored because development gate failed"
        ),
        "gate_pass": outer_gate_pass,
        "origins": list(OUTER_ORIGINS),
    },
}
with (OUTPUT_DIR / "tuning_decision.json").open("w", encoding="utf-8") as stream:
    json.dump(decision, stream, indent=2, allow_nan=False)

script_completion_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
if script_completion_sha256 != SCRIPT_START_SHA256:
    with RUN_STATUS_PATH.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "status": "failed_script_changed_during_run",
                "run_started_unix_seconds": RUN_STARTED_AT,
                "run_failed_unix_seconds": time.time(),
                "script_start_sha256": SCRIPT_START_SHA256,
                "script_completion_sha256": script_completion_sha256,
            },
            stream,
            indent=2,
        )
    raise RuntimeError(
        "The tutorial script changed during execution; outputs are not "
        "attributed to either source version. Freeze the script and rerun."
    )

manifest = {
    "script": Path(__file__).name,
    "script_sha256": SCRIPT_START_SHA256,
    "seed": SEED,
    "pydmd_version": version("pydmd"),
    "data": str(DATA_PATH.relative_to(REPO_ROOT)),
    "data_size_bytes": DATA_PATH.stat().st_size,
    "data_mtime_unix_seconds": DATA_PATH.stat().st_mtime,
    "run_started_unix_seconds": RUN_STARTED_AT,
    "run_completed_unix_seconds": time.time(),
    "outer_opened": outer_opened,
    "generated_files": sorted(
        [
            "run_manifest.json",
            *[
                path.name
                for path in OUTPUT_DIR.iterdir()
                if path.is_file() and path.stat().st_mtime >= RUN_STARTED_AT
            ],
        ]
    ),
}
with (OUTPUT_DIR / "run_manifest.json").open("w", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2)
with RUN_STATUS_PATH.open("w", encoding="utf-8") as stream:
    json.dump(
        {
            "status": "complete",
            "run_started_unix_seconds": RUN_STARTED_AT,
            "run_completed_unix_seconds": manifest["run_completed_unix_seconds"],
            "script_sha256": manifest["script_sha256"],
        },
        stream,
        indent=2,
    )

print("FINAL VERIFICATION DECISION")
print(f"  technical prediction gate:      {final_technical_pass}")
print(
    f"  practical local-mean R² gate ({PRACTICAL_R2_TARGET:.2f}): "
    f"{practical_prediction_pass} (observed {final_development_local_r2:.4f})"
)
print(
    "  delete-pair one-SE recurrence:   "
    f"{delete_pair_selection_match_fraction:.3f} -> {selection_stability_pass}"
)
print(f"  modal/residual gate:             {modal_gate_pass}")
print(
    "  raw-rate practical sensitivity: "
    f"{raw_sensitivity_practical_pass}"
)
print(f"  satisfactory on development:     {development_satisfactory}")
print(f"  outer tail opened:                {outer_opened}")
if not outer_opened:
    print(
        "STOP — the final 41.83 s remains score-locked and unscored. "
        "The development-selected predictive advantage is unconfirmed, and "
        "no frequency-resolved, damping-qualified DMD mode is verified for "
        "tracking."
    )
else:
    print(f"  descriptive outer gate:          {outer_gate_pass}")
print(f"Saved machine-readable decision: {OUTPUT_DIR / 'tuning_decision.json'}")
print(f"Saved run manifest: {OUTPUT_DIR / 'run_manifest.json'}")
