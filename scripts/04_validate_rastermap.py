# %% [markdown]
# # 04 · Essential Rastermap validation
#
# Tutorial 03 shows how to make a paper-style Rastermap plot. This tutorial asks
# the smaller set of questions needed before interpreting that plot:
#
# 1. Is the order reproducible on the same input and on independent time blocks?
# 2. Does coarse Rastermap organization transfer better than simple controls?
# 3. Is the transferable structure state-specific?
# 4. Does removing the lower-activity half of the documented active population
#    materially change the answer?
#
# All six sleep and four anesthesia recordings are analyzed. Neurons are never
# randomly sampled. The primary population is the dataset's `nonzero_ROI` mask,
# intersected only with the numerical validity required for every comparison.
# Spatial localization is intentionally outside this tutorial.

# %% Step 0 — imports
from __future__ import annotations

import csv
import gc
import hashlib
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import h5py
import matplotlib.pyplot as plt
import numpy as np
from rastermap import Rastermap
from rastermap.cluster import compute_cc_tdelay
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr, zscore

from src.funcnet import dataio, rastermap_tools as rmt, timeseries as ts
from src.funcnet.paths import FIG_DIR, RESULTS_DIR

FIG_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_DIR = RESULTS_DIR / "rastermap_validation"
CHECKPOINT_DIR = VALIDATION_DIR / "04_checkpoints"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Step 1 — settings and fit budget
#
# The defaults deliberately retain the important uncertainty found by the
# archived validation notebooks while removing their overlapping sweeps:
#
# - Three state-matched block allocations are fitted with seed 0. Allocation 0
#   is repeated with seed 1. This is eight mixed-state fits per recording.
# - Allocation 0 is also fitted separately by state (four fits) and for the
#   onset-ranked top 50% population (two fits).
# - Thirty-second blocks never cross a state change or acquisition break. Three
#   seconds are removed from both ends, and lag-wide row-mean seams prevent
#   Rastermap from comparing unrelated block endpoints.
# - The orientation control uses at most 31 balanced block reversals. It keeps
#   every instantaneous population vector and each projected component's
#   within-block univariate autocovariance at every tested lag.
#
# Thus the full real-data run contains 140 official fits. The optional synthetic
# positive/null control adds two small fits. On the workstation used to develop
# this tutorial, a clean run is expected to take roughly 15–20 minutes.

# %%
SLEEP_RECORDINGS = (
    "mouse01_sleep",
    "mouse02_sleep",
    "mouse03_sleep",
    "mouse04_day1_sleep",
    "mouse04_day2_sleep",
    "mouse05_sleep",
)
ANESTHESIA_RECORDINGS = (
    "mouse03_ane",
    "mouse05_ane",
    "mouse06_ane",
    "mouse07_ane",
)
RECORDINGS = SLEEP_RECORDINGS + ANESTHESIA_RECORDINGS

ANALYSIS_SCHEMA = 1
QUALIFICATION_WINDOW_FRAMES = {"sleep": 1500, "anesthesia": 2900}
TOP_ACTIVE_FRACTION = 0.50

N_CLUSTERS = 100
N_PCS = 128
LOCALITY = 0.0
MEAN_TIME = True
PAPER_LAG_SECONDS = 5 / 3.2
PRIMARY_FIT_SEED = 0
SECONDARY_FIT_SEED = 1

SPLIT_SEEDS = (50_000, 60_000, 70_000)
BLOCK_SECONDS = 30.0
BLOCK_EDGE_GUARD_SECONDS = 3.0
NEIGHBORHOOD_SIZE = 50
TIE_PERMUTATIONS = 8
TIE_SEED_BASE = 700_000
ORIENTATION_SEED_BASE = 800_000
N_BLOCK_ORIENTATION_CONTROLS = 31
MIN_NEURONS_FOR_FIT = 256

RUN_FULL_ANALYSIS = True
REUSE_CHECKPOINTS = True
RUN_SYNTHETIC_CONTROL = True
SHOW_FIGURES = True

# This environment-only switch is used for automated helper/synthetic smoke
# testing. It does not change the interactive default above.
SMOKE_ONLY = os.environ.get("FUNCNET_RASTERMAP_SMOKE", "0") == "1"
if SMOKE_ONLY:
    RUN_FULL_ANALYSIS = False
    REUSE_CHECKPOINTS = False
    SHOW_FIGURES = False

SESSION_PATH = VALIDATION_DIR / "04_rastermap_sessions.csv"
ALLOCATION_PATH = VALIDATION_DIR / "04_rastermap_allocations.csv"
MOUSE_PATH = VALIDATION_DIR / "04_rastermap_mouse_summary.csv"
SYNTHETIC_PATH = VALIDATION_DIR / "04_rastermap_synthetic.csv"
RASTERMAP_VERSION = version("rastermap")


# %% [markdown]
# ## Step 2 — lightweight loading and exact active-mask reconstruction
#
# Only deconvolved activity, state annotations, acquisition breaks,
# `used_frame`, and `nonzero_ROI` are loaded. The published mask is reconstructed
# exactly before any fit: every neuron must contain a positive OASIS value in
# every complete qualification window (1,500 frames for sleep; 2,900 for
# anesthesia). Incomplete remainders are excluded, matching the source data.
#
# Positive-run onsets reset at qualification-window boundaries, gaps in
# `used_frame`, and microscope breaks. They are a numerical activity proxy, not
# calibrated spikes or a physiological firing rate. Positive-bin counts are
# used only to show whether the ranking proxy itself changes the selected half.


# %%
def condition_and_states(recording_name: str):
    """Return condition, mouse, two primary states, and readable labels."""
    mouse = recording_name.split("_")[0]
    if recording_name.endswith("_sleep"):
        return "sleep", mouse, (0.0, 1.0), {0.0: "awake", 1.0: "nrem"}
    if recording_name.endswith("_ane"):
        return (
            "anesthesia",
            mouse,
            (0.0, 1.0),
            {0.0: "awake", 1.0: "anesthesia"},
        )
    raise ValueError(f"Cannot infer condition from {recording_name!r}")


def load_analysis_arrays(recording_name: str):
    """Load only arrays required by this validation."""
    path = dataio.RAW_DIR / f"{recording_name}.mat"
    with h5py.File(path, "r") as raw:
        state = np.asarray(raw["state"], dtype=np.float32).ravel()
        stored = raw["spike_deconv"].astype(np.float32)[:]
        if stored.shape[0] == state.size:
            activity = np.ascontiguousarray(stored.T)
        elif stored.shape[1] == state.size:
            activity = np.ascontiguousarray(stored)
        else:
            raise ValueError(
                f"{recording_name}: spike_deconv does not align with state"
            )
        nonzero_roi = np.asarray(raw["nonzero_ROI"], dtype=bool).ravel()
        boundary_ind = (
            np.asarray(raw["frame"]["boundary_ind"], dtype=np.int64).ravel() - 1
        )
        used_frame = tuple(
            np.asarray(raw[reference], dtype=np.int64).ravel() - 1
            for reference in np.asarray(raw["frame"]["used_frame"]).ravel()
        )
    if activity.shape[0] != nonzero_roi.size:
        raise ValueError(f"{recording_name}: nonzero_ROI is not neuron-aligned")
    if len(used_frame) != 2:
        raise ValueError(f"{recording_name}: expected two used_frame vectors")
    for frames in used_frame:
        if (
            frames.size == 0
            or frames.min() < 0
            or frames.max() >= activity.shape[1]
            or np.any(np.diff(frames) <= 0)
        ):
            raise ValueError(f"{recording_name}: invalid used_frame vector")
    return path, activity, state, nonzero_roi, boundary_ind, used_frame


@dataclass(frozen=True)
class QualificationSummary:
    """Activity proxies and exact mask-reconstruction provenance."""

    positive_bins: np.ndarray
    positive_run_onsets: np.ndarray
    analyzed_frames: int
    windows_by_state: tuple[int, int]
    reconstructed_nonzero_roi: np.ndarray


def qualification_proxy_counts(
    activity: np.ndarray,
    used_frame: tuple[np.ndarray, np.ndarray],
    boundary_ind: np.ndarray,
    window_frames: int,
) -> QualificationSummary:
    """Reconstruct nonzero_ROI and count support proxies in complete windows."""
    n_neurons, n_frames = activity.shape
    break_after_frame = np.zeros(n_frames, dtype=bool)
    boundaries = np.asarray(boundary_ind, dtype=np.int64).ravel()
    boundaries = boundaries[(boundaries >= 0) & (boundaries < n_frames)]
    break_after_frame[boundaries] = True

    positive_bins = np.zeros(n_neurons, dtype=np.int64)
    positive_run_onsets = np.zeros(n_neurons, dtype=np.int64)
    reconstructed = np.ones(n_neurons, dtype=bool)
    windows_by_state = []
    analyzed_frames = 0
    for state_frames in used_frame:
        n_windows = state_frames.size // window_frames
        windows_by_state.append(n_windows)
        for window_index in range(n_windows):
            frames = state_frames[
                window_index * window_frames : (window_index + 1) * window_frames
            ]
            positive = activity[:, frames] > 0
            counts = np.count_nonzero(positive, axis=1)
            positive_bins += counts
            reconstructed &= counts > 0

            starts = np.ones(window_frames, dtype=bool)
            starts[1:] = (np.diff(frames) != 1) | break_after_frame[frames[:-1]]
            onset = positive.copy()
            onset[:, 1:] &= (~positive[:, :-1]) | starts[np.newaxis, 1:]
            positive_run_onsets += np.count_nonzero(onset, axis=1)
            analyzed_frames += window_frames

    if analyzed_frames == 0 or any(count == 0 for count in windows_by_state):
        raise ValueError("Each primary state must provide a complete window")
    return QualificationSummary(
        positive_bins=positive_bins,
        positive_run_onsets=positive_run_onsets,
        analyzed_frames=analyzed_frames,
        windows_by_state=(windows_by_state[0], windows_by_state[1]),
        reconstructed_nonzero_roi=reconstructed,
    )


def top_fraction_with_ties(
    values: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, float]:
    """Select a top fraction and every row tied at its boundary."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a nonempty finite vector")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    target = int(np.ceil(fraction * values.size))
    cutoff = float(np.sort(values)[-target])
    return values >= cutoff, cutoff


def row_hash(rows: np.ndarray) -> str:
    """Hash exact original ROI rows for checkpoint provenance."""
    canonical = np.ascontiguousarray(rows, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


# %% [markdown]
# ## Step 3 — state-pure guarded blocks
#
# These helpers split at both state changes and acquisition breaks, tile only
# complete blocks, and allocate equal A/B block counts independently within each
# primary state. The same original ROI rows are required to be finite and
# nonconstant in every matrix that will be compared.

# %%
Block = tuple[int, int, float]


def primary_segments(
    state: np.ndarray,
    boundary_ind: np.ndarray,
    allowed_codes: tuple[float, ...],
) -> list[Block]:
    """Return half-open constant-state segments that do not cross a break."""
    transitions = (
        np.flatnonzero(
            (state[1:] != state[:-1])
            | ~np.isfinite(state[1:])
            | ~np.isfinite(state[:-1])
        )
        + 1
    )
    segments = []
    for start, stop in ts.acquisition_segments(
        state.size,
        boundary_ind,
        extra_splits=transitions,
    ):
        code = float(state[start])
        if np.isfinite(code) and code in allowed_codes:
            segments.append((start, stop, code))
    return segments


def guarded_blocks(
    segments: list[Block],
    block_frames: int,
    guard_frames: int,
) -> list[Block]:
    """Tile state-pure segments and retain only guarded block interiors."""
    if 2 * guard_frames >= block_frames:
        raise ValueError("Two edge guards must be shorter than one block")
    blocks = []
    for start, stop, code in segments:
        for block_index in range((stop - start) // block_frames):
            block_start = start + block_index * block_frames
            blocks.append(
                (
                    block_start + guard_frames,
                    block_start + block_frames - guard_frames,
                    code,
                )
            )
    return blocks


def matched_fold_blocks(
    blocks: list[Block],
    allowed_codes: tuple[float, ...],
    seed: int,
) -> tuple[list[Block], list[Block], dict[float, int]]:
    """Allocate equal A/B counts separately within each state."""
    rng = np.random.default_rng(seed)
    fold_a: list[Block] = []
    fold_b: list[Block] = []
    counts_by_code = {}
    for code in allowed_codes:
        state_blocks = [block for block in blocks if block[2] == code]
        order = rng.permutation(len(state_blocks))
        usable = 2 * (len(state_blocks) // 2)
        if usable < 2:
            raise ValueError(f"State {code:g} has fewer than two complete blocks")
        half = usable // 2
        fold_a.extend(state_blocks[index] for index in order[:half])
        fold_b.extend(state_blocks[index] for index in order[half:usable])
        counts_by_code[code] = half
    return sorted(fold_a), sorted(fold_b), counts_by_code


def block_hash(fold_a: list[Block], fold_b: list[Block]) -> str:
    """Hash an ordered allocation, including fold and state code."""
    encoded = []
    for fold_index, blocks in enumerate((fold_a, fold_b)):
        for start, stop, code in blocks:
            encoded.extend((fold_index, start, stop, int(round(2 * code))))
    canonical = np.ascontiguousarray(encoded, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def canonical_allocation_hash(fold_a: list[Block], fold_b: list[Block]) -> str:
    """Treat a complete A/B swap as the same allocation."""
    return min(block_hash(fold_a, fold_b), block_hash(fold_b, fold_a))


def valid_rows_in_blocks(activity: np.ndarray, blocks: list[Block]) -> np.ndarray:
    """Find finite, nonconstant rows without constructing a joined matrix."""
    finite = np.ones(activity.shape[0], dtype=bool)
    minimum = np.full(activity.shape[0], np.inf, dtype=np.float32)
    maximum = np.full(activity.shape[0], -np.inf, dtype=np.float32)
    for start, stop, _code in blocks:
        values = activity[:, start:stop]
        finite &= np.all(np.isfinite(values), axis=1)
        minimum = np.minimum(minimum, np.min(values, axis=1))
        maximum = np.maximum(maximum, np.max(values, axis=1))
    return finite & (maximum > minimum)


def matrix_from_blocks(
    activity: np.ndarray,
    blocks: list[Block],
    separator_frames: int,
) -> tuple[np.ndarray, np.ndarray, list[slice], np.ndarray]:
    """Join blocks with row-mean seams and return counts/slices/state codes."""
    if not blocks:
        raise ValueError("At least one block is required")
    data_frames = sum(stop - start for start, stop, _code in blocks)
    total_frames = data_frames + separator_frames * (len(blocks) - 1)
    matrix = np.empty((activity.shape[0], total_frames), dtype=np.float32)
    positive_counts = np.zeros(activity.shape[0], dtype=np.int64)
    row_sum = np.zeros(activity.shape[0], dtype=np.float64)
    data_slices = []
    seam_slices = []
    block_codes = np.empty(len(blocks), dtype=np.float32)
    position = 0
    for block_index, (start, stop, code) in enumerate(blocks):
        values = activity[:, start:stop]
        width = stop - start
        data_slice = slice(position, position + width)
        matrix[:, data_slice] = values
        positive_counts += np.count_nonzero(values > 0, axis=1)
        row_sum += np.sum(values, axis=1, dtype=np.float64)
        data_slices.append(data_slice)
        block_codes[block_index] = code
        position += width
        if block_index < len(blocks) - 1 and separator_frames:
            seam_slices.append(slice(position, position + separator_frames))
            position += separator_frames
    row_mean = (row_sum / data_frames).astype(np.float32)
    for seam in seam_slices:
        matrix[:, seam] = row_mean[:, np.newaxis]
    if position != total_frames:
        raise RuntimeError("Block concatenation produced an unexpected width")
    return matrix, positive_counts, data_slices, block_codes


# %% [markdown]
# ## Step 4 — official fits and comparison metrics
#
# Fine order is described by absolute Spearman correlation and chance-adjusted
# local-neighborhood overlap. Absolute correlation is appropriate because a
# one-dimensional map can be globally reversed. Ties in Rastermap's upsampled
# grid are broken independently and repeatedly for the local metric.
#
# Coarse transfer projects a target fold through the source model's neuron PCA
# basis, rebuilds the directed lagged cluster similarity, and scores the source
# cluster order with Rastermap's own matching kernel. Runtime assertions require
# this transfer equation to reproduce every fitted model's training similarity.
# The learned order is compared with exact-random, activity-ranked, and reversed
# cluster orders.


# %%
@dataclass
class FitSummary:
    """Compact neuron-aligned output retained from one fit."""

    embedding: np.ndarray
    runtime_seconds: float


def fit_matrix(
    activity: np.ndarray,
    lag_frames: int,
    seed: int,
    *,
    n_clusters: int = N_CLUSTERS,
    n_pcs: int = N_PCS,
    locality: float = LOCALITY,
) -> tuple[Rastermap, FitSummary]:
    """Fit one official Rastermap model to a prescreened matrix."""
    if activity.shape[0] < MIN_NEURONS_FOR_FIT:
        raise ValueError(
            f"Only {activity.shape[0]} neurons remain; need at least "
            f"{MIN_NEURONS_FOR_FIT}"
        )
    fitted_pcs = min(n_pcs, activity.shape[0] - 1, activity.shape[1] - 1)
    model = Rastermap(
        n_clusters=n_clusters,
        n_PCs=fitted_pcs,
        locality=locality,
        time_lag_window=lag_frames,
        time_bin=1,
        mean_time=MEAN_TIME,
        bin_size=50,
        random_state=seed,
        keep_norm_X=True,
        verbose=False,
    ).fit(np.ascontiguousarray(activity, dtype=np.float32), compute_X_embedding=False)
    good = np.asarray(model.igood, dtype=bool).ravel()
    if good.size != activity.shape[0] or not np.all(good):
        raise RuntimeError("A prescreened fit unexpectedly removed neuron rows")
    return model, FitSummary(
        embedding=np.asarray(model.embedding, dtype=np.float32).ravel().copy(),
        runtime_seconds=float(model.runtime),
    )


def verify_normalized_separators(
    model: Rastermap,
    block_slices: list[slice],
) -> None:
    """Assert that seams are lag-wide and zero after row normalization."""
    lag_frames = int(model.time_lag_window)
    for first, second in zip(block_slices[:-1], block_slices[1:], strict=True):
        if second.start - first.stop < lag_frames:
            raise RuntimeError("A concatenation seam is shorter than the fitted lag")
    separator_mask = np.ones(model.X.shape[1], dtype=bool)
    for block_slice in block_slices:
        separator_mask[block_slice] = False
    if np.any(separator_mask) and not np.allclose(
        model.X[:, separator_mask],
        0,
        atol=2e-5,
        rtol=0,
    ):
        raise RuntimeError("Row-mean separators are not zero after normalization")


def transferred_temporal_scores(
    source_model: Rastermap,
    target_normalized_activity: np.ndarray,
) -> np.ndarray:
    """Project target activity through the source model's neuron PCA basis."""
    singular_values = np.asarray(source_model.sv, dtype=np.float32)
    if np.any(~np.isfinite(singular_values)) or np.any(singular_values <= 0):
        raise RuntimeError("Source singular values must be finite and positive")
    source_left = np.asarray(source_model.Usv, dtype=np.float32) / singular_values
    return (target_normalized_activity.T @ source_left) / singular_values


def transferred_node_similarity(
    source_model: Rastermap,
    target_normalized_activity: np.ndarray,
) -> np.ndarray:
    """Evaluate source cluster templates on target normalized activity."""
    temporal_scores = transferred_temporal_scores(
        source_model,
        target_normalized_activity,
    )
    return compute_cc_tdelay(
        temporal_scores,
        np.asarray(source_model.U_nodes, dtype=np.float32),
        time_lag_window=int(source_model.time_lag_window),
        symmetric=False,
    )


def verify_training_replay(model: Rastermap) -> None:
    """Require the transfer equations to reconstruct the fitted training cc."""
    replay = transferred_node_similarity(model, model.X)
    if not np.allclose(replay, model.cc, atol=2e-5, rtol=2e-5):
        raise RuntimeError("Transfer math did not replay Rastermap training cc")


def fit_and_verify(
    activity: np.ndarray,
    block_slices: list[slice],
    lag_frames: int,
    seed: int,
    **fit_options,
) -> tuple[Rastermap, FitSummary]:
    """Fit once, then run the separator and transfer-replay assertions."""
    model, summary = fit_matrix(
        activity,
        lag_frames,
        seed,
        **fit_options,
    )
    verify_normalized_separators(model, block_slices)
    verify_training_replay(model)
    return model, summary


def fine_metrics(
    first: FitSummary,
    second: FitSummary,
    metric_seed: int,
) -> dict[str, float]:
    """Return reversal-invariant global and tie-aware local agreement."""
    n_rows = first.embedding.size
    if second.embedding.shape != (n_rows,):
        raise ValueError("Fine comparisons require aligned neuron populations")
    neighborhood = min(NEIGHBORHOOD_SIZE, n_rows - 1)
    raw = rmt.rank_neighborhood_overlap(
        first.embedding,
        second.embedding,
        neighborhood_size=neighborhood,
        tie_permutations=TIE_PERMUTATIONS,
        random_state=metric_seed,
    )
    chance = neighborhood / (n_rows - 1)
    return {
        "abs_spearman": rmt.reversal_invariant_rank_correlation(
            first.embedding,
            second.embedding,
        ),
        "local_overlap_adjusted": float((raw - chance) / (1 - chance)),
    }


def directional_objective(
    node_similarity: np.ndarray,
    matching_target: np.ndarray,
    cluster_order: np.ndarray,
) -> float:
    """Score a complete directed cluster order with Rastermap's kernel."""
    similarity = np.asarray(node_similarity, dtype=np.float64)
    target = np.asarray(matching_target, dtype=np.float64)
    order = np.asarray(cluster_order, dtype=np.int64)
    if similarity.ndim != 2 or similarity.shape != target.shape:
        raise ValueError("Similarity and target must be aligned square matrices")
    if order.size != similarity.shape[0] or np.unique(order).size != order.size:
        raise ValueError("cluster_order must be a complete permutation")
    weights = np.triu(target, k=1)
    weight_sum = weights.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("Rastermap matching weights must have positive mass")
    ordered = similarity[np.ix_(order, order)]
    return float(np.sum(weights * ordered) / weight_sum)


def exact_random_order_expectation(node_similarity: np.ndarray) -> float:
    """Return the exact objective mean under a uniform complete order."""
    similarity = np.asarray(node_similarity, dtype=np.float64)
    if (
        similarity.ndim != 2
        or similarity.shape[0] != similarity.shape[1]
        or similarity.shape[0] < 2
    ):
        raise ValueError("node_similarity must be square with at least two nodes")
    n_clusters = similarity.shape[0]
    return float(
        (similarity.sum() - np.trace(similarity)) / (n_clusters * (n_clusters - 1))
    )


def activity_cluster_order(
    model: Rastermap,
    neuron_positive_counts: np.ndarray,
) -> np.ndarray:
    """Rank clusters by source activity and orient that order on source data."""
    assignments = np.asarray(model.embedding_clust, dtype=np.int64)
    n_clusters = np.asarray(model.U_nodes).shape[0]
    cluster_activity = np.empty(n_clusters, dtype=np.float64)
    for cluster in range(n_clusters):
        members = assignments == cluster
        if not np.any(members):
            raise RuntimeError("A fitted Rastermap cluster has no assigned neurons")
        cluster_activity[cluster] = neuron_positive_counts[members].mean()
    ascending = np.argsort(cluster_activity, kind="stable")
    descending = ascending[::-1]
    return (
        ascending
        if directional_objective(model.cc, model.BBt, ascending)
        >= directional_objective(model.cc, model.BBt, descending)
        else descending
    )


def balanced_orientation_masks(
    block_codes: np.ndarray,
    maximum_masks: int,
    seed: int,
) -> tuple[list[np.ndarray], int]:
    """Reverse approximately half of each state's blocks in unique ways."""
    block_codes = np.asarray(block_codes)
    groups = [np.flatnonzero(block_codes == code) for code in np.unique(block_codes)]
    sizes_by_group = []
    option_counts = []
    for indices in groups:
        if indices.size < 2:
            raise ValueError("Orientation controls require at least two blocks")
        lower = indices.size // 2
        sizes = (lower,) if indices.size % 2 == 0 else (lower, lower + 1)
        sizes_by_group.append(sizes)
        option_counts.append(sum(math.comb(indices.size, size) for size in sizes))
    total_possible = math.prod(option_counts)
    n_requested = min(maximum_masks, total_possible)

    masks = []
    if total_possible <= maximum_masks:
        group_options = []
        for indices, sizes in zip(groups, sizes_by_group, strict=True):
            options = []
            for size in sizes:
                options.extend(itertools.combinations(indices.tolist(), size))
            group_options.append(options)
        for choices in itertools.product(*group_options):
            mask = np.zeros(block_codes.size, dtype=bool)
            for chosen in choices:
                mask[np.asarray(chosen, dtype=np.int64)] = True
            masks.append(mask)
        return masks, total_possible

    rng = np.random.default_rng(seed)
    seen = set()
    while len(masks) < n_requested:
        mask = np.zeros(block_codes.size, dtype=bool)
        for indices, sizes in zip(groups, sizes_by_group, strict=True):
            size = sizes[int(rng.integers(len(sizes)))]
            mask[rng.choice(indices, size=size, replace=False)] = True
        key = tuple(np.flatnonzero(mask).tolist())
        if key not in seen:
            seen.add(key)
            masks.append(mask)
    return masks, total_possible


def block_autocovariance_signature(
    temporal_scores: np.ndarray,
    block_slices: list[slice],
    maximum_lag: int,
) -> np.ndarray:
    """Sum component-wise within-block lag products for an invariant check."""
    signature = np.zeros(
        (maximum_lag + 1, temporal_scores.shape[1]),
        dtype=np.float64,
    )
    for block_slice in block_slices:
        block = temporal_scores[block_slice]
        for lag in range(maximum_lag + 1):
            if lag == 0:
                signature[lag] += np.sum(block * block, axis=0, dtype=np.float64)
            else:
                signature[lag] += np.sum(
                    block[lag:] * block[:-lag],
                    axis=0,
                    dtype=np.float64,
                )
    return signature


def zero_lag_node_similarity(
    source_model: Rastermap,
    temporal_scores: np.ndarray,
) -> np.ndarray:
    """Return zero-lag node similarity for the orientation invariant check."""
    node_activity = zscore(
        np.asarray(source_model.U_nodes, dtype=np.float32) @ temporal_scores.T,
        axis=1,
    )
    return node_activity @ node_activity.T / node_activity.shape[1]


def coarse_transfer_metrics(
    source_model: Rastermap,
    target_model: Rastermap,
    source_positive_counts: np.ndarray,
    *,
    target_block_slices: list[slice] | None = None,
    target_block_codes: np.ndarray | None = None,
    orientation_seed: int | None = None,
) -> dict[str, float | int]:
    """Evaluate learned, exact-random, activity, reversal, and orientation controls."""
    temporal_scores = transferred_temporal_scores(source_model, target_model.X)
    transferred = compute_cc_tdelay(
        temporal_scores,
        np.asarray(source_model.U_nodes, dtype=np.float32),
        time_lag_window=int(source_model.time_lag_window),
        symmetric=False,
    )
    identity = np.arange(source_model.U_nodes.shape[0], dtype=np.int64)
    learned = directional_objective(transferred, source_model.BBt, identity)
    reversed_score = directional_objective(
        transferred,
        source_model.BBt,
        identity[::-1],
    )
    activity_order = activity_cluster_order(source_model, source_positive_counts)
    activity_score = directional_objective(
        transferred,
        source_model.BBt,
        activity_order,
    )
    random_expectation = exact_random_order_expectation(transferred)
    result: dict[str, float | int] = {
        "learned": learned,
        "learned_minus_random": learned - random_expectation,
        "learned_minus_activity": learned - activity_score,
        "learned_minus_reversed": learned - reversed_score,
    }

    requested = (
        target_block_slices is not None,
        target_block_codes is not None,
        orientation_seed is not None,
    )
    if any(requested) and not all(requested):
        raise ValueError("Orientation-control arguments must be supplied together")
    if all(requested):
        assert target_block_slices is not None
        assert target_block_codes is not None
        assert orientation_seed is not None
        masks, total_possible = balanced_orientation_masks(
            target_block_codes,
            N_BLOCK_ORIENTATION_CONTROLS,
            orientation_seed,
        )
        orientation_scores = np.empty(len(masks), dtype=np.float64)
        reference_autocovariance = block_autocovariance_signature(
            temporal_scores,
            target_block_slices,
            int(source_model.time_lag_window),
        )
        reference_zero_lag = zero_lag_node_similarity(source_model, temporal_scores)
        for index, reverse_mask in enumerate(masks):
            oriented_scores = temporal_scores.copy()
            for reverse, block_slice in zip(
                reverse_mask,
                target_block_slices,
                strict=True,
            ):
                if reverse:
                    oriented_scores[block_slice] = temporal_scores[block_slice][::-1]
            if index == 0:
                surrogate_autocovariance = block_autocovariance_signature(
                    oriented_scores,
                    target_block_slices,
                    int(source_model.time_lag_window),
                )
                if not np.allclose(
                    surrogate_autocovariance,
                    reference_autocovariance,
                    atol=1e-4,
                    rtol=2e-5,
                ):
                    raise RuntimeError(
                        "Block reversal changed univariate component autocovariance"
                    )
                if not np.allclose(
                    zero_lag_node_similarity(source_model, oriented_scores),
                    reference_zero_lag,
                    atol=2e-5,
                    rtol=2e-5,
                ):
                    raise RuntimeError("Block reversal changed zero-lag similarity")
            oriented_similarity = compute_cc_tdelay(
                oriented_scores,
                np.asarray(source_model.U_nodes, dtype=np.float32),
                time_lag_window=int(source_model.time_lag_window),
                symmetric=False,
            )
            orientation_scores[index] = directional_objective(
                oriented_similarity,
                source_model.BBt,
                identity,
            )
        result.update(
            {
                "learned_minus_orientation": (
                    learned - float(orientation_scores.mean())
                ),
                "orientation_controls_used": len(masks),
                "orientation_controls_possible": total_possible,
            }
        )
    return result


def reciprocal_coarse_metrics(
    model_a: Rastermap,
    model_b: Rastermap,
    counts_a: np.ndarray,
    counts_b: np.ndarray,
) -> dict[str, float]:
    """Average deterministic coarse transfer in both directions."""
    forward = coarse_transfer_metrics(model_a, model_b, counts_a)
    backward = coarse_transfer_metrics(model_b, model_a, counts_b)
    return {
        field: float(np.mean([forward[field], backward[field]]))
        for field in (
            "learned_minus_random",
            "learned_minus_activity",
            "learned_minus_reversed",
        )
    }


def mean_field(records: list[dict[str, object]], field: str) -> float:
    """Return one explicitly converted descriptive mean."""
    return float(np.mean([float(record[field]) for record in records]))


# %% [markdown]
# ## Step 5 — compact exact checkpoints
#
# Each recording is one checkpoint unit. Reuse requires exact equality of the
# source-file signature, package version, settings, derived frame counts, block
# allocations, and original ROI-row hashes. A partial or stale checkpoint is
# ignored; models from different configurations are never combined.


# %%
def exact_configuration(
    source_path: Path,
    recording_name: str,
    *,
    fs: float,
    lag_frames: int,
    block_frames: int,
    guard_frames: int,
    window_frames: int,
    common_rows: np.ndarray,
    top_rows: np.ndarray,
    allocations: list[tuple[list[Block], list[Block], dict[float, int]]],
) -> dict[str, object]:
    """Describe every input whose change invalidates a recording checkpoint."""
    source_stat = source_path.stat()
    recording_index = RECORDINGS.index(recording_name)
    return {
        "schema": ANALYSIS_SCHEMA,
        "recording": recording_name,
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "rastermap_version": RASTERMAP_VERSION,
        "signal": "spike_deconv",
        "selection": "finite_nonconstant_and_exact_nonzero_ROI_common_all_fits",
        "sensitivity_selection": "qualification_positive_run_onsets_top_half_with_ties",
        "common_rows_sha256": row_hash(common_rows),
        "top_onset_rows_sha256": row_hash(top_rows),
        "top_fraction": TOP_ACTIVE_FRACTION,
        "qualification_window_frames": window_frames,
        "fs": fs,
        "lag_frames": lag_frames,
        "block_frames": block_frames,
        "guard_frames": guard_frames,
        "split_seed_bases": list(SPLIT_SEEDS),
        "split_seeds_actual": [seed + recording_index for seed in SPLIT_SEEDS],
        "allocation_hashes": [block_hash(a, b) for a, b, _counts in allocations],
        "primary_seed": PRIMARY_FIT_SEED,
        "secondary_seed": SECONDARY_FIT_SEED,
        "tie_metric_seed": TIE_SEED_BASE + recording_index,
        "orientation_seed_base_actual": ORIENTATION_SEED_BASE + 10 * recording_index,
        "n_clusters": N_CLUSTERS,
        "n_pcs": N_PCS,
        "locality": LOCALITY,
        "mean_time": MEAN_TIME,
        "time_bin": 1,
        "display_bin_size_unused_for_metrics": 50,
        "neighborhood_size": NEIGHBORHOOD_SIZE,
        "tie_permutations": TIE_PERMUTATIONS,
        "orientation_controls": N_BLOCK_ORIENTATION_CONTROLS,
    }


def valid_checkpoint(payload: object, configuration: dict[str, object]) -> bool:
    """Require an exact configuration and complete compact result structure."""
    if not isinstance(payload, dict) or payload.get("configuration") != configuration:
        return False
    selection_fields = {
        "recording",
        "condition",
        "mouse",
        "recorded_neurons",
        "dataset_active_neurons",
        "common_fit_neurons",
        "dataset_active_fraction",
        "onset_bin_spearman",
        "top50_onset_bin_jaccard",
    }
    mixed_fields = {
        "recording",
        "allocation",
        "fold_abs_spearman",
        "fold_local_adjusted",
        "coarse_learned_minus_random",
        "coarse_learned_minus_activity",
        "coarse_learned_minus_reversed",
        "same_input_seed_abs_spearman",
        "same_input_seed_local_adjusted",
        "fit_runtime_seconds",
    }
    state_fields = {
        "recording",
        "fine_within_abs_spearman",
        "fine_cross_abs_spearman",
        "fine_within_local_adjusted",
        "fine_cross_local_adjusted",
        "coarse_same_learned_minus_random",
        "coarse_cross_learned_minus_random",
        "coarse_same_learned_minus_orientation",
        "coarse_cross_learned_minus_orientation",
        "fit_runtime_seconds",
    }
    sensitivity_fields = {
        "recording",
        "primary_abs_spearman",
        "top50_abs_spearman",
        "primary_local_adjusted",
        "top50_local_adjusted",
        "primary_coarse_learned_minus_random",
        "top50_coarse_learned_minus_random",
        "fit_runtime_seconds",
    }
    selection = payload.get("selection")
    mixed = payload.get("mixed")
    state = payload.get("state")
    sensitivity = payload.get("sensitivity")
    return bool(
        isinstance(selection, dict)
        and selection_fields <= selection.keys()
        and selection.get("recording") == configuration["recording"]
        and isinstance(mixed, list)
        and len(mixed) == len(SPLIT_SEEDS)
        and all(
            isinstance(record, dict) and mixed_fields <= record.keys()
            for record in mixed
        )
        and all(
            record.get("recording") == configuration["recording"] for record in mixed
        )
        and {record.get("allocation") for record in mixed}
        == set(range(len(SPLIT_SEEDS)))
        and isinstance(state, dict)
        and state_fields <= state.keys()
        and state.get("recording") == configuration["recording"]
        and isinstance(sensitivity, dict)
        and sensitivity_fields <= sensitivity.keys()
        and sensitivity.get("recording") == configuration["recording"]
    )


def load_checkpoint(path: Path, configuration: dict[str, object]):
    """Load one exact checkpoint or return None."""
    if not path.exists() or not REUSE_CHECKPOINTS:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if valid_checkpoint(payload, configuration) else None


def save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    """Atomically replace one complete recording checkpoint."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_csv(path: Path, records: list[dict[str, object]]) -> None:
    """Write a same-schema list as a human-readable summary table."""
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


# %% [markdown]
# ## Step 6 — small end-to-end synthetic control
#
# Ground truth is used only for scoring, never fitting. Coordinated traveling
# waves should recover their latent neuron order; independent circular row
# shifts preserve every neuron's trace but destroy population coordination.


# %%
def simulate_traveling_waves(seed: int = 60_000):
    """Return a small shuffled traveling-wave population and latent positions."""
    n_neurons, n_frames, n_waves = 800, 2048, 10
    rng = np.random.default_rng(seed)
    latent = np.linspace(0, 1, n_neurons, dtype=np.float32)
    rate = np.full((n_neurons, n_frames), 0.003, dtype=np.float32)
    frame_axis = np.arange(n_frames, dtype=np.float32)[None, :]
    onsets = np.linspace(80, n_frames - 300, n_waves).astype(int)
    onsets += rng.integers(-25, 26, size=n_waves)
    for onset in onsets:
        centers = onset + int(rng.integers(100, 201)) * latent
        amplitude = rng.uniform(0.2, 0.5, n_neurons).astype(np.float32)
        rate += amplitude[:, None] * np.exp(
            -0.5 * ((frame_axis - centers[:, None]) / rng.uniform(5, 10)) ** 2
        )
    nuisance = gaussian_filter1d(rng.standard_normal(n_frames), 40)
    nuisance = (nuisance - nuisance.min()) / (np.ptp(nuisance) + 1e-6)
    activity = rng.poisson(rate + 0.01 * nuisance).astype(np.float32)
    permutation = rng.permutation(n_neurons)
    return activity[permutation], latent[permutation]


def independently_shift_rows(activity: np.ndarray, min_shift: int, seed: int):
    """Destroy synchrony while retaining every complete per-neuron trace."""
    rng = np.random.default_rng(seed)
    shifted = activity.copy()
    shifts = rng.integers(
        min_shift, activity.shape[1] - min_shift + 1, activity.shape[0]
    )
    for row, shift in enumerate(shifts):
        shifted[row] = np.roll(activity[row], int(shift))
    return shifted


def run_synthetic_control() -> list[dict[str, object]]:
    """Run one deterministic positive/null pair through the same assertions."""
    activity, latent = simulate_traveling_waves()
    shifted = independently_shift_rows(activity, min_shift=9, seed=70_000)
    valid = rmt.valid_activity_rows(activity) & rmt.valid_activity_rows(shifted)
    latent = latent[valid]
    blocks = [(0, 1016, 0.0), (1032, 2048, 0.0)]
    records = []
    for label, values in (
        ("coordinated", activity[valid]),
        ("row_shift_null", shifted[valid]),
    ):
        matrix, _counts, slices, _codes = matrix_from_blocks(values, blocks, 8)
        model, summary = fit_and_verify(
            matrix,
            slices,
            8,
            0,
            n_clusters=50,
            n_pcs=64,
            locality=0.75,
        )
        metrics = fine_metrics(
            FitSummary(latent, 0.0),
            summary,
            metric_seed=0,
        )
        records.append(
            {
                "control": label,
                "neurons": int(valid.sum()),
                **metrics,
                "runtime_seconds": summary.runtime_seconds,
            }
        )
        del model, matrix
        gc.collect()
    return records


def smoke_helper_assertions() -> None:
    """Exercise non-fit helpers on tiny deterministic arrays."""
    activity = np.array(
        [[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
        dtype=np.float32,
    )
    used = (np.arange(6), np.arange(6, 12))
    qualification = qualification_proxy_counts(activity, used, np.array([], int), 3)
    np.testing.assert_array_equal(
        qualification.reconstructed_nonzero_roi, [True, False]
    )
    masks, total = balanced_orientation_masks(np.array([0, 0, 1, 1]), 31, 1)
    assert len(masks) == total == 4 and len({tuple(mask) for mask in masks}) == 4
    similarity = np.arange(16, dtype=float).reshape(4, 4)
    target = np.triu(np.ones((4, 4)), 1)
    scores = [
        directional_objective(similarity, target, np.asarray(order))
        for order in itertools.permutations(range(4))
    ]
    np.testing.assert_allclose(
        np.mean(scores), exact_random_order_expectation(similarity)
    )


if SMOKE_ONLY:
    smoke_helper_assertions()
    print("Helper smoke assertions passed.")

synthetic_records = run_synthetic_control() if RUN_SYNTHETIC_CONTROL else []
if synthetic_records:
    save_csv(SYNTHETIC_PATH, synthetic_records)
    print("saved ->", SYNTHETIC_PATH)


# %% [markdown]
# ## Step 7 — run every recording with per-recording checkpointing
#
# Sleep fits use awake and NREM; anesthesia fits use awake and anesthesia.
# Quiet awake, REM, short block tails, and guard intervals remain visible in
# tutorial 02 but are not long/even enough for this matched validation design.

# %%
selection_records: list[dict[str, object]] = []
mixed_records: list[dict[str, object]] = []
state_records: list[dict[str, object]] = []
sensitivity_records: list[dict[str, object]] = []
new_real_fits = 0

if RUN_FULL_ANALYSIS:
    for recording_index, recording_name in enumerate(RECORDINGS):
        print(f"[{recording_index + 1}/{len(RECORDINGS)}] {recording_name}", flush=True)
        condition, mouse, state_codes, state_labels = condition_and_states(
            recording_name
        )
        source_path, activity, state, nonzero_roi, boundary_ind, used_frame = (
            load_analysis_arrays(recording_name)
        )
        fs = float(dataio.FS_HZ)
        lag_frames = round(PAPER_LAG_SECONDS * fs)
        block_frames = round(BLOCK_SECONDS * fs)
        guard_frames = round(BLOCK_EDGE_GUARD_SECONDS * fs)
        window_frames = QUALIFICATION_WINDOW_FRAMES[condition]

        qualification = qualification_proxy_counts(
            activity, used_frame, boundary_ind, window_frames
        )
        if not np.array_equal(qualification.reconstructed_nonzero_roi, nonzero_roi):
            mismatch = np.count_nonzero(
                qualification.reconstructed_nonzero_roi != nonzero_roi
            )
            raise RuntimeError(
                f"{recording_name}: nonzero_ROI reconstruction differs at "
                f"{mismatch} neurons"
            )
        whole_valid = rmt.valid_activity_rows(activity)
        primary_rows = np.flatnonzero(whole_valid & nonzero_roi)
        selected_activity = np.ascontiguousarray(
            activity[primary_rows], dtype=np.float32
        )
        n_recorded = activity.shape[0]
        del activity
        gc.collect()

        segments = primary_segments(state, boundary_ind, state_codes)
        blocks = guarded_blocks(segments, block_frames, guard_frames)
        session_split_seeds = tuple(seed + recording_index for seed in SPLIT_SEEDS)
        allocations = [
            matched_fold_blocks(blocks, state_codes, seed)
            for seed in session_split_seeds
        ]
        fingerprints = [canonical_allocation_hash(a, b) for a, b, _ in allocations]
        if len(set(fingerprints)) != len(fingerprints):
            raise RuntimeError(f"{recording_name}: duplicate A/B block allocation")

        comparison_block_sets = [
            fold for fold_a, fold_b, _counts in allocations for fold in (fold_a, fold_b)
        ]
        for code in state_codes:
            comparison_block_sets.extend(
                [
                    [block for block in allocations[0][fold_index] if block[2] == code]
                    for fold_index in (0, 1)
                ]
            )
        common_valid = np.logical_and.reduce(
            [
                valid_rows_in_blocks(selected_activity, group)
                for group in comparison_block_sets
            ]
        )
        common_rows = primary_rows[common_valid]
        if common_rows.size < MIN_NEURONS_FOR_FIT:
            raise RuntimeError(f"{recording_name}: only {common_rows.size} common rows")
        common_activity = np.ascontiguousarray(
            selected_activity[common_valid], dtype=np.float32
        )
        del selected_activity
        gc.collect()

        onset_values = qualification.positive_run_onsets[common_rows].astype(float)
        bin_values = qualification.positive_bins[common_rows].astype(float)
        onset_top, onset_cutoff = top_fraction_with_ties(
            onset_values, TOP_ACTIVE_FRACTION
        )
        bin_top, _bin_cutoff = top_fraction_with_ties(bin_values, TOP_ACTIVE_FRACTION)
        top_rows = common_rows[onset_top]
        union = np.count_nonzero(onset_top | bin_top)
        proxy_correlation = float(spearmanr(onset_values, bin_values).statistic)
        if not np.isfinite(proxy_correlation):
            raise RuntimeError(f"{recording_name}: activity proxies are constant")
        selection = {
            "recording": recording_name,
            "condition": condition,
            "mouse": mouse,
            "state0_label": state_labels[state_codes[0]],
            "state1_label": state_labels[state_codes[1]],
            "recorded_neurons": n_recorded,
            "dataset_active_neurons": int(nonzero_roi.sum()),
            "common_fit_neurons": int(common_rows.size),
            "onset_top50_neurons": int(onset_top.sum()),
            "dataset_active_fraction": float(nonzero_roi.mean()),
            "common_fraction_of_dataset_active": float(
                common_rows.size / nonzero_roi.sum()
            ),
            "onset_bin_spearman": proxy_correlation,
            "top50_onset_bin_jaccard": float(
                np.count_nonzero(onset_top & bin_top) / union
            ),
            "onset_cutoff_per_qualification_second": float(
                onset_cutoff * fs / qualification.analyzed_frames
            ),
            "qualification_frames": qualification.analyzed_frames,
            "state0_windows": qualification.windows_by_state[0],
            "state1_windows": qualification.windows_by_state[1],
        }
        configuration = exact_configuration(
            source_path,
            recording_name,
            fs=fs,
            lag_frames=lag_frames,
            block_frames=block_frames,
            guard_frames=guard_frames,
            window_frames=window_frames,
            common_rows=common_rows,
            top_rows=top_rows,
            allocations=allocations,
        )
        checkpoint_path = CHECKPOINT_DIR / f"{recording_name}.json"
        payload = load_checkpoint(checkpoint_path, configuration)
        if payload is not None:
            print("  reusing exact checkpoint", flush=True)
        else:
            session_mixed = []
            metric_seed = TIE_SEED_BASE + recording_index
            for allocation_index, (fold_a, fold_b, counts_by_code) in enumerate(
                allocations
            ):
                matrix_a, counts_a, slices_a, _codes_a = matrix_from_blocks(
                    common_activity, fold_a, lag_frames
                )
                matrix_b, counts_b, slices_b, _codes_b = matrix_from_blocks(
                    common_activity, fold_b, lag_frames
                )
                model_a, fit_a = fit_and_verify(
                    matrix_a, slices_a, lag_frames, PRIMARY_FIT_SEED
                )
                model_b, fit_b = fit_and_verify(
                    matrix_b, slices_b, lag_frames, PRIMARY_FIT_SEED
                )
                fine = fine_metrics(fit_a, fit_b, metric_seed)
                coarse = reciprocal_coarse_metrics(model_a, model_b, counts_a, counts_b)
                mixed = {
                    "recording": recording_name,
                    "condition": condition,
                    "mouse": mouse,
                    "allocation": allocation_index,
                    "split_seed": session_split_seeds[allocation_index],
                    "fit_neurons": int(common_rows.size),
                    "state0_blocks_per_fold": counts_by_code[state_codes[0]],
                    "state1_blocks_per_fold": counts_by_code[state_codes[1]],
                    "fold_abs_spearman": fine["abs_spearman"],
                    "fold_local_adjusted": fine["local_overlap_adjusted"],
                    **{f"coarse_{key}": value for key, value in coarse.items()},
                    "same_input_seed_abs_spearman": None,
                    "same_input_seed_local_adjusted": None,
                    "fit_runtime_seconds": fit_a.runtime_seconds
                    + fit_b.runtime_seconds,
                }
                if allocation_index == 0:
                    _model_a1, fit_a1 = fit_and_verify(
                        matrix_a, slices_a, lag_frames, SECONDARY_FIT_SEED
                    )
                    _model_b1, fit_b1 = fit_and_verify(
                        matrix_b, slices_b, lag_frames, SECONDARY_FIT_SEED
                    )
                    seed_a = fine_metrics(fit_a, fit_a1, metric_seed)
                    seed_b = fine_metrics(fit_b, fit_b1, metric_seed)
                    mixed["same_input_seed_abs_spearman"] = float(
                        np.mean([seed_a["abs_spearman"], seed_b["abs_spearman"]])
                    )
                    mixed["same_input_seed_local_adjusted"] = float(
                        np.mean(
                            [
                                seed_a["local_overlap_adjusted"],
                                seed_b["local_overlap_adjusted"],
                            ]
                        )
                    )
                    mixed["fit_runtime_seconds"] += (
                        fit_a1.runtime_seconds + fit_b1.runtime_seconds
                    )
                    del _model_a1, _model_b1
                    new_real_fits += 2
                session_mixed.append(mixed)
                new_real_fits += 2
                del model_a, model_b, matrix_a, matrix_b
                gc.collect()

            fold_blocks = {"A": allocations[0][0], "B": allocations[0][1]}
            models, summaries, counts, slices, codes = {}, {}, {}, {}, {}
            for state_index, code in enumerate(state_codes):
                for fold_index, fold_label in enumerate(("A", "B")):
                    key = (code, fold_label)
                    state_blocks = [
                        block for block in fold_blocks[fold_label] if block[2] == code
                    ]
                    matrix, counts[key], slices[key], codes[key] = matrix_from_blocks(
                        common_activity, state_blocks, lag_frames
                    )
                    models[key], summaries[key] = fit_and_verify(
                        matrix, slices[key], lag_frames, PRIMARY_FIT_SEED
                    )
                    del matrix
                    new_real_fits += 1
            fine_pairs = [
                (state_codes[0], "A", state_codes[0], "B", "within"),
                (state_codes[1], "A", state_codes[1], "B", "within"),
                *[
                    (state_codes[0], first, state_codes[1], second, "cross")
                    for first in ("A", "B")
                    for second in ("A", "B")
                ],
            ]
            fine_state = []
            for (
                first_code,
                first_fold,
                second_code,
                second_fold,
                comparison,
            ) in fine_pairs:
                fine_state.append(
                    {
                        "comparison": comparison,
                        **fine_metrics(
                            summaries[(first_code, first_fold)],
                            summaries[(second_code, second_fold)],
                            metric_seed,
                        ),
                    }
                )
            transfer_pairs = [
                (state_codes[0], "A", state_codes[0], "B", "same"),
                (state_codes[0], "B", state_codes[0], "A", "same"),
                (state_codes[1], "A", state_codes[1], "B", "same"),
                (state_codes[1], "B", state_codes[1], "A", "same"),
                *[
                    (source, source_fold, target, target_fold, "cross")
                    for source, target in (
                        (state_codes[0], state_codes[1]),
                        (state_codes[1], state_codes[0]),
                    )
                    for source_fold in ("A", "B")
                    for target_fold in ("A", "B")
                ],
            ]
            transfer_state = []
            for (
                source_code,
                source_fold,
                target_code,
                target_fold,
                comparison,
            ) in transfer_pairs:
                target_index = state_codes.index(target_code) * 2 + (target_fold == "B")
                metrics = coarse_transfer_metrics(
                    models[(source_code, source_fold)],
                    models[(target_code, target_fold)],
                    counts[(source_code, source_fold)],
                    target_block_slices=slices[(target_code, target_fold)],
                    target_block_codes=codes[(target_code, target_fold)],
                    orientation_seed=ORIENTATION_SEED_BASE
                    + 10 * recording_index
                    + target_index,
                )
                transfer_state.append({"comparison": comparison, **metrics})
            within = [
                record for record in fine_state if record["comparison"] == "within"
            ]
            cross = [record for record in fine_state if record["comparison"] == "cross"]
            same = [
                record for record in transfer_state if record["comparison"] == "same"
            ]
            cross_transfer = [
                record for record in transfer_state if record["comparison"] == "cross"
            ]
            state_result = {
                "recording": recording_name,
                "fine_within_abs_spearman": mean_field(within, "abs_spearman"),
                "fine_cross_abs_spearman": mean_field(cross, "abs_spearman"),
                "fine_within_local_adjusted": mean_field(
                    within, "local_overlap_adjusted"
                ),
                "fine_cross_local_adjusted": mean_field(
                    cross, "local_overlap_adjusted"
                ),
                **{
                    f"coarse_{group}_{field}": mean_field(records, field)
                    for group, records in (("same", same), ("cross", cross_transfer))
                    for field in (
                        "learned_minus_random",
                        "learned_minus_activity",
                        "learned_minus_reversed",
                        "learned_minus_orientation",
                    )
                },
                "fit_runtime_seconds": float(
                    sum(summary.runtime_seconds for summary in summaries.values())
                ),
            }
            del models, summaries
            gc.collect()

            top_activity = np.ascontiguousarray(
                common_activity[onset_top], dtype=np.float32
            )
            top_a, top_counts_a, top_slices_a, _ = matrix_from_blocks(
                top_activity, allocations[0][0], lag_frames
            )
            top_b, top_counts_b, top_slices_b, _ = matrix_from_blocks(
                top_activity, allocations[0][1], lag_frames
            )
            top_model_a, top_fit_a = fit_and_verify(
                top_a, top_slices_a, lag_frames, PRIMARY_FIT_SEED
            )
            top_model_b, top_fit_b = fit_and_verify(
                top_b, top_slices_b, lag_frames, PRIMARY_FIT_SEED
            )
            top_fine = fine_metrics(top_fit_a, top_fit_b, metric_seed)
            top_coarse = reciprocal_coarse_metrics(
                top_model_a, top_model_b, top_counts_a, top_counts_b
            )
            primary = session_mixed[0]
            sensitivity = {
                "recording": recording_name,
                "primary_abs_spearman": primary["fold_abs_spearman"],
                "top50_abs_spearman": top_fine["abs_spearman"],
                "primary_local_adjusted": primary["fold_local_adjusted"],
                "top50_local_adjusted": top_fine["local_overlap_adjusted"],
                "primary_coarse_learned_minus_random": primary[
                    "coarse_learned_minus_random"
                ],
                "top50_coarse_learned_minus_random": top_coarse["learned_minus_random"],
                "fit_runtime_seconds": top_fit_a.runtime_seconds
                + top_fit_b.runtime_seconds,
            }
            new_real_fits += 2
            payload = {
                "configuration": configuration,
                "selection": selection,
                "mixed": session_mixed,
                "state": state_result,
                "sensitivity": sensitivity,
            }
            save_checkpoint(checkpoint_path, payload)
            print("  saved exact checkpoint", flush=True)
            del top_activity, top_a, top_b, top_model_a, top_model_b
            del counts, slices, codes
            gc.collect()

        selection_records.append(payload["selection"])
        mixed_records.extend(payload["mixed"])
        state_records.append(payload["state"])
        sensitivity_records.append(payload["sensitivity"])
        del common_activity, state, nonzero_roi, qualification
        gc.collect()


# %% [markdown]
# ## Step 8 — concise session/mouse summaries
#
# The allocation table retains the three perturbations. The session table uses
# their median and range, and the mouse table averages mouse 4's two sleep days
# before cohort plotting. Repeated fits are uncertainty checks, not biological
# replicates.

# %%
session_records: list[dict[str, object]] = []
mouse_records: list[dict[str, object]] = []
if selection_records:
    for selection in selection_records:
        recording_name = str(selection["recording"])
        allocations = [
            record for record in mixed_records if record["recording"] == recording_name
        ]
        state_result = next(
            record for record in state_records if record["recording"] == recording_name
        )
        sensitivity = next(
            record
            for record in sensitivity_records
            if record["recording"] == recording_name
        )
        allocation_zero = next(
            record for record in allocations if record["allocation"] == 0
        )
        record = dict(selection)
        record["common_fraction_of_recorded"] = float(
            record["common_fit_neurons"]
        ) / float(record["recorded_neurons"])
        for field in (
            "fold_abs_spearman",
            "fold_local_adjusted",
            "coarse_learned_minus_random",
            "coarse_learned_minus_activity",
            "coarse_learned_minus_reversed",
        ):
            values = np.array([float(item[field]) for item in allocations])
            record[f"mixed_{field}_median"] = float(np.median(values))
            record[f"mixed_{field}_min"] = float(values.min())
            record[f"mixed_{field}_max"] = float(values.max())
        record["same_input_seed_abs_spearman"] = allocation_zero[
            "same_input_seed_abs_spearman"
        ]
        record["same_input_seed_local_adjusted"] = allocation_zero[
            "same_input_seed_local_adjusted"
        ]
        record.update(
            {key: value for key, value in state_result.items() if key != "recording"}
        )
        record.update(
            {
                f"sensitivity_{key}": value
                for key, value in sensitivity.items()
                if key not in ("recording", "fit_runtime_seconds")
            }
        )
        record["fit_runtime_seconds"] = float(
            sum(float(item["fit_runtime_seconds"]) for item in allocations)
            + float(state_result["fit_runtime_seconds"])
            + float(sensitivity["fit_runtime_seconds"])
        )
        session_records.append(record)

    mouse_metric_fields = [
        "dataset_active_fraction",
        "common_fraction_of_recorded",
        "onset_bin_spearman",
        "top50_onset_bin_jaccard",
        "same_input_seed_abs_spearman",
        "same_input_seed_local_adjusted",
        "mixed_fold_abs_spearman_median",
        "mixed_fold_local_adjusted_median",
        "mixed_coarse_learned_minus_random_median",
        "mixed_coarse_learned_minus_activity_median",
        "mixed_coarse_learned_minus_reversed_median",
        "fine_within_abs_spearman",
        "fine_cross_abs_spearman",
        "fine_within_local_adjusted",
        "fine_cross_local_adjusted",
        "coarse_same_learned_minus_random",
        "coarse_cross_learned_minus_random",
        "coarse_same_learned_minus_orientation",
        "coarse_cross_learned_minus_orientation",
        "sensitivity_primary_abs_spearman",
        "sensitivity_top50_abs_spearman",
        "sensitivity_primary_local_adjusted",
        "sensitivity_top50_local_adjusted",
        "sensitivity_primary_coarse_learned_minus_random",
        "sensitivity_top50_coarse_learned_minus_random",
    ]
    for condition in ("sleep", "anesthesia"):
        condition_sessions = [
            record for record in session_records if record["condition"] == condition
        ]
        for mouse in sorted({str(record["mouse"]) for record in condition_sessions}):
            sessions = [
                record for record in condition_sessions if record["mouse"] == mouse
            ]
            mouse_records.append(
                {
                    "condition": condition,
                    "mouse": mouse,
                    "n_sessions": len(sessions),
                    **{
                        field: float(
                            np.mean([float(record[field]) for record in sessions])
                        )
                        for field in mouse_metric_fields
                    },
                }
            )
    save_csv(ALLOCATION_PATH, mixed_records)
    save_csv(SESSION_PATH, session_records)
    save_csv(MOUSE_PATH, mouse_records)
    print("saved ->", ALLOCATION_PATH)
    print("saved ->", SESSION_PATH)
    print("saved ->", MOUSE_PATH)


# %% [markdown]
# ## Step 9 — two compact verification figures
#
# Lines connect metrics from the same mouse; colors denote experimental
# condition. The figures show individual mice rather than only cohort averages.

# %%
CONDITION_COLORS = {"sleep": "tab:blue", "anesthesia": "tab:orange"}


def paired_panel(axis, records, fields, labels, title, ylabel, *, zero=False):
    """Draw one small within-mouse comparison panel."""
    x = np.arange(len(fields))
    for record in records:
        values = [float(record[field]) for field in fields]
        axis.plot(
            x,
            values,
            marker="o" if record["condition"] == "sleep" else "s",
            ms=4,
            lw=0.8,
            alpha=0.75,
            color=CONDITION_COLORS[str(record["condition"])],
        )
    if zero:
        axis.axhline(0, color="0.6", lw=0.8, ls=(0, (3, 2)))
    axis.set_xticks(x, labels, rotation=18, ha="right")
    axis.set_title(title)
    axis.set_ylabel(ylabel)


if mouse_records:
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    paired_panel(
        axes[0, 0],
        mouse_records,
        ("dataset_active_fraction", "common_fraction_of_recorded"),
        ("dataset active", "common fit"),
        "Neuron eligibility",
        "fraction recorded",
    )
    paired_panel(
        axes[0, 1],
        mouse_records,
        ("onset_bin_spearman", "top50_onset_bin_jaccard"),
        ("proxy rank rho", "top-50 Jaccard"),
        "Activity-proxy agreement",
        "agreement",
    )
    if synthetic_records:
        axes[0, 2].bar(
            [0, 1],
            [float(record["abs_spearman"]) for record in synthetic_records],
            color=("tab:green", "0.55"),
        )
        axes[0, 2].set_xticks([0, 1], ("coordinated", "row-shift null"), rotation=18)
        axes[0, 2].set_ylabel("|Spearman| to truth")
        axes[0, 2].set_title("Synthetic end-to-end control")
    paired_panel(
        axes[1, 0],
        mouse_records,
        ("same_input_seed_abs_spearman", "mixed_fold_abs_spearman_median"),
        ("same input", "disjoint time"),
        "Fine global recurrence",
        "|Spearman|",
    )
    paired_panel(
        axes[1, 1],
        mouse_records,
        ("same_input_seed_local_adjusted", "mixed_fold_local_adjusted_median"),
        ("same input", "disjoint time"),
        "Fine local recurrence",
        "adjusted overlap",
        zero=True,
    )
    paired_panel(
        axes[1, 2],
        mouse_records,
        (
            "mixed_coarse_learned_minus_random_median",
            "mixed_coarse_learned_minus_activity_median",
            "mixed_coarse_learned_minus_reversed_median",
        ),
        ("minus random", "minus activity", "minus reverse"),
        "Held-out coarse transfer",
        "objective excess",
        zero=True,
    )
    figure.subplots_adjust(
        left=0.08, right=0.98, bottom=0.10, top=0.84, hspace=0.42, wspace=0.27
    )
    figure.suptitle("Rastermap selection and temporal robustness", y=0.975)
    figure.text(
        0.5,
        0.935,
        "blue circles = sleep · orange squares = anesthesia",
        ha="center",
    )
    first_figure_path = FIG_DIR / "04_rastermap_01_core_validation.png"
    figure.savefig(first_figure_path, dpi=150, bbox_inches="tight")
    print("saved ->", first_figure_path)
    if SHOW_FIGURES:
        plt.show()

    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    paired_panel(
        axes[0, 0],
        mouse_records,
        ("fine_within_abs_spearman", "fine_cross_abs_spearman"),
        ("within state", "cross state"),
        "State-specific fine order",
        "|Spearman|",
    )
    paired_panel(
        axes[0, 1],
        mouse_records,
        ("fine_within_local_adjusted", "fine_cross_local_adjusted"),
        ("within state", "cross state"),
        "State-specific local order",
        "adjusted overlap",
        zero=True,
    )
    paired_panel(
        axes[0, 2],
        mouse_records,
        (
            "coarse_same_learned_minus_random",
            "coarse_cross_learned_minus_random",
            "coarse_same_learned_minus_orientation",
            "coarse_cross_learned_minus_orientation",
        ),
        ("same−rand", "cross−rand", "same−orient", "cross−orient"),
        "State transfer and temporal control",
        "objective excess",
        zero=True,
    )
    for axis, suffix, title, ylabel in (
        (axes[1, 0], "abs_spearman", "High-activity fine order", "|Spearman|"),
        (axes[1, 1], "local_adjusted", "High-activity local order", "adjusted overlap"),
        (
            axes[1, 2],
            "coarse_learned_minus_random",
            "High-activity coarse transfer",
            "objective excess",
        ),
    ):
        paired_panel(
            axis,
            mouse_records,
            (f"sensitivity_primary_{suffix}", f"sensitivity_top50_{suffix}"),
            ("dataset active", "onset top 50%"),
            title,
            ylabel,
            zero="overlap" in ylabel or "excess" in ylabel,
        )
    figure.subplots_adjust(
        left=0.08, right=0.98, bottom=0.10, top=0.84, hspace=0.42, wspace=0.27
    )
    figure.suptitle("Rastermap state specificity and activity sensitivity", y=0.975)
    figure.text(
        0.5,
        0.935,
        "blue circles = sleep · orange squares = anesthesia",
        ha="center",
    )
    second_figure_path = FIG_DIR / "04_rastermap_02_state_and_activity.png"
    figure.savefig(second_figure_path, dpi=150, bbox_inches="tight")
    print("saved ->", second_figure_path)
    if SHOW_FIGURES:
        plt.show()


# %% [markdown]
# ## Step 10 — interpretation guardrails
#
# - High same-input agreement with weaker disjoint-time agreement means an
#   optimizer can reproduce a fit without establishing a unique biological
#   neuron sequence.
# - Positive learned-minus-random coarse transfer supports recurring population
#   organization. A much smaller learned-minus-reversed/orientation effect is
#   weaker evidence for a unique arrow of time.
# - Top 50% is an intentionally arbitrary sensitivity arm. It must not replace
#   the exactly reconstructed dataset-active population as the primary default.
# - Whole-recording selection, common validity, and target-side normalization
#   make this conditional/transductive validation, not prospective prediction.
# - Blocks from one recording are perturbations, not independent animals. State
#   same/cross contrasts do not test REM, quiet awake, anatomy, or localization.

# %%
if mouse_records:
    for condition in ("sleep", "anesthesia"):
        cohort = [
            record for record in mouse_records if record["condition"] == condition
        ]
        print(f"\n{condition} mouse summary (n={len(cohort)}):")
        for field in (
            "same_input_seed_abs_spearman",
            "mixed_fold_abs_spearman_median",
            "mixed_fold_local_adjusted_median",
            "mixed_coarse_learned_minus_random_median",
            "mixed_coarse_learned_minus_reversed_median",
        ):
            values = np.array([float(record[field]) for record in cohort])
            print(
                f"  {field}: median={np.median(values):.4f}, range={values.min():.4f}…{values.max():.4f}"
            )

print("\nEssential Rastermap validation configured.")
print("  expected real-data fits:", 14 * len(RECORDINGS))
print("  expected synthetic fits:", 2 if RUN_SYNTHETIC_CONTROL else 0)
print("  new real-data fits completed this run:", new_real_fits)
print("  estimated clean-run time: approximately 15–20 minutes")
