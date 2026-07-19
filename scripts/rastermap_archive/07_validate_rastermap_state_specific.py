# %% [markdown]
# # 07 · Is the transferable Rastermap organization state specific?
#
# Tutorial 05 fits awake and NREM/anesthesia blocks together. Its folds contain
# the same number of blocks from each state, but a mixed-state fit can still use
# fixed differences between state means. This tutorial removes that ambiguity:
# it fits Rastermap separately to each state in each time fold.
#
# For every recording, four official models are fitted:
#
# - state 0, fold A
# - state 0, fold B
# - state 1, fold A
# - state 1, fold B
#
# Fine neuron order is compared within and across states. Coarse Rastermap
# cluster order is transferred in both directions, standardized against complete
# random cluster orders, and compared with an activity-ranked order.
#
# This remains a conditional, transductive validation. The dataset's supplied
# published-window activity mask uses the complete recording's analysis windows,
# the common validity mask inspects all four matrices, and each held-out matrix
# is normalized using its own data. It is not a prospective prediction analysis.

# %% Step 0 — imports
import csv
import gc
import itertools
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import h5py
import matplotlib.pyplot as plt
import numpy as np
from rastermap import Rastermap
from rastermap.cluster import compute_cc_tdelay
from scipy.stats import zscore
from sklearn.metrics import adjusted_rand_score

from src.funcnet import dataio, rastermap_tools as rmt, timeseries as ts
from src.funcnet.paths import FIG_DIR, RESULTS_DIR

FIG_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_DIR = RESULTS_DIR / "rastermap_validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Step 1 — settings
#
# The primary active-neuron definition is finite/nonconstant activity intersected
# with nonzero_ROI. The latter is the dataset's supplied filter for neurons active
# in its published network-analysis windows. No additional occupancy threshold
# is imposed on the primary fit.
#
# Fixed positive-bin rates are retained only as a sensitivity audit. A positive
# OASIS bin is not a calibrated spike or physiological firing-rate measurement.
#
# Thirty-second blocks are made inside constant-state acquisition segments.
# Three seconds are discarded from both block edges. A lag-wide row-mean
# separator is inserted between retained block interiors; it becomes zero after
# Rastermap row normalization and prevents a lagged comparison from joining two
# unrelated blocks.
#
# The block-orientation control reverses approximately half of the blocks. A
# complete block is reversed for every neuron together, preserving instantaneous
# population vectors and each neuron's within-block autocorrelation. It tests
# whether one temporal direction is consistent across blocks. Its exceedance is
# descriptive because the observed all-forward sequence is not a member of the
# balanced conditional null.
#
# The primary random-order reference is analytic: under a uniformly random
# complete cluster order, Rastermap's weighted objective has expectation equal
# to the mean off-diagonal transferred similarity. Sampled permutations provide
# only its spread, z score, and descriptive exceedance.

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

ANALYSIS_SCHEMA = 2
POSITIVE_BIN_RATE_SENSITIVITY = (0.020, 0.025, 0.100)

N_CLUSTERS = 100
N_PCS = 128
LOCALITY = 0.0
MEAN_TIME = True
PAPER_LAG_SECONDS = 5 / 3.2
FIT_SEED = 0
SPLIT_SEED_BASE = 50_000
FINE_TIE_SEED_BASE = 70_000
CLUSTER_ORDER_SEED_BASE = 80_000
ORIENTATION_SEED_BASE = 1_080_000

BLOCK_SECONDS = 30.0
BLOCK_EDGE_GUARD_SECONDS = 3.0
MIN_NEURONS_FOR_FIT = 256

NEIGHBORHOOD_SIZE = 50
TIE_PERMUTATIONS = 8
N_CLUSTER_ORDER_PERMUTATIONS = 999
N_BLOCK_ORIENTATION_NULLS = 199

REUSE_COMPLETE_RESULTS = True
SHOW_FIGURES = True

FINE_PATH = VALIDATION_DIR / "07_state_specific_fine.csv"
TRANSFER_PATH = VALIDATION_DIR / "07_state_specific_transfer.csv"
SUMMARY_PATH = VALIDATION_DIR / "07_state_specific_summary.csv"
MOUSE_SUMMARY_PATH = VALIDATION_DIR / "07_state_specific_mouse_summary.csv"
RASTERMAP_VERSION = version("rastermap")

EXPECTED_FINE_ROWS_PER_RECORDING = 6
EXPECTED_TRANSFER_ROWS_PER_RECORDING = 12
EXPECTED_SUMMARY_ROWS_PER_RECORDING = 1


# %% [markdown]
# ## Step 2 — lightweight loading and state-matched blocks
#
# Only the deconvolved activity, state vector, acquisition boundaries, and
# supplied nonzero_ROI mask are needed. The helpers below:
#
# - split the original timeline at state changes and microscope breaks;
# - count positive OASIS bins without joining disjoint epochs;
# - tile each state segment into guarded blocks;
# - assign equal numbers of blocks from each state to folds A and B;
# - concatenate blocks with lag-safe separators.


# %%
def load_deconvolution_only(recording_name: str):
    """Load the four arrays needed here without loading raw fluorescence."""
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
    if nonzero_roi.size != activity.shape[0]:
        raise ValueError(f"{recording_name}: nonzero_ROI is not neuron-aligned")
    return path, activity, state, nonzero_roi, boundary_ind


def condition_and_states(recording_name: str):
    """Return condition, mouse, primary codes, and readable state names."""
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


def primary_segments(
    state: np.ndarray,
    boundary_ind: np.ndarray,
    allowed_codes: tuple[float, ...],
) -> list[tuple[int, int, float]]:
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


def positive_counts_in_segments(
    activity: np.ndarray,
    segments: list[tuple[int, int, float]],
    chunk_frames: int = 2048,
) -> tuple[np.ndarray, int]:
    """Count positive bins over explicit original-time segments."""
    counts = np.zeros(activity.shape[0], dtype=np.int64)
    selected_frames = 0
    for start, stop, _code in segments:
        selected_frames += stop - start
        for chunk_start in range(start, stop, chunk_frames):
            chunk_stop = min(stop, chunk_start + chunk_frames)
            counts += np.count_nonzero(
                activity[:, chunk_start:chunk_stop] > 0,
                axis=1,
            )
    if selected_frames == 0:
        raise ValueError("No primary-state frames were found")
    return counts, selected_frames


def guarded_blocks(
    segments: list[tuple[int, int, float]],
    block_frames: int,
    guard_frames: int,
) -> list[tuple[int, int, float]]:
    """Tile constant-state segments and retain guarded block interiors."""
    if 2 * guard_frames >= block_frames:
        raise ValueError("Two edge guards must be shorter than one block")
    blocks = []
    for start, stop, code in segments:
        n_blocks = (stop - start) // block_frames
        for block_index in range(n_blocks):
            original_start = start + block_index * block_frames
            blocks.append(
                (
                    original_start + guard_frames,
                    original_start + block_frames - guard_frames,
                    code,
                )
            )
    return blocks


def matched_fold_blocks(
    blocks: list[tuple[int, int, float]],
    allowed_codes: tuple[float, ...],
    seed: int,
):
    """Allocate equal block counts to folds A/B within each state."""
    rng = np.random.default_rng(seed)
    fold_a = []
    fold_b = []
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


def verify_fold_blocks(
    fold_a: list[tuple[int, int, float]],
    fold_b: list[tuple[int, int, float]],
    allowed_codes: tuple[float, ...],
) -> None:
    """Assert equal state counts and no temporal overlap between folds."""
    for code in allowed_codes:
        count_a = sum(block[2] == code for block in fold_a)
        count_b = sum(block[2] == code for block in fold_b)
        if count_a != count_b or count_a == 0:
            raise RuntimeError("State-matched fold allocation failed")
    for start_a, stop_a, _code_a in fold_a:
        for start_b, stop_b, _code_b in fold_b:
            if max(start_a, start_b) < min(stop_a, stop_b):
                raise RuntimeError("Fold A and B contain overlapping frames")


def matrix_from_blocks(
    selected_activity: np.ndarray,
    blocks: list[tuple[int, int, float]],
    separator_frames: int,
) -> tuple[np.ndarray, np.ndarray, list[slice], np.ndarray]:
    """Concatenate blocks and return activity counts, slices, and state codes."""
    if not blocks:
        raise ValueError("At least one block is required")
    data_frames = sum(stop - start for start, stop, _code in blocks)
    total_frames = data_frames + separator_frames * (len(blocks) - 1)
    matrix = np.empty((selected_activity.shape[0], total_frames), dtype=np.float32)
    positive_counts = np.zeros(selected_activity.shape[0], dtype=np.int64)
    row_sum = np.zeros(selected_activity.shape[0], dtype=np.float64)
    data_slices = []
    separator_slices = []
    block_codes = np.empty(len(blocks), dtype=np.float32)
    position = 0
    for block_index, (start, stop, code) in enumerate(blocks):
        values = selected_activity[:, start:stop]
        width = stop - start
        data_slices.append(slice(position, position + width))
        block_codes[block_index] = code
        matrix[:, position : position + width] = values
        positive_counts += np.count_nonzero(values > 0, axis=1)
        row_sum += np.sum(values, axis=1, dtype=np.float64)
        position += width
        if block_index < len(blocks) - 1 and separator_frames:
            separator_slices.append(slice(position, position + separator_frames))
            position += separator_frames
    row_mean = (row_sum / data_frames).astype(np.float32)
    for seam in separator_slices:
        matrix[:, seam] = row_mean[:, np.newaxis]
    if position != total_frames:
        raise RuntimeError("Block concatenation produced an unexpected width")
    return matrix, positive_counts, data_slices, block_codes


# %% [markdown]
# ## Step 3 — official fits and exact transfer math
#
# Fine comparisons are orientation-free because a one-dimensional map can be
# globally reversed. Coarse transfer keeps the direction learned by Rastermap.
# A source model's neuron PCA basis and sorted cluster templates are evaluated
# on the target model's independently normalized activity. Runtime assertions
# require this formula to reproduce the installed model's own training matrix.


# %%
@dataclass
class FitSummary:
    embedding: np.ndarray
    clusters: np.ndarray
    runtime_seconds: float


def fit_matrix(
    activity: np.ndarray,
    lag_frames: int,
    seed: int,
) -> tuple[Rastermap, FitSummary]:
    """Fit one official Rastermap model to a prescreened population."""
    if activity.shape[0] < MIN_NEURONS_FOR_FIT:
        raise ValueError(
            f"Only {activity.shape[0]} neurons remain; need at least "
            f"{MIN_NEURONS_FOR_FIT}"
        )
    n_pcs = min(N_PCS, activity.shape[0] - 1, activity.shape[1] - 1)
    model = Rastermap(
        n_clusters=N_CLUSTERS,
        n_PCs=n_pcs,
        locality=LOCALITY,
        time_lag_window=lag_frames,
        time_bin=1,
        mean_time=MEAN_TIME,
        bin_size=50,
        random_state=seed,
        keep_norm_X=True,
        verbose=False,
    ).fit(activity, compute_X_embedding=False)
    good = np.asarray(model.igood, dtype=bool).ravel()
    if good.size != activity.shape[0] or not np.all(good):
        raise RuntimeError("A prescreened state fit unexpectedly removed rows")
    summary = FitSummary(
        embedding=np.asarray(model.embedding, dtype=np.float32).ravel().copy(),
        clusters=np.asarray(model.embedding_clust, dtype=np.int32).ravel().copy(),
        runtime_seconds=float(model.runtime),
    )
    return model, summary


def verify_normalized_separators(
    model: Rastermap,
    block_slices: list[slice],
) -> None:
    """Assert lag-wide seams and zero normalized separator columns."""
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


def adjusted_neighborhood_agreement(
    first_embedding: np.ndarray,
    second_embedding: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    """Return raw and finite-population chance-adjusted local overlap."""
    neighborhood = min(NEIGHBORHOOD_SIZE, first_embedding.size - 1)
    raw = rmt.rank_neighborhood_overlap(
        first_embedding,
        second_embedding,
        neighborhood_size=neighborhood,
        tie_permutations=TIE_PERMUTATIONS,
        random_state=seed,
    )
    chance = neighborhood / (first_embedding.size - 1)
    adjusted = (raw - chance) / (1 - chance)
    return raw, adjusted


def tie_fraction(embedding: np.ndarray) -> tuple[int, float]:
    """Return occupied positions and the fraction of rows in tied positions."""
    _positions, counts = np.unique(embedding, return_counts=True)
    return counts.size, float(counts[counts > 1].sum() / embedding.size)


def transferred_temporal_scores(
    source_model: Rastermap,
    target_normalized_activity: np.ndarray,
) -> np.ndarray:
    """Project target activity through the source model's neuron PCA basis."""
    singular_values = np.asarray(source_model.sv, dtype=np.float32)
    source_left = np.asarray(source_model.Usv, dtype=np.float32) / singular_values
    return (target_normalized_activity.T @ source_left) / singular_values


def node_similarity_from_temporal_scores(
    source_model: Rastermap,
    temporal_scores: np.ndarray,
) -> np.ndarray:
    """Apply the installed directed lag similarity to projected target time."""
    return compute_cc_tdelay(
        temporal_scores,
        np.asarray(source_model.U_nodes, dtype=np.float32),
        time_lag_window=int(source_model.time_lag_window),
        symmetric=False,
    )


def transferred_node_similarity(
    source_model: Rastermap,
    target_normalized_activity: np.ndarray,
) -> np.ndarray:
    """Evaluate source cluster templates on target normalized activity."""
    return node_similarity_from_temporal_scores(
        source_model,
        transferred_temporal_scores(source_model, target_normalized_activity),
    )


def directional_objective(
    node_similarity: np.ndarray,
    matching_target: np.ndarray,
    cluster_order: np.ndarray,
) -> float:
    """Score one complete directed cluster order with Rastermap's kernel."""
    similarity = np.asarray(node_similarity, dtype=np.float64)
    target = np.asarray(matching_target, dtype=np.float64)
    order = np.asarray(cluster_order, dtype=np.int64)
    if similarity.shape != target.shape or similarity.ndim != 2:
        raise ValueError("Similarity and target must be aligned square matrices")
    if order.size != similarity.shape[0] or np.unique(order).size != order.size:
        raise ValueError("cluster_order must be a complete permutation")
    ordered = similarity[np.ix_(order, order)]
    weights = np.triu(target, k=1)
    weight_sum = weights.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("Rastermap matching weights must have positive mass")
    return float(np.sum(weights * ordered) / weight_sum)


def activity_cluster_order(
    model: Rastermap,
    neuron_positive_counts: np.ndarray,
) -> np.ndarray:
    """Choose activity-order orientation from source data only."""
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
    ascending_score = directional_objective(model.cc, model.BBt, ascending)
    descending_score = directional_objective(model.cc, model.BBt, descending)
    return ascending if ascending_score >= descending_score else descending


def exact_random_order_expectation(node_similarity: np.ndarray) -> float:
    """Return the exact objective expectation for a uniform cluster order."""
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


# %% [markdown]
# ## Step 4 — balanced block-orientation control
#
# Reversing an entire block for all neurons is an exact time permutation. It
# preserves that block's population vectors, zero-lag covariance, marginal
# distributions, and linear univariate autocovariance at every lag. Choosing a
# different orientation across blocks removes a globally consistent arrow of
# time without erasing calcium smoothness or population synchrony.
#
# Small state folds have few unique balanced choices. The helper enumerates all
# choices when possible and otherwise samples unique choices without
# replacement. The CSV records both the number used and the total available.


# %%
def balanced_orientation_masks(
    block_codes: np.ndarray,
    maximum_masks: int,
    seed: int,
) -> tuple[list[np.ndarray], int]:
    """Return unique masks reversing approximately half of each state's blocks."""
    block_codes = np.asarray(block_codes)
    groups = [np.flatnonzero(block_codes == code) for code in np.unique(block_codes)]
    sizes_by_group = []
    option_counts = []
    for indices in groups:
        n_blocks = indices.size
        if n_blocks < 2:
            raise ValueError("The orientation control needs at least two blocks")
        lower = n_blocks // 2
        sizes = (lower,) if n_blocks % 2 == 0 else (lower, lower + 1)
        sizes_by_group.append(sizes)
        option_counts.append(sum(math.comb(n_blocks, size) for size in sizes))
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
            chosen = rng.choice(indices, size=size, replace=False)
            mask[chosen] = True
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
    """Sum each projected component's within-block lag products."""
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
    """Compute the source nodes' zero-lag similarity on projected target time."""
    node_activity = zscore(
        np.asarray(source_model.U_nodes, dtype=np.float32) @ temporal_scores.T,
        axis=1,
    )
    return node_activity @ node_activity.T / node_activity.shape[1]


def heldout_transfer_metrics(
    source_model: Rastermap,
    target_model: Rastermap,
    source_positive_counts: np.ndarray,
    target_block_slices: list[slice],
    target_block_codes: np.ndarray,
    cluster_seed: int,
    orientation_seed: int,
) -> dict[str, float | int]:
    """Score transfer with source-paired orders and target-paired orientations."""
    temporal_scores = transferred_temporal_scores(source_model, target_model.X)
    transferred = node_similarity_from_temporal_scores(source_model, temporal_scores)
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
    rng = np.random.default_rng(cluster_seed)
    random_scores = np.array(
        [
            directional_objective(
                transferred,
                source_model.BBt,
                rng.permutation(identity),
            )
            for _ in range(N_CLUSTER_ORDER_PERMUTATIONS)
        ],
        dtype=np.float64,
    )
    random_mean = float(random_scores.mean())
    random_sd = float(random_scores.std(ddof=1))
    if not np.isfinite(random_sd) or random_sd <= 0:
        raise RuntimeError("Random cluster-order objective has zero variance")

    orientation_masks, total_orientations = balanced_orientation_masks(
        target_block_codes,
        N_BLOCK_ORIENTATION_NULLS,
        seed=orientation_seed,
    )
    orientation_scores = np.empty(len(orientation_masks), dtype=np.float64)
    reference_autocovariance = block_autocovariance_signature(
        temporal_scores,
        target_block_slices,
        int(source_model.time_lag_window),
    )
    reference_zero_lag = zero_lag_node_similarity(source_model, temporal_scores)
    for repetition, reverse_mask in enumerate(orientation_masks):
        oriented_scores = temporal_scores.copy()
        for reverse, block_slice in zip(
            reverse_mask,
            target_block_slices,
            strict=True,
        ):
            if reverse:
                oriented_scores[block_slice] = temporal_scores[block_slice][::-1]
        if repetition == 0:
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
                    "Block reversal did not preserve within-block autocovariance"
                )
            surrogate_zero_lag = zero_lag_node_similarity(
                source_model,
                oriented_scores,
            )
            if not np.allclose(
                surrogate_zero_lag,
                reference_zero_lag,
                atol=2e-5,
                rtol=2e-5,
            ):
                raise RuntimeError(
                    "Block reversal did not preserve zero-lag similarity"
                )
        oriented_similarity = node_similarity_from_temporal_scores(
            source_model,
            oriented_scores,
        )
        orientation_scores[repetition] = directional_objective(
            oriented_similarity,
            source_model.BBt,
            identity,
        )

    return {
        "learned": learned,
        "reversed": reversed_score,
        "activity": activity_score,
        "learned_minus_activity": learned - activity_score,
        "learned_minus_reversed": learned - reversed_score,
        "random_expectation": random_expectation,
        "learned_minus_random_expectation": learned - random_expectation,
        "sampled_random_mean": random_mean,
        "sampled_random_mean_minus_expectation": (random_mean - random_expectation),
        "sampled_random_sd": random_sd,
        "random_z_using_exact_mean": (learned - random_expectation) / random_sd,
        "cluster_order_permutation_exceedance": (
            1 + np.count_nonzero(random_scores >= learned)
        )
        / (N_CLUSTER_ORDER_PERMUTATIONS + 1),
        "block_orientation_mean": float(orientation_scores.mean()),
        "block_orientation_sd": (
            float(orientation_scores.std(ddof=1))
            if orientation_scores.size > 1
            else np.nan
        ),
        "block_orientation_min": float(orientation_scores.min()),
        "block_orientation_max": float(orientation_scores.max()),
        "learned_minus_block_orientation_mean": (
            learned - float(orientation_scores.mean())
        ),
        "block_orientation_exceedance_descriptive": (
            1 + np.count_nonzero(orientation_scores >= learned)
        )
        / (orientation_scores.size + 1),
        "block_orientation_assignments_used": orientation_scores.size,
        "block_orientation_assignments_possible": total_orientations,
    }


def write_records_to_path(
    path: Path,
    records: list[dict[str, object]],
) -> None:
    """Write one complete table, including an intentionally empty table."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        if records:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def unique_temporary_path(path: Path) -> Path:
    """Reserve a concurrency-safe temporary path beside its destination."""
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    return Path(name)


def checkpoint_tables(
    fine_records: list[dict[str, object]],
    transfer_records: list[dict[str, object]],
    summary_records: list[dict[str, object]],
) -> None:
    """Stage all three tables, then atomically replace each checkpoint file."""
    tables = (
        (FINE_PATH, fine_records),
        (TRANSFER_PATH, transfer_records),
        (SUMMARY_PATH, summary_records),
    )
    staged = []
    try:
        for path, records in tables:
            temporary = unique_temporary_path(path)
            write_records_to_path(temporary, records)
            staged.append((temporary, path))
        for temporary, path in staged:
            os.replace(temporary, path)
    finally:
        for temporary, _path in staged:
            if temporary.exists():
                temporary.unlink()


def save_records(path: Path, records: list[dict[str, object]]) -> None:
    """Atomically save one derived table."""
    temporary = unique_temporary_path(path)
    try:
        write_records_to_path(temporary, records)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_records(path: Path) -> list[dict[str, str]]:
    """Load a CSV checkpoint; empty or absent files contain no rows."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def configuration_record(source_path: Path) -> dict[str, object]:
    """Return exact source and method provenance for checkpoint validation."""
    source_stat = source_path.stat()
    fs = dataio.FS_HZ
    recording_name = source_path.stem
    recording_index = RECORDINGS.index(recording_name)
    cluster_seed_map = "|".join(
        f"{state_index}{fold}:"
        f"{source_model_seed(recording_index, state_index, fold)}"
        for state_index in (0, 1)
        for fold in ("A", "B")
    )
    orientation_seed_map = "|".join(
        f"{state_index}{fold}:"
        f"{target_model_seed(recording_index, state_index, fold)}"
        for state_index in (0, 1)
        for fold in ("A", "B")
    )
    return {
        "analysis_schema": ANALYSIS_SCHEMA,
        "source_size_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "rastermap_version": RASTERMAP_VERSION,
        "recording_index": recording_index,
        "signal_name": "spike_deconv",
        "selection_definition": "finite_nonconstant_and_dataset_nonzero_ROI",
        "validity_scheme": "four_way_state_by_fold_intersection",
        "primary_state_codes": "0|1",
        "positive_bin_rate_sensitivity": "|".join(
            f"{threshold:.3f}" for threshold in POSITIVE_BIN_RATE_SENSITIVITY
        ),
        "fs_hz": fs,
        "block_seconds": BLOCK_SECONDS,
        "block_frames": round(BLOCK_SECONDS * fs),
        "block_edge_guard_seconds": BLOCK_EDGE_GUARD_SECONDS,
        "guard_frames": round(BLOCK_EDGE_GUARD_SECONDS * fs),
        "paper_lag_seconds": PAPER_LAG_SECONDS,
        "lag_frames": round(PAPER_LAG_SECONDS * fs),
        "separator_scheme": "row_mean_lag_width",
        "block_allocation_scheme": "state_matched_random_complete_blocks",
        "n_clusters": N_CLUSTERS,
        "n_pcs": N_PCS,
        "locality": LOCALITY,
        "mean_time": int(MEAN_TIME),
        "time_bin": 1,
        "rastermap_bin_size": 50,
        "keep_normalized_x": 1,
        "compute_x_embedding": 0,
        "minimum_neurons_for_fit": MIN_NEURONS_FOR_FIT,
        "fit_seed": FIT_SEED,
        "split_seed_base": SPLIT_SEED_BASE,
        "split_seed_scheme": "base_plus_recording_index",
        "split_seed_actual": SPLIT_SEED_BASE + recording_index,
        "fine_tie_seed_base": FINE_TIE_SEED_BASE,
        "fine_tie_seed_scheme": "one_common_seed_per_recording",
        "fine_tie_seed_actual": FINE_TIE_SEED_BASE + recording_index,
        "cluster_order_seed_base": CLUSTER_ORDER_SEED_BASE,
        "cluster_order_seed_scheme": "recording_state_fold_source_key",
        "cluster_order_seed_map": cluster_seed_map,
        "orientation_seed_base": ORIENTATION_SEED_BASE,
        "orientation_seed_scheme": "recording_state_fold_target_key",
        "orientation_seed_map": orientation_seed_map,
        "neighborhood_size": NEIGHBORHOOD_SIZE,
        "tie_permutations": TIE_PERMUTATIONS,
        "cluster_order_permutations": N_CLUSTER_ORDER_PERMUTATIONS,
        "random_expectation_scheme": (
            "uniform_complete_order_off_diagonal_similarity_mean"
        ),
        "objective_scheme": "rastermap_BBt_weighted_directed_upper_triangle",
        "activity_order_scheme": "source_cluster_positive_bin_count",
        "block_orientation_nulls": N_BLOCK_ORIENTATION_NULLS,
        "orientation_null_scheme": (
            "unique_balanced_within_target_state_shared_block_reversal"
        ),
        "fine_pair_scheme": "2_within_plus_4_cross_all_fold_pairings",
        "transfer_pair_scheme": (
            "4_same_opposite_fold_plus_8_cross_all_directed_fold_pairings"
        ),
    }


def checkpoint_configuration_matches(
    record: dict[str, object],
    recording_name: str,
) -> bool:
    """Return whether a row exactly matches source data and every setting."""
    expected = configuration_record(dataio.RAW_DIR / f"{recording_name}.mat")
    exact_fields = (
        "analysis_schema",
        "source_size_bytes",
        "source_mtime_ns",
        "rastermap_version",
        "recording_index",
        "signal_name",
        "selection_definition",
        "validity_scheme",
        "primary_state_codes",
        "positive_bin_rate_sensitivity",
        "block_frames",
        "guard_frames",
        "lag_frames",
        "separator_scheme",
        "block_allocation_scheme",
        "n_clusters",
        "n_pcs",
        "mean_time",
        "time_bin",
        "rastermap_bin_size",
        "keep_normalized_x",
        "compute_x_embedding",
        "minimum_neurons_for_fit",
        "fit_seed",
        "split_seed_base",
        "split_seed_scheme",
        "split_seed_actual",
        "fine_tie_seed_base",
        "fine_tie_seed_scheme",
        "fine_tie_seed_actual",
        "cluster_order_seed_base",
        "cluster_order_seed_scheme",
        "cluster_order_seed_map",
        "orientation_seed_base",
        "orientation_seed_scheme",
        "orientation_seed_map",
        "neighborhood_size",
        "tie_permutations",
        "cluster_order_permutations",
        "random_expectation_scheme",
        "objective_scheme",
        "activity_order_scheme",
        "block_orientation_nulls",
        "orientation_null_scheme",
        "fine_pair_scheme",
        "transfer_pair_scheme",
    )
    float_fields = (
        "fs_hz",
        "block_seconds",
        "block_edge_guard_seconds",
        "paper_lag_seconds",
        "locality",
    )
    try:
        return all(
            str(record[field]) == str(expected[field]) for field in exact_fields
        ) and all(
            np.isclose(
                float(record[field]),
                float(expected[field]),
                rtol=0,
                atol=0,
            )
            for field in float_fields
        )
    except (KeyError, TypeError, ValueError):
        return False


def source_model_seed(
    recording_index: int,
    state_index: int,
    fold_label: str,
) -> int:
    """Key cluster-order draws only to the source model identity."""
    fold_index = {"A": 0, "B": 1}[fold_label]
    return (
        CLUSTER_ORDER_SEED_BASE + 100 * recording_index + 10 * state_index + fold_index
    )


def target_model_seed(
    recording_index: int,
    state_index: int,
    fold_label: str,
) -> int:
    """Key orientation masks only to the target model identity."""
    fold_index = {"A": 0, "B": 1}[fold_label]
    return ORIENTATION_SEED_BASE + 100 * recording_index + 10 * state_index + fold_index


# %% [markdown]
# ## Step 5 — stream all recordings and fit state-specific models
#
# The same dataset-filtered active population enters all four candidate matrices:
# finite/nonconstant rows intersected with nonzero_ROI. A four-way validity
# intersection is then applied so every fine-order comparison uses exactly the
# same neurons. This avoids population mismatch but is one reason the analysis is
# transductive: eligibility is informed by both states and both folds.
#
# The transfer table is tidy: one row represents one source-to-target direction.
# Same-state transfer uses disjoint folds. Cross-state transfer includes all
# eight directed state/fold pairings, including same-fold pairs, so every model
# occurs equally often as source and target.
#
# Existing rows are reusable only when source provenance, every method setting,
# and the exact expected row keys agree. Partial or mismatched rows for one
# recording are purged from all three tables before recomputation. Every table
# replacement is atomic; a crash between table replacements is detected as an
# incomplete recording on the next run.

# %%
EXPECTED_FINE_KEYS = {
    ("within", 0.0, "A", 0.0, "B"),
    ("within", 1.0, "A", 1.0, "B"),
    ("cross", 0.0, "A", 1.0, "A"),
    ("cross", 0.0, "A", 1.0, "B"),
    ("cross", 0.0, "B", 1.0, "A"),
    ("cross", 0.0, "B", 1.0, "B"),
}
EXPECTED_TRANSFER_KEYS = {
    ("same", 0.0, "A", 0.0, "B"),
    ("same", 0.0, "B", 0.0, "A"),
    ("same", 1.0, "A", 1.0, "B"),
    ("same", 1.0, "B", 1.0, "A"),
    *{
        ("cross", source_code, source_fold, target_code, target_fold)
        for source_code, target_code in ((0.0, 1.0), (1.0, 0.0))
        for source_fold in ("A", "B")
        for target_fold in ("A", "B")
    },
}


def fine_checkpoint_key(record: dict[str, object]):
    """Return the exact semantic key of one fine-comparison row."""
    return (
        str(record["comparison_type"]),
        float(record["first_state_code"]),
        str(record["first_fold"]),
        float(record["second_state_code"]),
        str(record["second_fold"]),
    )


def transfer_checkpoint_key(record: dict[str, object]):
    """Return the exact semantic key of one directed transfer row."""
    return (
        str(record["comparison_type"]),
        float(record["source_state_code"]),
        str(record["source_fold"]),
        float(record["target_state_code"]),
        str(record["target_fold"]),
    )


fine_records: list[dict[str, object]] = [
    record
    for record in load_records(FINE_PATH)
    if record.get("recording") in RECORDINGS
    and checkpoint_configuration_matches(record, str(record["recording"]))
]
transfer_records: list[dict[str, object]] = [
    record
    for record in load_records(TRANSFER_PATH)
    if record.get("recording") in RECORDINGS
    and checkpoint_configuration_matches(record, str(record["recording"]))
]
summary_records: list[dict[str, object]] = [
    record
    for record in load_records(SUMMARY_PATH)
    if record.get("recording") in RECORDINGS
    and checkpoint_configuration_matches(record, str(record["recording"]))
]

# Persist the schema/provenance filter before any expensive fit. This removes
# provisional-schema rows rather than allowing old and new methods to mix.
checkpoint_tables(fine_records, transfer_records, summary_records)

for recording_index, recording_name in enumerate(RECORDINGS):
    existing_fine = [
        record for record in fine_records if record["recording"] == recording_name
    ]
    existing_transfer = [
        record for record in transfer_records if record["recording"] == recording_name
    ]
    existing_summary = [
        record for record in summary_records if record["recording"] == recording_name
    ]
    can_reuse = (
        REUSE_COMPLETE_RESULTS
        and len(existing_fine) == EXPECTED_FINE_ROWS_PER_RECORDING
        and len(existing_transfer) == EXPECTED_TRANSFER_ROWS_PER_RECORDING
        and len(existing_summary) == EXPECTED_SUMMARY_ROWS_PER_RECORDING
        and {fine_checkpoint_key(record) for record in existing_fine}
        == EXPECTED_FINE_KEYS
        and {transfer_checkpoint_key(record) for record in existing_transfer}
        == EXPECTED_TRANSFER_KEYS
    )
    if can_reuse:
        print(
            f"[{recording_index + 1}/{len(RECORDINGS)}] {recording_name}: "
            "reusing complete checkpoint",
            flush=True,
        )
        continue

    fine_records = [
        record for record in fine_records if record["recording"] != recording_name
    ]
    transfer_records = [
        record for record in transfer_records if record["recording"] != recording_name
    ]
    summary_records = [
        record for record in summary_records if record["recording"] != recording_name
    ]
    checkpoint_tables(fine_records, transfer_records, summary_records)

    print(
        f"\n[{recording_index + 1}/{len(RECORDINGS)}] {recording_name}",
        flush=True,
    )
    condition, mouse_id, allowed_codes, code_labels = condition_and_states(
        recording_name
    )
    source_path, activity, state, nonzero_roi, boundary_ind = load_deconvolution_only(
        recording_name
    )
    session_configuration = configuration_record(source_path)
    fs = dataio.FS_HZ
    n_neurons, n_frames = activity.shape
    lag_frames = round(PAPER_LAG_SECONDS * fs)
    block_frames = round(BLOCK_SECONDS * fs)
    guard_frames = round(BLOCK_EDGE_GUARD_SECONDS * fs)

    segments = primary_segments(state, boundary_ind, allowed_codes)
    positive_counts, primary_state_frames = positive_counts_in_segments(
        activity,
        segments,
    )
    positive_rates = positive_counts.astype(np.float64) * fs / primary_state_frames
    valid_rows = rmt.valid_activity_rows(activity)
    primary_mask = valid_rows & nonzero_roi
    primary_rows = np.flatnonzero(primary_mask)
    sensitivity_counts = {
        threshold: int(np.count_nonzero(primary_mask & (positive_rates >= threshold)))
        for threshold in POSITIVE_BIN_RATE_SENSITIVITY
    }

    blocks = guarded_blocks(segments, block_frames, guard_frames)
    fold_a_blocks, fold_b_blocks, blocks_per_fold = matched_fold_blocks(
        blocks,
        allowed_codes,
        seed=SPLIT_SEED_BASE + recording_index,
    )
    verify_fold_blocks(fold_a_blocks, fold_b_blocks, allowed_codes)
    fold_blocks = {"A": fold_a_blocks, "B": fold_b_blocks}

    selected_activity = np.ascontiguousarray(
        activity[primary_rows],
        dtype=np.float32,
    )
    del activity
    gc.collect()

    matrices = {}
    positive_counts_by_fit = {}
    block_slices_by_fit = {}
    block_codes_by_fit = {}
    validity_masks = {}
    for state_code in allowed_codes:
        for fold_label in ("A", "B"):
            key = (state_code, fold_label)
            state_blocks = [
                block for block in fold_blocks[fold_label] if block[2] == state_code
            ]
            (
                matrices[key],
                positive_counts_by_fit[key],
                block_slices_by_fit[key],
                block_codes_by_fit[key],
            ) = matrix_from_blocks(
                selected_activity,
                state_blocks,
                separator_frames=lag_frames,
            )
            validity_masks[key] = rmt.valid_activity_rows(matrices[key])

    del selected_activity
    common_valid = np.logical_and.reduce(list(validity_masks.values()))
    common_neurons = int(common_valid.sum())
    if common_neurons < MIN_NEURONS_FOR_FIT:
        raise RuntimeError(
            f"{recording_name}: four-way state/fold intersection retains only "
            f"{common_neurons} neurons"
        )
    for key in matrices:
        matrices[key] = np.ascontiguousarray(
            matrices[key][common_valid],
            dtype=np.float32,
        )
        positive_counts_by_fit[key] = positive_counts_by_fit[key][common_valid]
    del validity_masks
    gc.collect()

    print(
        f"  common state/fold population: {common_neurons:,}/"
        f"{primary_rows.size:,} globally selected active neurons",
        flush=True,
    )

    models = {}
    summaries = {}
    for state_code in allowed_codes:
        for fold_label in ("A", "B"):
            key = (state_code, fold_label)
            model, fit_summary = fit_matrix(
                matrices[key],
                lag_frames,
                seed=FIT_SEED,
            )
            verify_normalized_separators(model, block_slices_by_fit[key])
            training_replay = transferred_node_similarity(model, model.X)
            if not np.allclose(
                training_replay,
                model.cc,
                atol=2e-5,
                rtol=2e-5,
            ):
                raise RuntimeError(
                    f"{recording_name} {key}: transfer formula did not replay "
                    "the fitted training similarity"
                )
            models[key] = model
            summaries[key] = fit_summary
            del matrices[key]
            gc.collect()
            print(
                f"  fit {code_labels[state_code]} fold {fold_label}: "
                f"{model.X.shape[1]:,} columns, "
                f"{fit_summary.runtime_seconds:.2f} s",
                flush=True,
            )
    del matrices

    session_fine_records = []
    fine_tie_seed = FINE_TIE_SEED_BASE + recording_index
    fine_pairs = (
        ("within", allowed_codes[0], "A", allowed_codes[0], "B"),
        ("within", allowed_codes[1], "A", allowed_codes[1], "B"),
        ("cross", allowed_codes[0], "A", allowed_codes[1], "A"),
        ("cross", allowed_codes[0], "A", allowed_codes[1], "B"),
        ("cross", allowed_codes[0], "B", allowed_codes[1], "A"),
        ("cross", allowed_codes[0], "B", allowed_codes[1], "B"),
    )
    for (
        comparison_type,
        first_code,
        first_fold,
        second_code,
        second_fold,
    ) in fine_pairs:
        first = summaries[(first_code, first_fold)]
        second = summaries[(second_code, second_fold)]
        local_raw, local_adjusted = adjusted_neighborhood_agreement(
            first.embedding,
            second.embedding,
            seed=fine_tie_seed,
        )
        session_fine_records.append(
            {
                "recording": recording_name,
                "condition": condition,
                "mouse": mouse_id,
                **session_configuration,
                "comparison_type": comparison_type,
                "first_state_code": first_code,
                "first_state_label": code_labels[first_code],
                "first_fold": first_fold,
                "second_state_code": second_code,
                "second_state_label": code_labels[second_code],
                "second_fold": second_fold,
                "common_neurons": common_neurons,
                "fine_tie_seed": fine_tie_seed,
                "abs_spearman": rmt.reversal_invariant_rank_correlation(
                    first.embedding,
                    second.embedding,
                ),
                "local_overlap_raw": local_raw,
                "local_overlap_adjusted": local_adjusted,
                "cluster_ari": adjusted_rand_score(
                    first.clusters,
                    second.clusters,
                ),
            }
        )

    transfer_pairs = (
        ("same", allowed_codes[0], "A", allowed_codes[0], "B"),
        ("same", allowed_codes[0], "B", allowed_codes[0], "A"),
        ("same", allowed_codes[1], "A", allowed_codes[1], "B"),
        ("same", allowed_codes[1], "B", allowed_codes[1], "A"),
        *(
            (
                "cross",
                source_code,
                source_fold,
                target_code,
                target_fold,
            )
            for source_code, target_code in (
                (allowed_codes[0], allowed_codes[1]),
                (allowed_codes[1], allowed_codes[0]),
            )
            for source_fold in ("A", "B")
            for target_fold in ("A", "B")
        ),
    )
    session_transfer_records = []
    state_indices = {code: index for index, code in enumerate(allowed_codes)}
    for (
        comparison_type,
        source_code,
        source_fold,
        target_code,
        target_fold,
    ) in transfer_pairs:
        source_key = (source_code, source_fold)
        target_key = (target_code, target_fold)
        cluster_seed = source_model_seed(
            recording_index,
            state_indices[source_code],
            source_fold,
        )
        orientation_seed = target_model_seed(
            recording_index,
            state_indices[target_code],
            target_fold,
        )
        metrics = heldout_transfer_metrics(
            models[source_key],
            models[target_key],
            positive_counts_by_fit[source_key],
            block_slices_by_fit[target_key],
            block_codes_by_fit[target_key],
            cluster_seed=cluster_seed,
            orientation_seed=orientation_seed,
        )
        record: dict[str, object] = {
            "recording": recording_name,
            "condition": condition,
            "mouse": mouse_id,
            **session_configuration,
            "comparison_type": comparison_type,
            "source_state_code": source_code,
            "source_state_label": code_labels[source_code],
            "source_fold": source_fold,
            "target_state_code": target_code,
            "target_state_label": code_labels[target_code],
            "target_fold": target_fold,
            "common_neurons": common_neurons,
            "source_blocks": blocks_per_fold[source_code],
            "target_blocks": blocks_per_fold[target_code],
            "cluster_order_seed": cluster_seed,
            "block_orientation_seed": orientation_seed,
        }
        record.update(metrics)
        session_transfer_records.append(record)

    fine_within = [
        record
        for record in session_fine_records
        if record["comparison_type"] == "within"
    ]
    fine_cross = [
        record
        for record in session_fine_records
        if record["comparison_type"] == "cross"
    ]
    coarse_same = [
        record
        for record in session_transfer_records
        if record["comparison_type"] == "same"
    ]
    coarse_cross = [
        record
        for record in session_transfer_records
        if record["comparison_type"] == "cross"
    ]

    occupied_positions = []
    tied_fractions = []
    for fit_summary in summaries.values():
        occupied, tied = tie_fraction(fit_summary.embedding)
        occupied_positions.append(occupied)
        tied_fractions.append(tied)

    def record_mean(records, field):
        return float(np.mean([float(record[field]) for record in records]))

    summary_record: dict[str, object] = {
        "recording": recording_name,
        "condition": condition,
        "mouse": mouse_id,
        **session_configuration,
        "recorded_neurons": n_neurons,
        "finite_nonconstant_neurons": int(valid_rows.sum()),
        "dataset_nonzero_roi_neurons": int(nonzero_roi.sum()),
        "primary_active_neurons": primary_rows.size,
        "four_way_common_neurons": common_neurons,
        "common_fraction_of_primary_active": common_neurons / primary_rows.size,
        "primary_state_frames": primary_state_frames,
        "primary_state_duration_seconds": primary_state_frames / fs,
        "sensitivity_selected_at_0p020_positive_bins_per_second": (
            sensitivity_counts[0.020]
        ),
        "sensitivity_selected_at_0p025_positive_bins_per_second": (
            sensitivity_counts[0.025]
        ),
        "sensitivity_selected_at_0p100_positive_bins_per_second": (
            sensitivity_counts[0.100]
        ),
        "state0_code": allowed_codes[0],
        "state0_label": code_labels[allowed_codes[0]],
        "state1_code": allowed_codes[1],
        "state1_label": code_labels[allowed_codes[1]],
        "state0_blocks_per_fold": blocks_per_fold[allowed_codes[0]],
        "state1_blocks_per_fold": blocks_per_fold[allowed_codes[1]],
        "lag_frames": lag_frames,
        "lag_seconds": lag_frames / fs,
        "fit_runtime_seconds_total": float(
            sum(summary.runtime_seconds for summary in summaries.values())
        ),
        "occupied_embedding_positions_mean": float(np.mean(occupied_positions)),
        "fraction_neurons_in_tied_positions_mean": float(np.mean(tied_fractions)),
        "fine_within_abs_spearman_mean": record_mean(
            fine_within,
            "abs_spearman",
        ),
        "fine_cross_abs_spearman_mean": record_mean(
            fine_cross,
            "abs_spearman",
        ),
        "fine_within_minus_cross_abs_spearman": (
            record_mean(fine_within, "abs_spearman")
            - record_mean(fine_cross, "abs_spearman")
        ),
        "fine_within_local_adjusted_mean": record_mean(
            fine_within,
            "local_overlap_adjusted",
        ),
        "fine_cross_local_adjusted_mean": record_mean(
            fine_cross,
            "local_overlap_adjusted",
        ),
        "fine_within_minus_cross_local_adjusted": (
            record_mean(fine_within, "local_overlap_adjusted")
            - record_mean(fine_cross, "local_overlap_adjusted")
        ),
        "fine_within_cluster_ari_mean": record_mean(
            fine_within,
            "cluster_ari",
        ),
        "fine_cross_cluster_ari_mean": record_mean(
            fine_cross,
            "cluster_ari",
        ),
        "coarse_same_learned_minus_random_expectation_mean": record_mean(
            coarse_same,
            "learned_minus_random_expectation",
        ),
        "coarse_cross_learned_minus_random_expectation_mean": record_mean(
            coarse_cross,
            "learned_minus_random_expectation",
        ),
        "coarse_same_minus_cross_learned_minus_random_expectation": (
            record_mean(coarse_same, "learned_minus_random_expectation")
            - record_mean(coarse_cross, "learned_minus_random_expectation")
        ),
        "coarse_same_random_z_using_exact_mean_mean": record_mean(
            coarse_same,
            "random_z_using_exact_mean",
        ),
        "coarse_cross_random_z_using_exact_mean_mean": record_mean(
            coarse_cross,
            "random_z_using_exact_mean",
        ),
        "coarse_same_minus_cross_random_z_using_exact_mean": (
            record_mean(coarse_same, "random_z_using_exact_mean")
            - record_mean(coarse_cross, "random_z_using_exact_mean")
        ),
        "coarse_same_learned_minus_activity_mean": record_mean(
            coarse_same,
            "learned_minus_activity",
        ),
        "coarse_cross_learned_minus_activity_mean": record_mean(
            coarse_cross,
            "learned_minus_activity",
        ),
        "coarse_same_learned_minus_block_orientation_mean": record_mean(
            coarse_same,
            "learned_minus_block_orientation_mean",
        ),
        "coarse_cross_learned_minus_block_orientation_mean": record_mean(
            coarse_cross,
            "learned_minus_block_orientation_mean",
        ),
        "coarse_same_block_orientation_exceedance_mean_descriptive": record_mean(
            coarse_same,
            "block_orientation_exceedance_descriptive",
        ),
        "coarse_cross_block_orientation_exceedance_mean_descriptive": record_mean(
            coarse_cross,
            "block_orientation_exceedance_descriptive",
        ),
    }

    if (
        len(session_fine_records) != EXPECTED_FINE_ROWS_PER_RECORDING
        or {fine_checkpoint_key(record) for record in session_fine_records}
        != EXPECTED_FINE_KEYS
    ):
        raise RuntimeError(f"{recording_name}: incomplete fine-comparison key set")
    if (
        len(session_transfer_records) != EXPECTED_TRANSFER_ROWS_PER_RECORDING
        or {transfer_checkpoint_key(record) for record in session_transfer_records}
        != EXPECTED_TRANSFER_KEYS
    ):
        raise RuntimeError(f"{recording_name}: incomplete transfer key set")

    fine_records.extend(session_fine_records)
    transfer_records.extend(session_transfer_records)
    summary_records.append(summary_record)
    checkpoint_tables(fine_records, transfer_records, summary_records)

    print(
        "  fine within/cross |Spearman|="
        f"{summary_record['fine_within_abs_spearman_mean']:.3f}/"
        f"{summary_record['fine_cross_abs_spearman_mean']:.3f}; "
        "adjusted local="
        f"{summary_record['fine_within_local_adjusted_mean']:.3f}/"
        f"{summary_record['fine_cross_local_adjusted_mean']:.3f}",
        flush=True,
    )
    print(
        "  coarse learned−exact-random same/cross="
        f"{summary_record['coarse_same_learned_minus_random_expectation_mean']:.4f}/"
        f"{summary_record['coarse_cross_learned_minus_random_expectation_mean']:.4f}; "
        "secondary random-z="
        f"{summary_record['coarse_same_random_z_using_exact_mean_mean']:.2f}/"
        f"{summary_record['coarse_cross_random_z_using_exact_mean_mean']:.2f}; "
        "learned−orientation-null="
        f"{summary_record['coarse_same_learned_minus_block_orientation_mean']:.4f}/"
        f"{summary_record['coarse_cross_learned_minus_block_orientation_mean']:.4f}",
        flush=True,
    )

    del (
        models,
        summaries,
        positive_counts_by_fit,
        block_slices_by_fit,
        block_codes_by_fit,
        state,
        nonzero_roi,
        positive_counts,
        positive_rates,
        valid_rows,
        primary_mask,
        common_valid,
    )
    gc.collect()

for recording_name in RECORDINGS:
    recording_fine = [
        record for record in fine_records if record["recording"] == recording_name
    ]
    recording_transfer = [
        record for record in transfer_records if record["recording"] == recording_name
    ]
    recording_summary = [
        record for record in summary_records if record["recording"] == recording_name
    ]
    if (
        len(recording_fine) != EXPECTED_FINE_ROWS_PER_RECORDING
        or {fine_checkpoint_key(record) for record in recording_fine}
        != EXPECTED_FINE_KEYS
        or len(recording_transfer) != EXPECTED_TRANSFER_ROWS_PER_RECORDING
        or {transfer_checkpoint_key(record) for record in recording_transfer}
        != EXPECTED_TRANSFER_KEYS
        or len(recording_summary) != EXPECTED_SUMMARY_ROWS_PER_RECORDING
    ):
        raise RuntimeError(f"{recording_name}: final checkpoint is incomplete")

fine_records.sort(
    key=lambda record: (
        RECORDINGS.index(str(record["recording"])),
        fine_checkpoint_key(record),
    )
)
transfer_records.sort(
    key=lambda record: (
        RECORDINGS.index(str(record["recording"])),
        transfer_checkpoint_key(record),
    )
)
summary_records.sort(key=lambda record: RECORDINGS.index(str(record["recording"])))
checkpoint_tables(fine_records, transfer_records, summary_records)

print("saved ->", FINE_PATH)
print("saved ->", TRANSFER_PATH)
print("saved ->", SUMMARY_PATH)


# %% [markdown]
# ## Step 6 — mouse-level descriptive aggregation
#
# Mouse 4 contributes two sleep days. Session-level points remain visible in the
# figures, while this table first averages those two days so one mouse does not
# receive double weight in condition summaries. No p-values are calculated:
# there are only five sleep mice and four anesthesia mice.

# %%
mouse_summary_fields = (
    "common_fraction_of_primary_active",
    "fine_within_abs_spearman_mean",
    "fine_cross_abs_spearman_mean",
    "fine_within_minus_cross_abs_spearman",
    "fine_within_local_adjusted_mean",
    "fine_cross_local_adjusted_mean",
    "fine_within_minus_cross_local_adjusted",
    "coarse_same_learned_minus_random_expectation_mean",
    "coarse_cross_learned_minus_random_expectation_mean",
    "coarse_same_minus_cross_learned_minus_random_expectation",
    "coarse_same_random_z_using_exact_mean_mean",
    "coarse_cross_random_z_using_exact_mean_mean",
    "coarse_same_minus_cross_random_z_using_exact_mean",
    "coarse_same_learned_minus_activity_mean",
    "coarse_cross_learned_minus_activity_mean",
    "coarse_same_learned_minus_block_orientation_mean",
    "coarse_cross_learned_minus_block_orientation_mean",
)

mouse_summary_records = []
for condition in ("sleep", "anesthesia"):
    condition_records = [
        record for record in summary_records if record["condition"] == condition
    ]
    for mouse in sorted({str(record["mouse"]) for record in condition_records}):
        records = [record for record in condition_records if record["mouse"] == mouse]
        mouse_record: dict[str, object] = {
            "condition": condition,
            "mouse": mouse,
            "n_sessions": len(records),
        }
        for field in mouse_summary_fields:
            mouse_record[field] = float(
                np.mean([float(record[field]) for record in records])
            )
        mouse_summary_records.append(mouse_record)

save_records(MOUSE_SUMMARY_PATH, mouse_summary_records)
print("saved ->", MOUSE_SUMMARY_PATH)


# %% [markdown]
# ## Step 7 — selection and fine-order figure
#
# A within-state advantage would place the filled markers above the cross-state
# markers. Fine-order statistics remain tie-aware; a large global rank
# correlation without adjusted local overlap should not be interpreted as
# reproducible neighboring neurons.

# %%
session_x = np.arange(len(summary_records))
session_labels = [
    str(record["recording"]).replace("mouse", "m") for record in summary_records
]
condition_colors = [
    "tab:blue" if record["condition"] == "sleep" else "tab:orange"
    for record in summary_records
]

fine_figure, fine_axes = plt.subplots(
    1,
    3,
    figsize=(18, 5.5),
    constrained_layout=True,
)

primary_counts = np.array(
    [record["primary_active_neurons"] for record in summary_records],
    dtype=float,
)
common_counts = np.array(
    [record["four_way_common_neurons"] for record in summary_records],
    dtype=float,
)
fine_axes[0].bar(
    session_x,
    primary_counts,
    color=condition_colors,
    alpha=0.25,
    label="finite/nonconstant ∩ dataset nonzero_ROI",
)
fine_axes[0].bar(
    session_x,
    common_counts,
    color=condition_colors,
    alpha=0.9,
    label="valid in all 4 state/fold matrices",
)
fine_axes[0].set_yscale("log")
fine_axes[0].set_ylabel("neurons (log scale)")
fine_axes[0].set_title("One common population is used for every comparison")
fine_axes[0].legend(frameon=False, fontsize=8)

for axis, within_field, cross_field, ylabel, title in (
    (
        fine_axes[1],
        "fine_within_abs_spearman_mean",
        "fine_cross_abs_spearman_mean",
        "reversal-invariant |Spearman|",
        "Fine global order: within versus across states",
    ),
    (
        fine_axes[2],
        "fine_within_local_adjusted_mean",
        "fine_cross_local_adjusted_mean",
        "chance-adjusted local overlap",
        "Fine local neighborhoods: within versus across states",
    ),
):
    within_values = np.array(
        [record[within_field] for record in summary_records],
        dtype=float,
    )
    cross_values = np.array(
        [record[cross_field] for record in summary_records],
        dtype=float,
    )
    for index, color in enumerate(condition_colors):
        axis.plot(
            (index - 0.12, index + 0.12),
            (within_values[index], cross_values[index]),
            color=color,
            alpha=0.45,
            lw=1,
        )
    axis.scatter(
        session_x - 0.12,
        within_values,
        c=condition_colors,
        marker="o",
        label="within-state A↔B",
    )
    axis.scatter(
        session_x + 0.12,
        cross_values,
        facecolors="white",
        edgecolors=condition_colors,
        marker="o",
        label="cross-state, all 4 fold pairings",
    )
    axis.axhline(0, color="0.65", lw=0.8)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend(frameon=False, fontsize=8)

for axis in fine_axes:
    axis.set_xticks(session_x, session_labels, rotation=45, ha="right")
fine_figure.suptitle(
    "State-specific Rastermap fits · conditional four-way common active population"
)
fine_path = FIG_DIR / "07_rastermap_01_state_specific_fine.png"
fine_figure.savefig(fine_path, dpi=160, bbox_inches="tight")
print("saved ->", fine_path)


# %% [markdown]
# ## Step 8 — coarse transfer and temporal-direction figure
#
# The first panel uses the exact, Monte-Carlo-free random-order expectation in
# raw objective units. Sampled permutation SD and z remain secondary CSV and
# printed diagnostics. Learned-minus-activity asks whether Rastermap improves
# over a simple source activity rank. Learned-minus-orientation-null asks whether
# one direction is more consistent than balanced mixtures of forward and
# reversed blocks.

# %%
coarse_figure, coarse_axes = plt.subplots(
    1,
    3,
    figsize=(18, 5.5),
    constrained_layout=True,
)

for axis, same_field, cross_field, ylabel, title in (
    (
        coarse_axes[0],
        "coarse_same_learned_minus_random_expectation_mean",
        "coarse_cross_learned_minus_random_expectation_mean",
        "learned − exact random-order expectation",
        "Coarse order above exact random expectation",
    ),
    (
        coarse_axes[1],
        "coarse_same_learned_minus_activity_mean",
        "coarse_cross_learned_minus_activity_mean",
        "learned − activity-ranked objective",
        "Rastermap gain over source activity rank",
    ),
    (
        coarse_axes[2],
        "coarse_same_learned_minus_block_orientation_mean",
        "coarse_cross_learned_minus_block_orientation_mean",
        "learned − balanced block-orientation mean",
        "Consistent direction beyond autocorrelation-preserving null",
    ),
):
    same_values = np.array(
        [record[same_field] for record in summary_records],
        dtype=float,
    )
    cross_values = np.array(
        [record[cross_field] for record in summary_records],
        dtype=float,
    )
    for index, color in enumerate(condition_colors):
        axis.plot(
            (index - 0.12, index + 0.12),
            (same_values[index], cross_values[index]),
            color=color,
            alpha=0.45,
            lw=1,
        )
    axis.scatter(
        session_x - 0.12,
        same_values,
        c=condition_colors,
        marker="s",
        label="same-state transfer",
    )
    axis.scatter(
        session_x + 0.12,
        cross_values,
        facecolors="white",
        edgecolors=condition_colors,
        marker="s",
        label="cross-state transfer",
    )
    axis.axhline(0, color="0.45", ls="--", lw=0.9)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend(frameon=False, fontsize=8)
    axis.set_xticks(session_x, session_labels, rotation=45, ha="right")

coarse_figure.suptitle(
    "Exact held-out cluster-template transfer · positive same−cross supports "
    "state specificity"
)
coarse_path = FIG_DIR / "07_rastermap_02_state_specific_coarse.png"
coarse_figure.savefig(coarse_path, dpi=160, bbox_inches="tight")
print("saved ->", coarse_path)


# %% [markdown]
# ## Step 9 — concise numerical assessment
#
# These summaries are descriptive. In particular, balanced-orientation
# exceedances are not p-values, and adjacent 30-s blocks from one long bout can
# enter opposite folds. The 3-s guards protect the fitted lag window but do not
# eliminate slow calcium autocorrelation or drift.


# %%
def median_and_range(values):
    """Format one compact cross-recording descriptive summary."""
    values = np.asarray(values, dtype=float)
    return (
        float(np.median(values)),
        float(np.min(values)),
        float(np.max(values)),
    )


for label, field in (
    ("fine within−cross |Spearman|", "fine_within_minus_cross_abs_spearman"),
    (
        "fine within−cross adjusted local",
        "fine_within_minus_cross_local_adjusted",
    ),
    (
        "coarse same−cross learned−exact-random",
        "coarse_same_minus_cross_learned_minus_random_expectation",
    ),
    (
        "secondary coarse same−cross random-z",
        "coarse_same_minus_cross_random_z_using_exact_mean",
    ),
    (
        "same-state learned−orientation null",
        "coarse_same_learned_minus_block_orientation_mean",
    ),
    (
        "cross-state learned−orientation null",
        "coarse_cross_learned_minus_block_orientation_mean",
    ),
):
    median, minimum, maximum = median_and_range(
        [record[field] for record in summary_records]
    )
    print(f"{label}: median={median:.4f}, range={minimum:.4f}…{maximum:.4f}")

print(
    "\nInterpretation guide: a positive within−cross contrast supports "
    "state-specific organization; similarly strong within and cross transfer "
    "supports a state-general organization. Weak values in both cases do not "
    "support a reproducible order. This screen does not test REM, prospective "
    "neuron selection, bout-level independence, or anatomical localization."
)

if SHOW_FIGURES:
    plt.show()
