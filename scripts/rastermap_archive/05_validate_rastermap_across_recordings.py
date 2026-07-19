# %% [markdown]
# # 05 · Does active-neuron Rastermap robustness recur across recordings?
#
# Tutorial 03 verifies one official fit and tutorial 04 stress-tests one sleep
# recording in depth. This tutorial addresses the strongest remaining gap: a
# result from one session is not a project-level verification. It streams all six
# sleep and four anesthesia recordings, never samples neurons randomly, and
# compares disjoint-time fits within each recording.
#
# This tutorial intentionally retains the original 0.020-positive-bin/s support
# floor as a stricter sensitivity screen. Tutorials 02, 06, and 07 use the more
# defensible primary population instead: finite/nonconstant rows intersected
# with the dataset-supplied `nonzero_ROI` mask. Results from 05 and 06 therefore
# differ in both resampling depth and neuron eligibility.
#
# The screen answers three deliberately separate questions:
#
# 1. How strongly does the definition of an "active neuron" change the retained
#    population across recordings?
# 2. Does the fine neuron-by-neuron order recur across random seeds and matched
#    time folds?
# 3. Does Rastermap's coarser, directional cluster-template objective transfer
#    to held-out time better than reversal, activity rank, and random orders?
#
# This is descriptive cross-recording verification. Mouse 4 contributes two sleep days, and
# sleep and anesthesia have only five and four mice respectively. No neuron is
# shared across animals, so embeddings are never correlated between mice.

# %% Step 0 — imports
import csv
import gc
import os
import sys
from dataclasses import dataclass
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
# ## Step 1 — settings and what they mean
#
# Stringer et al. report minimum *firing rates* of 0.1--0.25 Hz in their
# applications. These recordings provide OASIS deconvolution values, not
# calibrated spike times, so a positive-bin rate is not the same quantity.
# Applying a literal 0.10-positive-bin/s proxy is especially severe in the
# anesthesia sessions; the selection-audit figure below shows that directly.
#
# The working population therefore combines two explicit safeguards:
#
# - `nonzero_ROI`, the dataset's supplied mask for neurons active in its network
#   analysis windows;
# - at least `PRIMARY_MIN_POSITIVE_BIN_RATE` positive OASIS bins per second over
#   the two primary states (awake + NREM, or awake + anesthesia).
#
# The 0.020 threshold is a numerical-support floor, not a claim about firing
# rate. `SELECTION_RATE_GRID` makes its arbitrariness visible. Every neuron that
# passes is retained; there is no fixed-N or random subsample.
#
# Time validation uses nonoverlapping 30-s blocks. Three seconds are removed
# from both edges, blocks are split equally between folds *within each state*,
# and a lag-length row-mean separator is inserted at every concatenation seam.
# After Rastermap's row centering those separator columns are zero, so a lagged
# similarity cannot jump directly from one original block into another.

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

PRIMARY_MIN_POSITIVE_BIN_RATE = 0.020
STRICT_MIN_POSITIVE_BIN_RATE = 0.025
SELECTION_RATE_GRID = (0.020, 0.025, 0.050, 0.100, 0.150, 0.250)

N_CLUSTERS = 100
N_PCS = 128
LOCALITY = 0.0
MEAN_TIME = True
PAPER_LAG_SECONDS = 5 / 3.2
FULL_SEEDS = (0, 1)

BLOCK_SECONDS = 30.0
BLOCK_EDGE_GUARD_SECONDS = 3.0
NEIGHBORHOOD_SIZE = 50
TIE_PERMUTATIONS = 8
N_OBJECTIVE_PERMUTATIONS = 999
N_TEMPORAL_ORDER_SHUFFLES = 19
MIN_NEURONS_FOR_FIT = 256
SHOW_FIGURES = True


# %% [markdown]
# ## Step 2 — lightweight loading and block helpers
#
# `dataio.load_recording` intentionally loads all three large activity matrices.
# This screen needs only `spike_deconv`, state, boundaries, and `nonzero_ROI`, so
# `load_deconvolution_only` reads those HDF5 datasets directly and converts the
# MATLAB time-by-neuron array to neuron-by-time. This keeps peak memory near one
# float32 signal matrix rather than all raw signals.
#
# The other helpers have narrow jobs:
#
# - `primary_segments` splits at both state changes and microscope breaks.
# - `positive_counts_in_segments` computes activity support without joining
#   disjoint epochs.
# - `guarded_blocks` tiles each constant-state segment and removes edge guards.
# - `matched_fold_blocks` gives A and B the same number of blocks from each
#   state, then restores chronological order inside each fold.
# - `matrix_from_blocks` inserts the zero-after-centering seam separators.


# %%
def load_deconvolution_only(recording_name: str):
    """Load only arrays needed by this screen from one v7.3 MAT/HDF5 file."""
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
    """Return condition, mouse ID, primary codes, and readable state labels."""
    mouse = recording_name.split("_")[0]
    if recording_name.endswith("_sleep"):
        return "sleep", mouse, (0.0, 1.0), {0.0: "awake", 1.0: "nrem"}
    if recording_name.endswith("_ane"):
        return (
            "anesthesia",
            mouse,
            (0.0, 1.0),
            {
                0.0: "awake",
                1.0: "anesthesia",
            },
        )
    raise ValueError(f"Cannot infer recording condition from {recording_name!r}")


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
    """Tile constant-state segments and retain only guarded block interiors."""
    if 2 * guard_frames >= block_frames:
        raise ValueError("Two edge guards must be shorter than one block")
    blocks = []
    for start, stop, code in segments:
        n_blocks = (stop - start) // block_frames
        for block_index in range(n_blocks):
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
    blocks: list[tuple[int, int, float]],
    allowed_codes: tuple[float, ...],
    seed: int,
):
    """Allocate equal block counts to folds A/B separately within each state."""
    rng = np.random.default_rng(seed)
    fold_a = []
    fold_b = []
    used = []
    counts_by_code = {}
    for code in allowed_codes:
        state_blocks = [block for block in blocks if block[2] == code]
        order = rng.permutation(len(state_blocks))
        usable = 2 * (len(state_blocks) // 2)
        if usable < 2:
            raise ValueError(f"State {code:g} has fewer than two complete blocks")
        half = usable // 2
        chosen_a = [state_blocks[index] for index in order[:half]]
        chosen_b = [state_blocks[index] for index in order[half:usable]]
        fold_a.extend(chosen_a)
        fold_b.extend(chosen_b)
        used.extend(chosen_a + chosen_b)
        counts_by_code[code] = half
    return (
        sorted(fold_a),
        sorted(fold_b),
        sorted(used),
        counts_by_code,
    )


def matrix_from_blocks(
    selected_activity: np.ndarray,
    blocks: list[tuple[int, int, float]],
    separator_frames: int,
) -> tuple[np.ndarray, np.ndarray, list[slice]]:
    """Concatenate blocks and return counts plus each data block's output slice."""
    if not blocks:
        raise ValueError("At least one block is required")
    data_frames = sum(stop - start for start, stop, _code in blocks)
    total_frames = data_frames + separator_frames * (len(blocks) - 1)
    matrix = np.empty((selected_activity.shape[0], total_frames), dtype=np.float32)
    positive_counts = np.zeros(selected_activity.shape[0], dtype=np.int64)
    row_sum = np.zeros(selected_activity.shape[0], dtype=np.float64)
    separator_slices = []
    data_slices = []
    position = 0
    for block_index, (start, stop, _code) in enumerate(blocks):
        values = selected_activity[:, start:stop]
        width = stop - start
        data_slices.append(slice(position, position + width))
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
    return matrix, positive_counts, data_slices


# %% [markdown]
# ## Step 3 — compact fit and comparison helpers
#
# Seed and fine-order comparisons are orientation-free because a one-dimensional
# map can be globally reversed. The lagged cluster score is different: with a
# nonzero time lag Rastermap's upper-triangle objective is directional, so the
# learned orientation and its reversal are scored separately.
#
# `transferred_node_similarity` evaluates cluster templates learned in one fold
# on the other fold's independently normalized activity. `directional_objective`
# then applies the installed model's own `BBt` matching kernel. The permutation
# exceedance treats a complete cluster order—not individual cluster pairs—as one
# randomization unit.


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
    keep_normalized: bool,
) -> tuple[Rastermap, FitSummary]:
    """Fit one official Rastermap model to a prescreened active population."""
    if activity.shape[0] < MIN_NEURONS_FOR_FIT:
        raise ValueError(
            f"Only {activity.shape[0]} neurons remain; need at least "
            f"{MIN_NEURONS_FOR_FIT} for this 100-cluster screen"
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
        keep_norm_X=keep_normalized,
        verbose=False,
    ).fit(activity, compute_X_embedding=False)
    good = np.asarray(model.igood, dtype=bool).ravel()
    if good.size != activity.shape[0] or not np.all(good):
        raise RuntimeError(
            "A prescreened fit unexpectedly removed rows; inspect fold-specific QC"
        )
    summary = FitSummary(
        embedding=np.asarray(model.embedding, dtype=np.float32).ravel().copy(),
        clusters=np.asarray(model.embedding_clust, dtype=np.int32).ravel().copy(),
        runtime_seconds=float(model.runtime),
    )
    return model, summary


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
    """Return occupied positions and fraction of neurons in tied positions."""
    _positions, counts = np.unique(embedding, return_counts=True)
    return counts.size, float(counts[counts > 1].sum() / embedding.size)


def verify_normalized_separators(
    model: Rastermap,
    block_slices: list[slice],
) -> None:
    """Assert that seams are lag-wide and exactly zero after normalization."""
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


def transferred_node_similarity(
    source_model: Rastermap,
    target_normalized_activity: np.ndarray,
) -> np.ndarray:
    """Evaluate source-fit sorted cluster templates on untouched target time."""
    temporal_scores = transferred_temporal_scores(
        source_model,
        target_normalized_activity,
    )
    return node_similarity_from_temporal_scores(source_model, temporal_scores)


def transferred_temporal_scores(
    source_model: Rastermap,
    target_normalized_activity: np.ndarray,
) -> np.ndarray:
    """Project held-out normalized activity through a source fit's PCA basis."""
    singular_values = np.asarray(source_model.sv, dtype=np.float32)
    source_left = np.asarray(source_model.Usv, dtype=np.float32) / singular_values
    return (target_normalized_activity.T @ source_left) / singular_values


def node_similarity_from_temporal_scores(
    source_model: Rastermap,
    temporal_scores: np.ndarray,
) -> np.ndarray:
    """Apply the installed directed lag similarity to projected held-out time."""
    return compute_cc_tdelay(
        temporal_scores,
        np.asarray(source_model.U_nodes, dtype=np.float32),
        time_lag_window=int(source_model.time_lag_window),
        symmetric=False,
    )


def lag_component_similarities(
    source_model: Rastermap,
    temporal_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact lag-0 and delayed-only (lags 1…L) node similarities."""
    node_activity = zscore(
        np.asarray(source_model.U_nodes, dtype=np.float32) @ temporal_scores.T,
        axis=1,
    )
    n_timepoints = node_activity.shape[1]
    zero_lag = node_activity @ node_activity.T / n_timepoints
    maximum_lag = int(source_model.time_lag_window)
    if maximum_lag < 1:
        return zero_lag, np.full_like(zero_lag, np.nan)
    delayed = np.stack(
        [
            node_activity[:, lag:] @ node_activity[:, :-lag].T / (n_timepoints - lag)
            for lag in range(1, maximum_lag + 1)
        ],
        axis=-1,
    ).max(axis=-1)
    return zero_lag, delayed


def directional_objective(
    node_similarity: np.ndarray,
    matching_target: np.ndarray,
    cluster_order: np.ndarray,
) -> float:
    """Score one complete directed cluster order using Rastermap's kernel."""
    similarity = np.asarray(node_similarity, dtype=np.float64)
    target = np.asarray(matching_target, dtype=np.float64)
    order = np.asarray(cluster_order, dtype=np.int64)
    if similarity.shape != target.shape or similarity.ndim != 2:
        raise ValueError("Similarity and target must be aligned square matrices")
    if order.size != similarity.shape[0] or np.unique(order).size != order.size:
        raise ValueError("cluster_order must be a complete permutation")
    ordered = similarity[np.ix_(order, order)]
    weights = np.triu(target, k=1)
    return float(np.sum(weights * ordered) / weights.sum())


def activity_cluster_order(
    model: Rastermap,
    neuron_positive_counts: np.ndarray,
) -> np.ndarray:
    """Choose activity-order orientation on training data, never held-out data."""
    assignments = np.asarray(model.embedding_clust, dtype=np.int64)
    n_clusters = np.asarray(model.U_nodes).shape[0]
    cluster_activity = np.array(
        [
            neuron_positive_counts[assignments == cluster].mean()
            for cluster in range(n_clusters)
        ]
    )
    ascending = np.argsort(cluster_activity, kind="stable")
    descending = ascending[::-1]
    ascending_score = directional_objective(model.cc, model.BBt, ascending)
    descending_score = directional_objective(model.cc, model.BBt, descending)
    return ascending if ascending_score >= descending_score else descending


def heldout_objective_record(
    source_model: Rastermap,
    target_model: Rastermap,
    source_positive_counts: np.ndarray,
    target_block_slices: list[slice],
    seed: int,
) -> dict[str, float]:
    """Compare learned order with cluster and held-out temporal-order nulls."""
    temporal_scores = transferred_temporal_scores(source_model, target_model.X)
    transferred = node_similarity_from_temporal_scores(source_model, temporal_scores)
    zero_lag_similarity, delayed_similarity = lag_component_similarities(
        source_model,
        temporal_scores,
    )
    if not np.allclose(
        transferred,
        np.maximum(zero_lag_similarity, delayed_similarity),
        atol=2e-5,
        rtol=2e-5,
    ):
        raise RuntimeError("Lag-0/delayed decomposition does not replay Rastermap")
    identity = np.arange(source_model.U_nodes.shape[0], dtype=np.int64)
    learned = directional_objective(transferred, source_model.BBt, identity)
    zero_lag_score = directional_objective(
        zero_lag_similarity,
        source_model.BBt,
        identity,
    )
    delayed_only_score = directional_objective(
        delayed_similarity,
        source_model.BBt,
        identity,
    )
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
    rng = np.random.default_rng(seed)
    random_scores = np.array(
        [
            directional_objective(
                transferred,
                source_model.BBt,
                rng.permutation(identity),
            )
            for _ in range(N_OBJECTIVE_PERMUTATIONS)
        ]
    )
    random_sd = float(random_scores.std(ddof=1))
    reversed_temporal_scores = temporal_scores.copy()
    for block_slice in target_block_slices:
        reversed_temporal_scores[block_slice] = temporal_scores[block_slice][::-1]
    reversed_time_similarity = node_similarity_from_temporal_scores(
        source_model,
        reversed_temporal_scores,
    )
    reversed_time_score = directional_objective(
        reversed_time_similarity,
        source_model.BBt,
        identity,
    )
    # A shared permutation retains every instantaneous population vector and
    # therefore zero-lag covariance, while destroying positive-lag order within
    # each guarded block. Separator columns remain untouched.
    temporal_shuffle_scores = np.empty(N_TEMPORAL_ORDER_SHUFFLES, dtype=float)
    for repetition in range(N_TEMPORAL_ORDER_SHUFFLES):
        shuffled_scores = temporal_scores.copy()
        for block_slice in target_block_slices:
            block_order = rng.permutation(block_slice.stop - block_slice.start)
            shuffled_scores[block_slice] = temporal_scores[block_slice][block_order]
        shuffled_similarity = node_similarity_from_temporal_scores(
            source_model,
            shuffled_scores,
        )
        if repetition == 0:
            shuffled_zero_lag, _shuffled_delayed = lag_component_similarities(
                source_model,
                shuffled_scores,
            )
            if not np.allclose(
                shuffled_zero_lag,
                zero_lag_similarity,
                atol=2e-5,
                rtol=2e-5,
            ):
                raise RuntimeError("Temporal shuffle did not preserve lag-0 similarity")
        temporal_shuffle_scores[repetition] = directional_objective(
            shuffled_similarity,
            source_model.BBt,
            identity,
        )
    return {
        "learned": learned,
        "reversed": reversed_score,
        "activity": activity_score,
        "random_mean": float(random_scores.mean()),
        "random_sd": random_sd,
        "random_z": (
            (learned - float(random_scores.mean())) / random_sd
            if random_sd > 0
            else np.nan
        ),
        "permutation_exceedance": (1 + np.count_nonzero(random_scores >= learned))
        / (N_OBJECTIVE_PERMUTATIONS + 1),
        "zero_lag": zero_lag_score,
        "delayed_only_lags_1_to_L": delayed_only_score,
        "blockwise_time_reversal": reversed_time_score,
        "temporal_order_shuffle_mean": float(temporal_shuffle_scores.mean()),
        "temporal_order_shuffle_min": float(temporal_shuffle_scores.min()),
        "temporal_order_shuffle_max": float(temporal_shuffle_scores.max()),
        "temporal_order_shuffle_exceedance_descriptive": (
            1 + np.count_nonzero(temporal_shuffle_scores >= learned)
        )
        / (N_TEMPORAL_ORDER_SHUFFLES + 1),
    }


def matrix_energy(activity: np.ndarray, chunk_frames: int = 512) -> float:
    """Accumulate squared matrix energy without a full float64 temporary."""
    total = 0.0
    for start in range(0, activity.shape[1], chunk_frames):
        total += float(
            np.sum(
                np.square(activity[:, start : start + chunk_frames]),
                dtype=np.float64,
            )
        )
    return total


def pca_transfer_efficiencies(
    first_model: Rastermap,
    second_model: Rastermap,
) -> tuple[float, float, float]:
    """Return adjusted subspace overlap and reciprocal held-out PCA efficiency."""
    first_left = first_model.Usv / first_model.sv
    second_left = second_model.Usv / second_model.sv
    n_components = min(first_left.shape[1], second_left.shape[1])
    first_left = first_left[:, :n_components]
    second_left = second_left[:, :n_components]
    overlap = float(
        np.linalg.norm(first_left.T @ second_left, ord="fro") ** 2 / n_components
    )
    chance = n_components / first_left.shape[0]
    adjusted_overlap = (overlap - chance) / (1 - chance)

    first_total = matrix_energy(first_model.X)
    second_total = matrix_energy(second_model.X)
    first_optimal = float(np.sum(first_model.sv[:n_components] ** 2)) / first_total
    second_optimal = float(np.sum(second_model.sv[:n_components] ** 2)) / second_total
    first_to_second = float(
        np.sum((first_left.T @ second_model.X) ** 2, dtype=np.float64)
        / second_total
        / second_optimal
    )
    second_to_first = float(
        np.sum((second_left.T @ first_model.X) ** 2, dtype=np.float64)
        / first_total
        / first_optimal
    )
    return adjusted_overlap, first_to_second, second_to_first


def lag_pair_fraction_crossing_boundaries(
    n_timepoints: int,
    boundaries: np.ndarray,
    maximum_lag: int,
) -> float:
    """Return the fraction of original lag pairs crossing acquisition breaks."""
    splits = np.asarray(boundaries, dtype=np.int64).ravel() + 1
    splits = splits[(splits > 0) & (splits < n_timepoints)]
    crossing = 0
    total = 0
    for lag in range(maximum_lag + 1):
        total += n_timepoints - lag
        if lag:
            for split in splits:
                crossing += max(
                    0,
                    min(split, n_timepoints - lag) - max(0, split - lag),
                )
    return crossing / total if total else 0.0


def save_records(path: Path, records: list[dict[str, object]]) -> None:
    """Write same-schema records after each streamed session."""
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


# %% [markdown]
# ## Step 4 — stream all ten recordings and run 50 official fits
#
# Each session receives two seeds on all matched usable guarded blocks, a third
# full-block fit at the stricter 0.025 support floor, and one fit per independent
# fold. Selection is fixed before fold allocation. A
# fold-specific validity intersection is applied once to both fold matrices, so
# transfer never compares different neuron sets. This makes the test conditional
# (and transductive), not prospective: whole-session activity and both folds
# influence which neurons are eligible. Results are checkpointed after every
# recording.

# %%
selection_records: list[dict[str, object]] = []
fit_records: list[dict[str, object]] = []

for recording_index, recording_name in enumerate(RECORDINGS):
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
    base_active = valid_rows & nonzero_roi
    primary_mask = base_active & (positive_rates >= PRIMARY_MIN_POSITIVE_BIN_RATE)
    primary_rows = np.flatnonzero(primary_mask)
    strict_within_primary = positive_rates[primary_rows] >= STRICT_MIN_POSITIVE_BIN_RATE

    base_rates = positive_rates[base_active]
    selection_record: dict[str, object] = {
        "recording": recording_name,
        "condition": condition,
        "mouse": mouse_id,
        "recorded_neurons": n_neurons,
        "finite_nonconstant_neurons": int(valid_rows.sum()),
        "dataset_nonzero_roi_neurons": int(nonzero_roi.sum()),
        "primary_state_frames": primary_state_frames,
        "primary_state_duration_seconds": primary_state_frames / fs,
        "base_rate_q25": float(np.quantile(base_rates, 0.25)),
        "base_rate_median": float(np.median(base_rates)),
        "base_rate_q75": float(np.quantile(base_rates, 0.75)),
    }
    for threshold in SELECTION_RATE_GRID:
        safe_threshold = f"{threshold:.3f}".replace(".", "p")
        selection_record[f"selected_at_{safe_threshold}_positive_bins_per_s"] = int(
            np.count_nonzero(base_active & (positive_rates >= threshold))
        )
    selection_records.append(selection_record)
    save_records(
        VALIDATION_DIR / "05_cross_recording_selection.csv",
        selection_records,
    )

    print(
        f"  active population: {primary_rows.size:,}/{n_neurons:,} "
        f"(nonzero_ROI and ≥{PRIMARY_MIN_POSITIVE_BIN_RATE:.3f} positive bins/s)",
        flush=True,
    )
    paper_proxy_count = int(np.count_nonzero(base_active & (positive_rates >= 0.100)))
    print(
        f"  literal 0.100-positive-bin/s proxy would retain " f"{paper_proxy_count:,}",
        flush=True,
    )

    blocks = guarded_blocks(segments, block_frames, guard_frames)
    fold_a_blocks, fold_b_blocks, full_blocks, blocks_per_fold = matched_fold_blocks(
        blocks,
        allowed_codes,
        seed=50_000 + recording_index,
    )
    selected_activity = np.ascontiguousarray(activity[primary_rows], dtype=np.float32)
    del activity
    gc.collect()

    full_matrix, _full_positive_counts, _full_block_slices = matrix_from_blocks(
        selected_activity,
        full_blocks,
        separator_frames=lag_frames,
    )
    full_valid = rmt.valid_activity_rows(full_matrix)
    strict_within_full = strict_within_primary[full_valid]
    full_matrix = np.ascontiguousarray(full_matrix[full_valid], dtype=np.float32)
    full_fit_neurons = full_matrix.shape[0]
    del _full_positive_counts, _full_block_slices

    seed0_model, seed0_summary = fit_matrix(
        full_matrix,
        lag_frames,
        FULL_SEEDS[0],
        keep_normalized=False,
    )
    del seed0_model
    gc.collect()
    seed1_model, seed1_summary = fit_matrix(
        full_matrix,
        lag_frames,
        FULL_SEEDS[1],
        keep_normalized=False,
    )
    del seed1_model
    strict_model, strict_summary = fit_matrix(
        np.ascontiguousarray(full_matrix[strict_within_full], dtype=np.float32),
        lag_frames,
        FULL_SEEDS[0],
        keep_normalized=False,
    )
    del strict_model, full_matrix
    gc.collect()

    seed_spearman = rmt.reversal_invariant_rank_correlation(
        seed0_summary.embedding,
        seed1_summary.embedding,
    )
    seed_local_raw, seed_local_adjusted = adjusted_neighborhood_agreement(
        seed0_summary.embedding,
        seed1_summary.embedding,
        seed=60_000 + recording_index,
    )
    seed_ari = adjusted_rand_score(seed0_summary.clusters, seed1_summary.clusters)
    threshold_spearman = rmt.reversal_invariant_rank_correlation(
        seed0_summary.embedding[strict_within_full],
        strict_summary.embedding,
    )
    threshold_local_raw, threshold_local_adjusted = adjusted_neighborhood_agreement(
        seed0_summary.embedding[strict_within_full],
        strict_summary.embedding,
        seed=65_000 + recording_index,
    )
    occupied_positions, tied_neuron_fraction = tie_fraction(seed0_summary.embedding)

    fold_a_matrix, fold_a_positive_counts, fold_a_block_slices = matrix_from_blocks(
        selected_activity,
        fold_a_blocks,
        separator_frames=lag_frames,
    )
    fold_b_matrix, fold_b_positive_counts, fold_b_block_slices = matrix_from_blocks(
        selected_activity,
        fold_b_blocks,
        separator_frames=lag_frames,
    )
    del selected_activity
    fold_a_valid = rmt.valid_activity_rows(fold_a_matrix)
    fold_b_valid = rmt.valid_activity_rows(fold_b_matrix)
    fold_common = fold_a_valid & fold_b_valid
    fold_a_matrix = np.ascontiguousarray(fold_a_matrix[fold_common], dtype=np.float32)
    fold_b_matrix = np.ascontiguousarray(fold_b_matrix[fold_common], dtype=np.float32)
    fold_a_positive_counts = fold_a_positive_counts[fold_common]
    fold_b_positive_counts = fold_b_positive_counts[fold_common]
    fold_fit_neurons = int(fold_common.sum())
    gc.collect()

    fold_a_model, fold_a_summary = fit_matrix(
        fold_a_matrix,
        lag_frames,
        seed=0,
        keep_normalized=True,
    )
    fold_b_model, fold_b_summary = fit_matrix(
        fold_b_matrix,
        lag_frames,
        seed=0,
        keep_normalized=True,
    )
    verify_normalized_separators(fold_a_model, fold_a_block_slices)
    verify_normalized_separators(fold_b_model, fold_b_block_slices)
    for fitted_model in (fold_a_model, fold_b_model):
        training_replay = transferred_node_similarity(fitted_model, fitted_model.X)
        if not np.allclose(
            training_replay,
            fitted_model.cc,
            atol=2e-5,
            rtol=2e-5,
        ):
            raise RuntimeError("Directed transfer formula did not replay training cc")

    fold_spearman = rmt.reversal_invariant_rank_correlation(
        fold_a_summary.embedding,
        fold_b_summary.embedding,
    )
    fold_local_raw, fold_local_adjusted = adjusted_neighborhood_agreement(
        fold_a_summary.embedding,
        fold_b_summary.embedding,
        seed=70_000 + recording_index,
    )
    fold_ari = adjusted_rand_score(
        fold_a_summary.clusters,
        fold_b_summary.clusters,
    )
    pca_overlap, pca_a_to_b, pca_b_to_a = pca_transfer_efficiencies(
        fold_a_model,
        fold_b_model,
    )
    objective_a_to_b = heldout_objective_record(
        fold_a_model,
        fold_b_model,
        fold_a_positive_counts,
        fold_b_block_slices,
        seed=80_000 + recording_index,
    )
    objective_b_to_a = heldout_objective_record(
        fold_b_model,
        fold_a_model,
        fold_b_positive_counts,
        fold_a_block_slices,
        seed=90_000 + recording_index,
    )

    fit_record: dict[str, object] = {
        "recording": recording_name,
        "condition": condition,
        "mouse": mouse_id,
        "source_size_bytes": source_path.stat().st_size,
        "recorded_neurons": n_neurons,
        "primary_active_neurons": primary_rows.size,
        "full_fit_neurons": full_fit_neurons,
        "strict_fit_neurons": strict_summary.embedding.size,
        "fold_common_fit_neurons": fold_fit_neurons,
        "primary_minimum_positive_bins_per_second": (PRIMARY_MIN_POSITIVE_BIN_RATE),
        "strict_minimum_positive_bins_per_second": STRICT_MIN_POSITIVE_BIN_RATE,
        "lag_frames": lag_frames,
        "lag_seconds": lag_frames / fs,
        "original_lag_pair_fraction_crossing_acquisition_breaks": (
            lag_pair_fraction_crossing_boundaries(
                n_frames,
                boundary_ind,
                lag_frames,
            )
        ),
        "block_seconds": BLOCK_SECONDS,
        "block_edge_guard_seconds": BLOCK_EDGE_GUARD_SECONDS,
        "fold_a_state0_blocks": blocks_per_fold[allowed_codes[0]],
        "fold_a_state1_blocks": blocks_per_fold[allowed_codes[1]],
        "fold_b_state0_blocks": blocks_per_fold[allowed_codes[0]],
        "fold_b_state1_blocks": blocks_per_fold[allowed_codes[1]],
        "state0_label": code_labels[allowed_codes[0]],
        "state1_label": code_labels[allowed_codes[1]],
        "occupied_embedding_positions_seed0": occupied_positions,
        "fraction_neurons_in_tied_positions_seed0": tied_neuron_fraction,
        "seed_abs_spearman": seed_spearman,
        "seed_local_overlap_raw": seed_local_raw,
        "seed_local_overlap_adjusted": seed_local_adjusted,
        "seed_cluster_ari": seed_ari,
        "seed0_runtime_seconds": seed0_summary.runtime_seconds,
        "seed1_runtime_seconds": seed1_summary.runtime_seconds,
        "threshold_primary_vs_strict_abs_spearman": threshold_spearman,
        "threshold_primary_vs_strict_local_overlap_raw": threshold_local_raw,
        "threshold_primary_vs_strict_local_overlap_adjusted": (
            threshold_local_adjusted
        ),
        "strict_runtime_seconds": strict_summary.runtime_seconds,
        "fold_abs_spearman": fold_spearman,
        "fold_local_overlap_raw": fold_local_raw,
        "fold_local_overlap_adjusted": fold_local_adjusted,
        "fold_cluster_ari": fold_ari,
        "fold_a_runtime_seconds": fold_a_summary.runtime_seconds,
        "fold_b_runtime_seconds": fold_b_summary.runtime_seconds,
        "pca_subspace_overlap_adjusted": pca_overlap,
        "pca_transfer_a_to_b_efficiency": pca_a_to_b,
        "pca_transfer_b_to_a_efficiency": pca_b_to_a,
    }
    for prefix, values in (
        ("objective_a_to_b", objective_a_to_b),
        ("objective_b_to_a", objective_b_to_a),
    ):
        for field, value in values.items():
            fit_record[f"{prefix}_{field}"] = value
    for comparator in (
        "reversed",
        "activity",
        "random_mean",
        "blockwise_time_reversal",
        "temporal_order_shuffle_mean",
    ):
        fit_record[f"objective_reciprocal_learned_minus_{comparator}"] = float(
            np.mean(
                [
                    objective_a_to_b["learned"] - objective_a_to_b[comparator],
                    objective_b_to_a["learned"] - objective_b_to_a[comparator],
                ]
            )
        )
    fit_records.append(fit_record)
    save_records(
        VALIDATION_DIR / "05_cross_recording_robustness.csv",
        fit_records,
    )

    print(
        f"  seed |Spearman|={seed_spearman:.3f}, adjusted local="
        f"{seed_local_adjusted:.3f}; matched-fold |Spearman|="
        f"{fold_spearman:.3f}, adjusted local={fold_local_adjusted:.3f}",
        flush=True,
    )
    print(
        f"  threshold 0.020→0.025 |Spearman|={threshold_spearman:.3f}, "
        f"adjusted local={threshold_local_adjusted:.3f}",
        flush=True,
    )
    print(
        f"  held-out cluster objective A→B learned/reversed/random="
        f"{objective_a_to_b['learned']:.4f}/"
        f"{objective_a_to_b['reversed']:.4f}/"
        f"{objective_a_to_b['random_mean']:.4f}; "
        f"order exceedance={objective_a_to_b['permutation_exceedance']:.3f}; "
        f"lag0/delayed={objective_a_to_b['zero_lag']:.4f}/"
        f"{objective_a_to_b['delayed_only_lags_1_to_L']:.4f}; "
        f"time reversal={objective_a_to_b['blockwise_time_reversal']:.4f}; "
        f"time-shuffle mean="
        f"{objective_a_to_b['temporal_order_shuffle_mean']:.4f}, "
        f"exceedance="
        f"{objective_a_to_b['temporal_order_shuffle_exceedance_descriptive']:.3f}",
        flush=True,
    )

    del (
        fold_a_matrix,
        fold_b_matrix,
        fold_a_model,
        fold_b_model,
        fold_a_summary,
        fold_b_summary,
        strict_summary,
        state,
        nonzero_roi,
        positive_counts,
        positive_rates,
    )
    gc.collect()

print("saved ->", VALIDATION_DIR / "05_cross_recording_selection.csv")
print("saved ->", VALIDATION_DIR / "05_cross_recording_robustness.csv")


# %% [markdown]
# ## Step 5 — selection audit across all recordings
#
# The left and middle panels show why activity selection cannot be described as
# a harmless preprocessing detail. The right panel shows the positive-bin-rate
# distribution only among finite `nonzero_ROI` rows. The red 0.10 line is a
# literal proxy, not the paper's calibrated firing-rate threshold.

# %%
selection_x = np.arange(len(selection_records))
selection_labels = [
    record["recording"].replace("mouse", "m") for record in selection_records
]
selection_colors = [
    "tab:blue" if record["condition"] == "sleep" else "tab:orange"
    for record in selection_records
]

selection_figure, selection_axes = plt.subplots(
    1,
    3,
    figsize=(18, 5.5),
    constrained_layout=True,
)
for threshold, marker in ((0.020, "o"), (0.025, "s"), (0.100, "^")):
    field = f"selected_at_{threshold:.3f}".replace(".", "p") + "_positive_bins_per_s"
    fractions = np.array(
        [record[field] / record["recorded_neurons"] for record in selection_records]
    )
    selection_axes[0].plot(
        selection_x,
        fractions,
        marker=marker,
        label=f"≥{threshold:.3f} positive bins/s",
    )
dataset_fractions = np.array(
    [
        record["dataset_nonzero_roi_neurons"] / record["recorded_neurons"]
        for record in selection_records
    ]
)
selection_axes[0].plot(
    selection_x,
    dataset_fractions,
    "d--",
    color="0.35",
    label="dataset nonzero_ROI",
)
selection_axes[0].set_ylim(0, 1.03)
selection_axes[0].set_ylabel("fraction of recorded neurons retained")
selection_axes[0].set_title("Selection is strongly condition dependent")
selection_axes[0].legend(frameon=False, fontsize=8)

width = 0.25
for offset, threshold, label in (
    (-width, 0.020, "working ≥0.020"),
    (0.0, 0.025, "sensitivity ≥0.025"),
    (width, 0.100, "literal proxy ≥0.100"),
):
    field = f"selected_at_{threshold:.3f}".replace(".", "p") + "_positive_bins_per_s"
    selection_axes[1].bar(
        selection_x + offset,
        [record[field] for record in selection_records],
        width,
        label=label,
    )
selection_axes[1].set_yscale("log")
selection_axes[1].set_ylabel("selected neurons (log scale)")
selection_axes[1].set_title("0.100 proxy can leave too few anesthesia neurons")
selection_axes[1].legend(frameon=False, fontsize=8)

for index, record in enumerate(selection_records):
    selection_axes[2].vlines(
        index,
        record["base_rate_q25"],
        record["base_rate_q75"],
        color=selection_colors[index],
        lw=4,
        alpha=0.65,
    )
    selection_axes[2].scatter(
        index,
        record["base_rate_median"],
        color=selection_colors[index],
        s=24,
        zorder=2,
    )
selection_axes[2].axhline(0.020, color="black", ls="--", lw=1, label="working floor")
selection_axes[2].axhline(
    0.100, color="tab:red", ls=":", lw=1.2, label="literal 0.10 proxy"
)
selection_axes[2].set_yscale("log")
selection_axes[2].set_ylabel("positive OASIS bins/s")
selection_axes[2].set_title("Median and interquartile activity support")
selection_axes[2].legend(frameon=False, fontsize=8)

for axis in selection_axes:
    axis.set_xticks(selection_x, selection_labels, rotation=45, ha="right")
    axis.grid(axis="y", color="0.88", lw=0.6)

selection_figure.suptitle(
    "Active-neuron definition audit · all eligible finite nonzero_ROI rows "
    "above each threshold retained"
)
selection_path = FIG_DIR / "05_rastermap_01_cross_recording_selection.png"
selection_figure.savefig(selection_path, dpi=150, bbox_inches="tight")
print("saved ->", selection_path)
if SHOW_FIGURES:
    plt.show()


# %% [markdown]
# ## Step 6 — cross-recording robustness summary
#
# Every point is a within-recording comparison. High fine-order agreement would
# require both global rank and adjusted local-neighborhood agreement. For the
# directional objective, positive learned-minus-comparator values indicate that
# the coarse learned cluster order transfers better than the stated baseline.
# The held-out temporal-order shuffle retains instantaneous population vectors
# but destroys positive-lag order within each original block. Both exceedances
# are descriptive because sessions—not cluster pairs or shuffled frames—are the
# descriptive units of recurrence.

# %%
fit_x = np.arange(len(fit_records))
fit_labels = [record["recording"].replace("mouse", "m") for record in fit_records]

replication_figure, replication_axes = plt.subplots(
    2,
    3,
    figsize=(18, 10),
    constrained_layout=True,
)

replication_axes[0, 0].plot(
    fit_x,
    [record["seed_abs_spearman"] for record in fit_records],
    "o-",
    label="|Spearman ρ|",
)
replication_axes[0, 0].plot(
    fit_x,
    [record["seed_local_overlap_adjusted"] for record in fit_records],
    "s-",
    label="adjusted local overlap",
)
replication_axes[0, 0].plot(
    fit_x,
    [record["threshold_primary_vs_strict_abs_spearman"] for record in fit_records],
    "o--",
    label="0.020 vs 0.025 |ρ|",
)
replication_axes[0, 0].plot(
    fit_x,
    [
        record["threshold_primary_vs_strict_local_overlap_adjusted"]
        for record in fit_records
    ],
    "s--",
    label="0.020 vs 0.025 local",
)
replication_axes[0, 0].set_title("All matched usable-block seed stability")
replication_axes[0, 0].set_ylabel("fine-order agreement")
replication_axes[0, 0].legend(frameon=False)

replication_axes[0, 1].plot(
    fit_x,
    [record["fold_abs_spearman"] for record in fit_records],
    "o-",
    label="|Spearman ρ|",
)
replication_axes[0, 1].plot(
    fit_x,
    [record["fold_local_overlap_adjusted"] for record in fit_records],
    "s-",
    label="adjusted local overlap",
)
replication_axes[0, 1].set_title("Disjoint matched-fold stability")
replication_axes[0, 1].set_ylabel("fine-order agreement")
replication_axes[0, 1].legend(frameon=False)

replication_axes[0, 2].plot(
    fit_x,
    [record["pca_subspace_overlap_adjusted"] for record in fit_records],
    "o-",
    label="adjusted subspace overlap",
)
replication_axes[0, 2].plot(
    fit_x,
    [record["pca_transfer_a_to_b_efficiency"] for record in fit_records],
    "s-",
    label="A→B efficiency",
)
replication_axes[0, 2].plot(
    fit_x,
    [record["pca_transfer_b_to_a_efficiency"] for record in fit_records],
    "^-",
    label="B→A efficiency",
)
replication_axes[0, 2].set_title("Held-out low-dimensional structure")
replication_axes[0, 2].set_ylabel("adjusted overlap / efficiency")
replication_axes[0, 2].legend(frameon=False, fontsize=8)

learned_minus_reversed = np.array(
    [
        np.mean(
            [
                record["objective_a_to_b_learned"]
                - record["objective_a_to_b_reversed"],
                record["objective_b_to_a_learned"]
                - record["objective_b_to_a_reversed"],
            ]
        )
        for record in fit_records
    ]
)
learned_minus_activity = np.array(
    [
        np.mean(
            [
                record["objective_a_to_b_learned"]
                - record["objective_a_to_b_activity"],
                record["objective_b_to_a_learned"]
                - record["objective_b_to_a_activity"],
            ]
        )
        for record in fit_records
    ]
)
learned_minus_random = np.array(
    [
        np.mean(
            [
                record["objective_a_to_b_learned"]
                - record["objective_a_to_b_random_mean"],
                record["objective_b_to_a_learned"]
                - record["objective_b_to_a_random_mean"],
            ]
        )
        for record in fit_records
    ]
)
learned_minus_temporal_shuffle = np.array(
    [
        np.mean(
            [
                record["objective_a_to_b_learned"]
                - record["objective_a_to_b_temporal_order_shuffle_mean"],
                record["objective_b_to_a_learned"]
                - record["objective_b_to_a_temporal_order_shuffle_mean"],
            ]
        )
        for record in fit_records
    ]
)
learned_minus_time_reversal = np.array(
    [
        np.mean(
            [
                record["objective_a_to_b_learned"]
                - record["objective_a_to_b_blockwise_time_reversal"],
                record["objective_b_to_a_learned"]
                - record["objective_b_to_a_blockwise_time_reversal"],
            ]
        )
        for record in fit_records
    ]
)
replication_axes[1, 0].plot(
    fit_x,
    learned_minus_reversed,
    "o-",
    label="learned − reversed",
)
replication_axes[1, 0].plot(
    fit_x,
    learned_minus_activity,
    "s-",
    label="learned − activity order",
)
replication_axes[1, 0].plot(
    fit_x,
    learned_minus_random,
    "^-",
    label="learned − permutation mean",
)
replication_axes[1, 0].plot(
    fit_x,
    learned_minus_temporal_shuffle,
    "d-",
    label="learned − time-shuffle mean",
)
replication_axes[1, 0].plot(
    fit_x,
    learned_minus_time_reversal,
    "x-",
    label="learned − blockwise time reversal",
)
replication_axes[1, 0].axhline(0, color="black", lw=0.8)
replication_axes[1, 0].set_title("Reciprocal held-out directional objective")
replication_axes[1, 0].set_ylabel("objective difference")
replication_axes[1, 0].legend(frameon=False, fontsize=8)

replication_axes[1, 1].plot(
    fit_x,
    [
        record["objective_a_to_b_temporal_order_shuffle_exceedance_descriptive"]
        for record in fit_records
    ],
    "o-",
    label="A→B time shuffle",
)
replication_axes[1, 1].plot(
    fit_x,
    [
        record["objective_b_to_a_temporal_order_shuffle_exceedance_descriptive"]
        for record in fit_records
    ],
    "s-",
    label="B→A time shuffle",
)
replication_axes[1, 1].plot(
    fit_x,
    [record["objective_a_to_b_permutation_exceedance"] for record in fit_records],
    ":",
    color="0.35",
    label="A→B cluster-order permutations",
)
replication_axes[1, 1].axhline(0.05, color="tab:red", ls=":", lw=1)
replication_axes[1, 1].set_yscale("log")
replication_axes[1, 1].set_title("Cluster-order and held-out time-order nulls")
replication_axes[1, 1].set_ylabel("descriptive exceedance")
replication_axes[1, 1].legend(frameon=False, fontsize=8)

replication_axes[1, 2].plot(
    fit_x,
    [record["fraction_neurons_in_tied_positions_seed0"] for record in fit_records],
    "o-",
    label="neurons in tied positions",
)
replication_axes[1, 2].plot(
    fit_x,
    [
        record["occupied_embedding_positions_seed0"] / record["full_fit_neurons"]
        for record in fit_records
    ],
    "s-",
    label="occupied positions / neurons",
)
replication_axes[1, 2].set_ylim(0, 1.03)
replication_axes[1, 2].set_title("Fine-order resolution")
replication_axes[1, 2].set_ylabel("fraction")
replication_axes[1, 2].legend(frameon=False, fontsize=8)

for axis in replication_axes.ravel():
    axis.set_xticks(fit_x, fit_labels, rotation=45, ha="right")
    axis.grid(axis="y", color="0.88", lw=0.6)

replication_figure.suptitle(
    "Active-neuron Rastermap verification screen · session-level, descriptive"
)
replication_path = FIG_DIR / "05_rastermap_02_cross_recording_robustness.png"
replication_figure.savefig(replication_path, dpi=150, bbox_inches="tight")
print("saved ->", replication_path)
if SHOW_FIGURES:
    plt.show()


# %% [markdown]
# ## Step 7 — is the transferred structure specifically directional?
#
# The lag-inclusive score is an elementwise maximum over lags 0…L. A strong
# value can therefore arise from synchronous covariance alone. The first panel
# separates exact lag-0 and delayed-only (lags 1…L) components. The second uses
# two controls that retain more temporal structure than an arbitrary shuffle:
# reversing time separately inside each block, and reversing the learned cluster
# direction. Small excesses here mean that coarse organization transfers more
# strongly than its specific temporal direction.


# %%
def reciprocal_objective_values(field):
    """Return the A→B/B→A mean of one objective field per recording."""
    return np.array(
        [
            np.mean(
                [
                    record[f"objective_a_to_b_{field}"],
                    record[f"objective_b_to_a_{field}"],
                ]
            )
            for record in fit_records
        ]
    )


inclusive_objective = reciprocal_objective_values("learned")
zero_lag_objective = reciprocal_objective_values("zero_lag")
delayed_objective = reciprocal_objective_values("delayed_only_lags_1_to_L")
time_reversed_objective = reciprocal_objective_values("blockwise_time_reversal")
time_shuffled_objective = reciprocal_objective_values("temporal_order_shuffle_mean")
lag_values = sorted({int(record["lag_frames"]) for record in fit_records})
lag_text = str(lag_values[0]) if len(lag_values) == 1 else "L"

direction_figure, direction_axes = plt.subplots(
    1,
    2,
    figsize=(15, 5.5),
    constrained_layout=True,
)
direction_axes[0].plot(
    fit_x,
    zero_lag_objective / inclusive_objective,
    "o-",
    label="lag 0 / lag-inclusive",
)
direction_axes[0].plot(
    fit_x,
    delayed_objective / inclusive_objective,
    "s-",
    label=f"lags 1…{lag_text} / lag-inclusive",
)
direction_axes[0].plot(
    fit_x,
    time_reversed_objective / inclusive_objective,
    "^-",
    label="blockwise time reversal / original",
)
direction_axes[0].plot(
    fit_x,
    time_shuffled_objective / inclusive_objective,
    "d-",
    label="time-shuffle mean / original",
)
direction_axes[0].axhline(1, color="black", lw=0.8)
direction_axes[0].set_ylabel("objective ratio")
direction_axes[0].set_title("Most coarse objective magnitude is not direction-specific")
direction_axes[0].legend(frameon=False, fontsize=8)

direction_axes[1].plot(
    fit_x,
    learned_minus_reversed,
    "o-",
    label="original − reversed cluster direction",
)
direction_axes[1].plot(
    fit_x,
    learned_minus_time_reversal,
    "s-",
    label="original − blockwise reversed time",
)
direction_axes[1].plot(
    fit_x,
    learned_minus_temporal_shuffle,
    "^-",
    label="original − time-shuffle mean",
)
direction_axes[1].axhline(0, color="black", lw=0.8)
direction_axes[1].set_ylabel("held-out objective difference")
direction_axes[1].set_title("Directional/temporal excess is modest and heterogeneous")
direction_axes[1].legend(frameon=False, fontsize=8)

for axis in direction_axes:
    axis.set_xticks(fit_x, fit_labels, rotation=45, ha="right")
    axis.grid(axis="y", color="0.88", lw=0.6)

direction_figure.suptitle(
    "Lag decomposition and direction controls · reciprocal matched-fold transfer"
)
direction_path = FIG_DIR / "05_rastermap_03_direction_controls.png"
direction_figure.savefig(direction_path, dpi=150, bbox_inches="tight")
print("saved ->", direction_path)
if SHOW_FIGURES:
    plt.show()


# %% [markdown]
# ## Step 8 — mouse-level descriptive aggregation
#
# The two mouse04 sleep days are averaged before the sleep cohort summary. This
# prevents that mouse from silently receiving twice the weight. With five sleep
# and four anesthesia mice, ranges and medians are more honest than asymptotic
# significance tests.

# %%
MOUSE_SUMMARY_FIELDS = (
    "seed_abs_spearman",
    "seed_local_overlap_adjusted",
    "threshold_primary_vs_strict_abs_spearman",
    "threshold_primary_vs_strict_local_overlap_adjusted",
    "fold_abs_spearman",
    "fold_local_overlap_adjusted",
    "pca_subspace_overlap_adjusted",
    "pca_transfer_a_to_b_efficiency",
    "pca_transfer_b_to_a_efficiency",
    "objective_reciprocal_learned_minus_reversed",
    "objective_reciprocal_learned_minus_activity",
    "objective_reciprocal_learned_minus_random_mean",
    "objective_reciprocal_learned_minus_blockwise_time_reversal",
    "objective_reciprocal_learned_minus_temporal_order_shuffle_mean",
)

mouse_records: list[dict[str, object]] = []
for condition in ("sleep", "anesthesia"):
    condition_records = [
        record for record in fit_records if record["condition"] == condition
    ]
    for mouse_id in sorted({record["mouse"] for record in condition_records}):
        sessions = [
            record for record in condition_records if record["mouse"] == mouse_id
        ]
        mouse_record: dict[str, object] = {
            "condition": condition,
            "mouse": mouse_id,
            "n_sessions": len(sessions),
        }
        for field in MOUSE_SUMMARY_FIELDS:
            mouse_record[field] = float(
                np.mean([float(record[field]) for record in sessions])
            )
        mouse_records.append(mouse_record)

save_records(
    VALIDATION_DIR / "05_cross_recording_mouse_summary.csv",
    mouse_records,
)
print("saved ->", VALIDATION_DIR / "05_cross_recording_mouse_summary.csv")

for condition in ("sleep", "anesthesia"):
    cohort = [record for record in mouse_records if record["condition"] == condition]
    print(f"\n{condition.title()} mouse-level descriptive summary (n={len(cohort)}):")
    for field in (
        "seed_abs_spearman",
        "seed_local_overlap_adjusted",
        "threshold_primary_vs_strict_abs_spearman",
        "threshold_primary_vs_strict_local_overlap_adjusted",
        "fold_abs_spearman",
        "fold_local_overlap_adjusted",
        "pca_subspace_overlap_adjusted",
        "objective_reciprocal_learned_minus_random_mean",
        "objective_reciprocal_learned_minus_reversed",
        "objective_reciprocal_learned_minus_blockwise_time_reversal",
    ):
        values = np.array([record[field] for record in cohort], dtype=float)
        print(
            f"  {field}: median={np.median(values):.3f}, "
            f"range={values.min():.3f}…{values.max():.3f}"
        )


# %% [markdown]
# ## Interpretation guardrails
#
# - Repeated high **cluster-objective transfer** supports recurring coarse,
#   lag-inclusive organization. The lag decomposition and reversal controls show
#   that the specifically directional excess is much smaller and heterogeneous.
#   Neither result implies a unique neuron-by-neuron order.
# - Low fine-order rank/local agreement, many tied positions, or seed dependence
#   argue against treating Rastermap positions as discrete biological modules.
# - `nonzero_ROI` is specific to this dataset's network windows, and 0.020
#   positive bins/s is uncalibrated. The selection figure must accompany any
#   result from the working population.
# - Selection and fold-common validity use both sides of the split, and each
#   target fold is normalized using its own values. Treat transfer as conditional
#   validation, not as a leakage-free prospective prediction.
# - Matched folds control state composition and block seams, but not every slow
#   drift, transition, motion, or global-population confound. `mean_time=True`
#   removes the instantaneous population mean, not all shared signals.
# - The arbitrary time shuffle also removes autocorrelation, so it is a
#   destructive diagnostic rather than an exchangeable-frame significance test;
#   the blockwise reversal is the more conservative direction control.
# - This screen uses one matched-block allocation and seed 0 for the fold fits.
#   Repeating both is the strongest remaining uncertainty analysis.
# - Only awake + NREM or awake + anesthesia enters this screen. Short bouts,
#   incomplete 30-s tails, and six seconds of block-edge guards are omitted, so
#   "all matched usable blocks" is not the complete recording.
# - Recurrence here means that a within-recording diagnostic appears across mice;
#   it never means that embeddings from different animals share neuron identity
#   or that this dataset reproduces the paper's unpublished Figure 3 cutoff.

print("\nCross-recording Rastermap verification complete.")
