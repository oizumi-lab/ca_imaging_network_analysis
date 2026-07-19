# %% [markdown]
# # 06 · How much Rastermap uncertainty comes from time blocks versus fit seeds?
#
# Tutorial 05 showed one state-matched time allocation per recording.  This
# tutorial repeats that screen in a fully crossed design: three independently
# matched block allocations × three official Rastermap seeds, with both folds
# fitted in every cell.  It therefore separates variability caused by which
# time blocks were selected from variability caused by scaled-k-means
# initialization.  No neuron is randomly sampled.
#
# The primary population is the dataset's documented active set:
# finite/nonconstant rows intersected with `nonzero_ROI`.  The latter was made
# from the publication's complete analysis windows and is more defensible here
# than converting the Rastermap paper's physiological 0.1--0.25-Hz statement
# into an arbitrary threshold on uncalibrated OASIS samples. Tutorial 05's
# 0.020-positive-bin/s population remains a deliberately stricter support-floor
# sensitivity arm, so differences between tutorials 05 and 06 must not be
# attributed to resampling alone.
#
# Results remain descriptive. Repeated splits and seeds are perturbations of
# one recording, not additional animals or independent biological replicates.

# %% Step 0 — imports
import csv
import gc
import os
import sys
from dataclasses import dataclass
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import h5py
import matplotlib.pyplot as plt
import numpy as np
from rastermap import Rastermap
from rastermap.cluster import compute_cc_tdelay
from scipy.stats import zscore
from sklearn.metrics import adjusted_rand_score

from src.funcnet import dataio
from src.funcnet import rastermap_tools as rmt
from src.funcnet import timeseries as ts
from src.funcnet.paths import RESULTS_DIR


# %% [markdown]
# ## Step 1 — settings
#
# - `N_BLOCK_ALLOCATIONS=3` and `FIT_SEEDS=(0,1,2)` form a balanced 3×3
#   factorial grid. Each grid cell fits fold A and fold B, for 180 official fits
#   over ten recordings.
# - Each allocation contains the same number of 30-s blocks from both states in
#   A and B. Three seconds are removed from both block edges.
# - A single neuron mask is fixed across every allocation and fold of a
#   recording. This prevents changing row eligibility from masquerading as
#   split variability.
# - `REUSE_COMPLETE_RESULTS=True` skips a recording only when both checkpoint
#   tables contain every expected row with the current configuration.
# - The random-order comparator is its exact analytic expectation, not a Monte
#   Carlo estimate. Tutorial 05 retains the expensive permutation and temporal
#   nulls.

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
N_BLOCK_ALLOCATIONS = 3
FIT_SEEDS = (0, 1, 2)
SPLIT_SEED_BASE = 50_000
SPLIT_SEED_STRIDE = 10_000
BLOCK_SECONDS = 30.0
BLOCK_EDGE_GUARD_SECONDS = 3.0

N_CLUSTERS = 100
N_PCS = 128
LOCALITY = 0.0
MEAN_TIME = True
PAPER_LAG_SECONDS = 5 / 3.2
NEIGHBORHOOD_SIZE = 50
TIE_PERMUTATIONS = 8
MIN_NEURONS_FOR_FIT = 256

REUSE_COMPLETE_RESULTS = True
SHOW_FIGURES = True

VALIDATION_DIR = RESULTS_DIR / "rastermap_validation"
FIG_DIR = RESULTS_DIR / "figures"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CELL_PATH = VALIDATION_DIR / "06_resampling_cells.csv"
SEED_PAIR_PATH = VALIDATION_DIR / "06_resampling_same_input_seed_pairs.csv"
SUMMARY_PATH = VALIDATION_DIR / "06_resampling_summary.csv"
RASTERMAP_VERSION = version("rastermap")


# %% [markdown]
# ## Step 2 — lightweight loading and state-matched blocks
#
# Only the deconvolved signal, state vector, acquisition boundaries, and
# `nonzero_ROI` are loaded. Blocks never cross a state transition or microscope
# break. `matrix_from_blocks` inserts a lag-wide row-mean separator; after
# Rastermap's row centering these columns must be exactly zero.


# %%
def load_deconvolution_only(recording_name: str):
    """Load only arrays required by this resampling screen."""
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
    """Return condition, mouse, primary state codes, and readable labels."""
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


def guarded_blocks(
    segments: list[tuple[int, int, float]],
    block_frames: int,
    guard_frames: int,
) -> list[tuple[int, int, float]]:
    """Tile state-pure segments and retain guarded block interiors."""
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
    blocks: list[tuple[int, int, float]],
    allowed_codes: tuple[float, ...],
    seed: int,
):
    """Allocate equal A/B block counts independently within each state."""
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


def block_fingerprint(blocks: list[tuple[int, int, float]]) -> tuple[int, ...]:
    """Return a deterministic fingerprint for allocation uniqueness checks."""
    return tuple(value for block in sorted(blocks) for value in (block[0], block[1]))


def valid_rows_in_blocks(
    selected_activity: np.ndarray,
    blocks: list[tuple[int, int, float]],
) -> np.ndarray:
    """Find finite, nonconstant rows without constructing a joined matrix."""
    finite = np.ones(selected_activity.shape[0], dtype=bool)
    minimum = np.full(selected_activity.shape[0], np.inf, dtype=np.float32)
    maximum = np.full(selected_activity.shape[0], -np.inf, dtype=np.float32)
    for start, stop, _code in blocks:
        values = selected_activity[:, start:stop]
        finite &= np.all(np.isfinite(values), axis=1)
        minimum = np.minimum(minimum, np.nanmin(values, axis=1))
        maximum = np.maximum(maximum, np.nanmax(values, axis=1))
    return finite & (maximum > minimum)


def matrix_from_blocks(
    selected_activity: np.ndarray,
    blocks: list[tuple[int, int, float]],
    separator_frames: int,
) -> tuple[np.ndarray, np.ndarray, list[slice]]:
    """Join blocks with row-mean seams and return activity counts and slices."""
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
        data_slice = slice(position, position + width)
        data_slices.append(data_slice)
        matrix[:, data_slice] = values
        positive_counts += np.count_nonzero(values > 0, axis=1)
        row_sum += np.sum(values, axis=1, dtype=np.float64)
        position += width
        if block_index < len(blocks) - 1 and separator_frames:
            seam = slice(position, position + separator_frames)
            separator_slices.append(seam)
            position += separator_frames
    row_mean = (row_sum / data_frames).astype(np.float32)
    for seam in separator_slices:
        matrix[:, seam] = row_mean[:, np.newaxis]
    if position != total_frames:
        raise RuntimeError("Block concatenation produced an unexpected width")
    return matrix, positive_counts, data_slices


# %% [markdown]
# ## Step 3 — fit and comparison helpers
#
# Fine-order metrics are reversal-invariant and tie-aware. Coarse transfer uses
# the exact Rastermap projection and lagged similarity equations already checked
# in tutorial 05. The expected objective of a uniformly random complete cluster
# order equals the mean off-diagonal transferred similarity, so it is computed
# analytically and introduces no Monte Carlo noise into the split×seed grid.


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
    """Fit official Rastermap to a prescreened active population."""
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
        raise RuntimeError("A prescreened fit unexpectedly removed neuron rows")
    return model, FitSummary(
        embedding=np.asarray(model.embedding, dtype=np.float32).ravel().copy(),
        clusters=np.asarray(model.embedding_clust, dtype=np.int32).ravel().copy(),
        runtime_seconds=float(model.runtime),
    )


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
    return raw, (raw - chance) / (1 - chance)


def fine_order_record(
    first: FitSummary,
    second: FitSummary,
    metric_seed: int,
) -> dict[str, float]:
    """Compare two aligned fits using global, local, and cluster metrics."""
    local_raw, local_adjusted = adjusted_neighborhood_agreement(
        first.embedding,
        second.embedding,
        seed=metric_seed,
    )
    return {
        "abs_spearman": rmt.reversal_invariant_rank_correlation(
            first.embedding,
            second.embedding,
        ),
        "local_overlap_raw": local_raw,
        "local_overlap_adjusted": local_adjusted,
        "cluster_ari": adjusted_rand_score(first.clusters, second.clusters),
    }


def verify_normalized_separators(
    model: Rastermap,
    block_slices: list[slice],
) -> None:
    """Assert that seams are lag-wide and zero after normalization."""
    lag_frames = int(model.time_lag_window)
    for first, second in zip(block_slices[:-1], block_slices[1:], strict=True):
        if second.start - first.stop < lag_frames:
            raise RuntimeError("A seam is shorter than the fitted lag")
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
    """Project held-out normalized activity through source PCA axes."""
    singular_values = np.asarray(source_model.sv, dtype=np.float32)
    source_left = np.asarray(source_model.Usv, dtype=np.float32) / singular_values
    return (target_normalized_activity.T @ source_left) / singular_values


def node_similarity_from_temporal_scores(
    source_model: Rastermap,
    temporal_scores: np.ndarray,
) -> np.ndarray:
    """Apply the installed directed lag-similarity calculation."""
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
    """Return exact lag-0 and delayed-only node similarities."""
    node_activity = zscore(
        np.asarray(source_model.U_nodes, dtype=np.float32) @ temporal_scores.T,
        axis=1,
    )
    n_timepoints = node_activity.shape[1]
    zero_lag = node_activity @ node_activity.T / n_timepoints
    maximum_lag = int(source_model.time_lag_window)
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
    """Score one complete directed cluster order with Rastermap's kernel."""
    similarity = np.asarray(node_similarity, dtype=np.float64)
    target = np.asarray(matching_target, dtype=np.float64)
    order = np.asarray(cluster_order, dtype=np.int64)
    if similarity.shape != target.shape or similarity.ndim != 2:
        raise ValueError("Similarity and target must be aligned square matrices")
    if order.size != similarity.shape[0] or np.unique(order).size != order.size:
        raise ValueError("cluster_order must be a complete permutation")
    weights = np.triu(target, k=1)
    ordered = similarity[np.ix_(order, order)]
    return float(np.sum(weights * ordered) / weights.sum())


def activity_cluster_order(
    model: Rastermap,
    neuron_positive_counts: np.ndarray,
) -> np.ndarray:
    """Orient an activity-ranked cluster order using source data only."""
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
    return (
        ascending
        if directional_objective(model.cc, model.BBt, ascending)
        >= directional_objective(model.cc, model.BBt, descending)
        else descending
    )


def exact_random_order_expectation(node_similarity: np.ndarray) -> float:
    """Return the exact expected objective under a uniform cluster order."""
    similarity = np.asarray(node_similarity, dtype=np.float64)
    n_clusters = similarity.shape[0]
    return float(
        (similarity.sum() - np.trace(similarity)) / (n_clusters * (n_clusters - 1))
    )


def coarse_transfer_record(
    source_model: Rastermap,
    target_model: Rastermap,
    source_positive_counts: np.ndarray,
    target_block_slices: list[slice],
) -> dict[str, float]:
    """Compute deterministic held-out coarse-order diagnostics."""
    temporal_scores = transferred_temporal_scores(source_model, target_model.X)
    transferred = node_similarity_from_temporal_scores(source_model, temporal_scores)
    zero_lag, delayed = lag_component_similarities(source_model, temporal_scores)
    if not np.allclose(
        transferred,
        np.maximum(zero_lag, delayed),
        atol=2e-5,
        rtol=2e-5,
    ):
        raise RuntimeError("Lag decomposition did not replay transferred similarity")
    identity = np.arange(source_model.U_nodes.shape[0], dtype=np.int64)
    reversed_scores = temporal_scores.copy()
    for block_slice in target_block_slices:
        reversed_scores[block_slice] = temporal_scores[block_slice][::-1]
    reversed_time = node_similarity_from_temporal_scores(
        source_model,
        reversed_scores,
    )
    activity_order = activity_cluster_order(source_model, source_positive_counts)
    return {
        "learned": directional_objective(transferred, source_model.BBt, identity),
        "reversed": directional_objective(
            transferred,
            source_model.BBt,
            identity[::-1],
        ),
        "activity": directional_objective(
            transferred,
            source_model.BBt,
            activity_order,
        ),
        "random_expectation": exact_random_order_expectation(transferred),
        "zero_lag": directional_objective(zero_lag, source_model.BBt, identity),
        "delayed_only_lags_1_to_L": directional_objective(
            delayed,
            source_model.BBt,
            identity,
        ),
        "blockwise_time_reversal": directional_objective(
            reversed_time,
            source_model.BBt,
            identity,
        ),
    }


def matrix_energy(activity: np.ndarray, chunk_frames: int = 512) -> float:
    """Accumulate squared matrix energy without a float64 full copy."""
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
    """Return adjusted subspace overlap and reciprocal PCA transfer efficiency."""
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


def save_records(path: Path, records: list[dict[str, object]]) -> None:
    """Checkpoint same-schema records."""
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def load_records(path: Path) -> list[dict[str, str]]:
    """Load a checkpoint table if it exists."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def numeric(record: dict[str, object], field: str) -> float:
    """Read a numeric field consistently from fresh or CSV-loaded records."""
    return float(record[field])


def configuration_record(source_path: Path) -> dict[str, object]:
    """Return exact analysis and source provenance for checkpoint validation."""
    source_stat = source_path.stat()
    return {
        "analysis_schema": ANALYSIS_SCHEMA,
        "source_size_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "rastermap_version": RASTERMAP_VERSION,
        "selection_definition": "finite_nonconstant_and_nonzero_ROI",
        "n_block_allocations": N_BLOCK_ALLOCATIONS,
        "fit_seeds": "|".join(str(seed) for seed in FIT_SEEDS),
        "split_seed_base": SPLIT_SEED_BASE,
        "split_seed_stride": SPLIT_SEED_STRIDE,
        "block_seconds": BLOCK_SECONDS,
        "block_edge_guard_seconds": BLOCK_EDGE_GUARD_SECONDS,
        "paper_lag_seconds": PAPER_LAG_SECONDS,
        "n_clusters": N_CLUSTERS,
        "n_pcs": N_PCS,
        "locality": LOCALITY,
        "mean_time": int(MEAN_TIME),
        "neighborhood_size": NEIGHBORHOOD_SIZE,
        "tie_permutations": TIE_PERMUTATIONS,
    }


def balanced_factor_shares(values: np.ndarray) -> tuple[float, float, float]:
    """Return descriptive split, seed, and interaction shares in a balanced grid."""
    values = np.asarray(values, dtype=float)
    if values.shape != (N_BLOCK_ALLOCATIONS, len(FIT_SEEDS)):
        raise ValueError("values must match the configured split×seed grid")
    grand = values.mean()
    split_means = values.mean(axis=1)
    seed_means = values.mean(axis=0)
    total = float(np.sum((values - grand) ** 2))
    if total <= 0:
        return 0.0, 0.0, 0.0
    split_ss = len(FIT_SEEDS) * float(np.sum((split_means - grand) ** 2))
    seed_ss = N_BLOCK_ALLOCATIONS * float(np.sum((seed_means - grand) ** 2))
    interaction_ss = max(0.0, total - split_ss - seed_ss)
    return split_ss / total, seed_ss / total, interaction_ss / total


# %% [markdown]
# ## Step 4 — stream all recordings through the 3×3 grid
#
# Before fitting, validity is intersected over A and B in every allocation. This
# makes the neuron population identical throughout the grid and isolates the
# two perturbations of interest. It is still conditional/transductive QC because
# every fold participates in the validity check; it is not a prospective test.
#
# A completed recording is reusable from the CSV checkpoints. A partial or
# configuration-mismatched recording is recomputed from the beginning so that
# factorial cells are never silently mixed across settings.

# %%
cell_records: list[dict[str, object]] = list(load_records(CELL_PATH))
seed_pair_records: list[dict[str, object]] = list(load_records(SEED_PAIR_PATH))


def checkpoint_configuration_matches(
    record: dict[str, object],
    recording_name: str,
) -> bool:
    """Return whether a checkpoint row exactly matches data and settings."""
    expected = configuration_record(dataio.RAW_DIR / f"{recording_name}.mat")
    try:
        exact_fields = (
            "analysis_schema",
            "source_size_bytes",
            "source_mtime_ns",
            "rastermap_version",
            "selection_definition",
            "n_block_allocations",
            "fit_seeds",
            "split_seed_base",
            "split_seed_stride",
            "n_clusters",
            "n_pcs",
            "mean_time",
            "neighborhood_size",
            "tie_permutations",
        )
        for field in exact_fields:
            if str(record[field]) != str(expected[field]):
                return False
        float_fields = (
            "block_seconds",
            "block_edge_guard_seconds",
            "paper_lag_seconds",
            "locality",
        )
        return all(
            np.isclose(float(record[field]), float(expected[field]))
            for field in float_fields
        )
    except (KeyError, TypeError, ValueError):
        return False


cell_records = [
    record
    for record in cell_records
    if record.get("recording") in RECORDINGS
    and checkpoint_configuration_matches(record, str(record["recording"]))
]
seed_pair_records = [
    record
    for record in seed_pair_records
    if record.get("recording") in RECORDINGS
    and checkpoint_configuration_matches(record, str(record["recording"]))
]


expected_cells_per_recording = N_BLOCK_ALLOCATIONS * len(FIT_SEEDS)
expected_pairs_per_recording = (
    N_BLOCK_ALLOCATIONS * 2 * len(tuple(combinations(FIT_SEEDS, 2)))
)

for recording_index, recording_name in enumerate(RECORDINGS):
    existing_cells = [
        record for record in cell_records if record["recording"] == recording_name
    ]
    existing_pairs = [
        record for record in seed_pair_records if record["recording"] == recording_name
    ]
    complete_keys = {
        (int(record["split_index"]), int(record["fit_seed"]))
        for record in existing_cells
        if checkpoint_configuration_matches(record, recording_name)
    }
    expected_keys = {
        (split_index, fit_seed)
        for split_index in range(N_BLOCK_ALLOCATIONS)
        for fit_seed in FIT_SEEDS
    }
    complete_pair_keys = {
        (
            int(record["split_index"]),
            str(record["fold"]),
            int(record["first_fit_seed"]),
            int(record["second_fit_seed"]),
        )
        for record in existing_pairs
        if checkpoint_configuration_matches(record, recording_name)
    }
    expected_pair_keys = {
        (split_index, fold, first_seed, second_seed)
        for split_index in range(N_BLOCK_ALLOCATIONS)
        for fold in ("A", "B")
        for first_seed, second_seed in combinations(FIT_SEEDS, 2)
    }
    can_reuse = (
        REUSE_COMPLETE_RESULTS
        and len(existing_cells) == expected_cells_per_recording
        and len(existing_pairs) == expected_pairs_per_recording
        and complete_keys == expected_keys
        and complete_pair_keys == expected_pair_keys
    )
    if can_reuse:
        print(
            f"[{recording_index + 1}/{len(RECORDINGS)}] {recording_name}: "
            "reusing complete checkpoint",
            flush=True,
        )
        continue

    cell_records = [
        record for record in cell_records if record["recording"] != recording_name
    ]
    seed_pair_records = [
        record for record in seed_pair_records if record["recording"] != recording_name
    ]

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
    n_recorded, _n_frames = activity.shape
    lag_frames = round(PAPER_LAG_SECONDS * fs)
    block_frames = round(BLOCK_SECONDS * fs)
    guard_frames = round(BLOCK_EDGE_GUARD_SECONDS * fs)

    full_valid = rmt.valid_activity_rows(activity)
    primary_mask = full_valid & nonzero_roi
    primary_rows = np.flatnonzero(primary_mask)
    selected_activity = np.ascontiguousarray(activity[primary_rows], dtype=np.float32)
    del activity
    gc.collect()

    segments = primary_segments(state, boundary_ind, allowed_codes)
    blocks = guarded_blocks(segments, block_frames, guard_frames)
    allocations = []
    allocation_fingerprints = set()
    for split_index in range(N_BLOCK_ALLOCATIONS):
        split_seed = SPLIT_SEED_BASE + recording_index + SPLIT_SEED_STRIDE * split_index
        fold_a_blocks, fold_b_blocks, blocks_per_fold = matched_fold_blocks(
            blocks,
            allowed_codes,
            seed=split_seed,
        )
        fingerprint = tuple(
            sorted(
                (
                    block_fingerprint(fold_a_blocks),
                    block_fingerprint(fold_b_blocks),
                )
            )
        )
        if fingerprint in allocation_fingerprints:
            raise RuntimeError("Repeated split seeds produced duplicate allocations")
        allocation_fingerprints.add(fingerprint)
        allocations.append(
            (
                split_index,
                split_seed,
                fold_a_blocks,
                fold_b_blocks,
                blocks_per_fold,
            )
        )

    common_valid = np.ones(primary_rows.size, dtype=bool)
    for _index, _seed, fold_a_blocks, fold_b_blocks, _counts in allocations:
        common_valid &= valid_rows_in_blocks(selected_activity, fold_a_blocks)
        common_valid &= valid_rows_in_blocks(selected_activity, fold_b_blocks)
    selected_activity = np.ascontiguousarray(
        selected_activity[common_valid],
        dtype=np.float32,
    )
    fit_neurons = selected_activity.shape[0]
    if fit_neurons < MIN_NEURONS_FOR_FIT:
        raise RuntimeError(
            f"{recording_name}: only {fit_neurons} neurons survive common validity"
        )
    print(
        f"  fixed population: {fit_neurons:,}/{n_recorded:,} recorded neurons "
        f"({primary_rows.size:,} finite nonzero_ROI before common-fold QC)",
        flush=True,
    )

    for (
        split_index,
        split_seed,
        fold_a_blocks,
        fold_b_blocks,
        blocks_per_fold,
    ) in allocations:
        fold_a_matrix, fold_a_counts, fold_a_slices = matrix_from_blocks(
            selected_activity,
            fold_a_blocks,
            separator_frames=lag_frames,
        )
        fold_b_matrix, fold_b_counts, fold_b_slices = matrix_from_blocks(
            selected_activity,
            fold_b_blocks,
            separator_frames=lag_frames,
        )
        if not np.all(rmt.valid_activity_rows(fold_a_matrix)) or not np.all(
            rmt.valid_activity_rows(fold_b_matrix)
        ):
            raise RuntimeError("The common validity mask failed on a joined fold")

        split_summaries: dict[int, tuple[FitSummary, FitSummary]] = {}
        for fit_seed in FIT_SEEDS:
            fold_a_model, fold_a_summary = fit_matrix(
                fold_a_matrix,
                lag_frames,
                seed=fit_seed,
            )
            fold_b_model, fold_b_summary = fit_matrix(
                fold_b_matrix,
                lag_frames,
                seed=fit_seed,
            )
            verify_normalized_separators(fold_a_model, fold_a_slices)
            verify_normalized_separators(fold_b_model, fold_b_slices)
            for fitted_model in (fold_a_model, fold_b_model):
                replay = node_similarity_from_temporal_scores(
                    fitted_model,
                    transferred_temporal_scores(fitted_model, fitted_model.X),
                )
                if not np.allclose(
                    replay,
                    fitted_model.cc,
                    atol=2e-5,
                    rtol=2e-5,
                ):
                    raise RuntimeError("Transfer math did not replay training cc")

            fine = fine_order_record(
                fold_a_summary,
                fold_b_summary,
                # Common random numbers keep tie-breaking noise out of the
                # split×seed variance attribution. Matching A/B fit seeds are
                # intentional paired initialization, not independent fits.
                metric_seed=100_000 + recording_index,
            )
            pca_overlap, pca_a_to_b, pca_b_to_a = pca_transfer_efficiencies(
                fold_a_model,
                fold_b_model,
            )
            objective_a_to_b = coarse_transfer_record(
                fold_a_model,
                fold_b_model,
                fold_a_counts,
                fold_b_slices,
            )
            objective_b_to_a = coarse_transfer_record(
                fold_b_model,
                fold_a_model,
                fold_b_counts,
                fold_a_slices,
            )

            cell_record: dict[str, object] = {
                "recording": recording_name,
                "condition": condition,
                "mouse": mouse_id,
                **configuration_record(source_path),
                "recorded_neurons": n_recorded,
                "dataset_active_neurons": primary_rows.size,
                "common_fold_fit_neurons": fit_neurons,
                "split_index": split_index,
                "split_seed": split_seed,
                "fit_seed": fit_seed,
                "lag_frames": lag_frames,
                "state0_label": code_labels[allowed_codes[0]],
                "state1_label": code_labels[allowed_codes[1]],
                "fold_a_state0_blocks": blocks_per_fold[allowed_codes[0]],
                "fold_a_state1_blocks": blocks_per_fold[allowed_codes[1]],
                "fold_b_state0_blocks": blocks_per_fold[allowed_codes[0]],
                "fold_b_state1_blocks": blocks_per_fold[allowed_codes[1]],
                "fold_abs_spearman": fine["abs_spearman"],
                "fold_local_overlap_raw": fine["local_overlap_raw"],
                "fold_local_overlap_adjusted": fine["local_overlap_adjusted"],
                "fold_cluster_ari": fine["cluster_ari"],
                "pca_subspace_overlap_adjusted": pca_overlap,
                "pca_transfer_a_to_b_efficiency": pca_a_to_b,
                "pca_transfer_b_to_a_efficiency": pca_b_to_a,
                "fold_a_runtime_seconds": fold_a_summary.runtime_seconds,
                "fold_b_runtime_seconds": fold_b_summary.runtime_seconds,
            }
            for prefix, values in (
                ("objective_a_to_b", objective_a_to_b),
                ("objective_b_to_a", objective_b_to_a),
            ):
                for field, value in values.items():
                    cell_record[f"{prefix}_{field}"] = value
            for comparator in (
                "reversed",
                "activity",
                "random_expectation",
                "blockwise_time_reversal",
            ):
                cell_record[f"objective_reciprocal_learned_minus_{comparator}"] = float(
                    np.mean(
                        [
                            objective_a_to_b["learned"] - objective_a_to_b[comparator],
                            objective_b_to_a["learned"] - objective_b_to_a[comparator],
                        ]
                    )
                )
            cell_record["objective_reciprocal_zero_lag_fraction"] = float(
                np.mean(
                    [
                        objective_a_to_b["zero_lag"] / objective_a_to_b["learned"],
                        objective_b_to_a["zero_lag"] / objective_b_to_a["learned"],
                    ]
                )
            )
            cell_record["objective_reciprocal_delayed_fraction"] = float(
                np.mean(
                    [
                        objective_a_to_b["delayed_only_lags_1_to_L"]
                        / objective_a_to_b["learned"],
                        objective_b_to_a["delayed_only_lags_1_to_L"]
                        / objective_b_to_a["learned"],
                    ]
                )
            )
            cell_records.append(cell_record)
            split_summaries[fit_seed] = (fold_a_summary, fold_b_summary)
            print(
                f"  split {split_index + 1}/{N_BLOCK_ALLOCATIONS}, "
                f"seed {fit_seed}: fold |ρ|={fine['abs_spearman']:.3f}, "
                f"local={fine['local_overlap_adjusted']:.3f}, "
                "coarse learned−random="
                f"{cell_record['objective_reciprocal_learned_minus_random_expectation']:.4f}",
                flush=True,
            )
            del fold_a_model, fold_b_model
            gc.collect()

        for fold_index, fold_name in enumerate(("A", "B")):
            for first_seed, second_seed in combinations(FIT_SEEDS, 2):
                pair_fine = fine_order_record(
                    split_summaries[first_seed][fold_index],
                    split_summaries[second_seed][fold_index],
                    metric_seed=200_000 + recording_index * 10 + fold_index,
                )
                seed_pair_records.append(
                    {
                        "recording": recording_name,
                        "condition": condition,
                        "mouse": mouse_id,
                        **configuration_record(source_path),
                        "split_index": split_index,
                        "split_seed": split_seed,
                        "fold": fold_name,
                        "first_fit_seed": first_seed,
                        "second_fit_seed": second_seed,
                        "common_fold_fit_neurons": fit_neurons,
                        "same_input_abs_spearman": pair_fine["abs_spearman"],
                        "same_input_local_overlap_raw": pair_fine["local_overlap_raw"],
                        "same_input_local_overlap_adjusted": pair_fine[
                            "local_overlap_adjusted"
                        ],
                        "same_input_cluster_ari": pair_fine["cluster_ari"],
                    }
                )

        del (
            fold_a_matrix,
            fold_b_matrix,
            fold_a_counts,
            fold_b_counts,
            split_summaries,
        )
        gc.collect()

    save_records(CELL_PATH, cell_records)
    save_records(SEED_PAIR_PATH, seed_pair_records)
    del selected_activity, state, nonzero_roi
    gc.collect()

print("saved ->", CELL_PATH)
print("saved ->", SEED_PAIR_PATH)


# %% [markdown]
# ## Step 5 — session summaries and variance attribution
#
# Each session is summarized before any mouse- or condition-level description.
# The factor shares below partition variability inside this finite 3×3 grid;
# they are not population variance estimates, confidence intervals, or p-values.

# %%
summary_records: list[dict[str, object]] = []
grid_fields = (
    "fold_abs_spearman",
    "fold_local_overlap_adjusted",
    "pca_subspace_overlap_adjusted",
    "objective_reciprocal_learned_minus_random_expectation",
    "objective_reciprocal_learned_minus_reversed",
    "objective_reciprocal_learned_minus_blockwise_time_reversal",
)

for recording_name in RECORDINGS:
    cells = [record for record in cell_records if record["recording"] == recording_name]
    pairs = [
        record for record in seed_pair_records if record["recording"] == recording_name
    ]
    if len(cells) != expected_cells_per_recording:
        raise RuntimeError(f"{recording_name}: incomplete factorial cell table")
    if len(pairs) != expected_pairs_per_recording:
        raise RuntimeError(f"{recording_name}: incomplete same-input seed table")
    lookup = {
        (int(record["split_index"]), int(record["fit_seed"])): record
        for record in cells
    }
    first = cells[0]
    summary: dict[str, object] = {
        "recording": recording_name,
        "condition": first["condition"],
        "mouse": first["mouse"],
        "recorded_neurons": int(float(first["recorded_neurons"])),
        "dataset_active_neurons": int(float(first["dataset_active_neurons"])),
        "common_fold_fit_neurons": int(float(first["common_fold_fit_neurons"])),
        "n_factorial_cells": len(cells),
        "n_same_input_seed_pairs": len(pairs),
    }
    for field in grid_fields:
        grid = np.array(
            [
                [numeric(lookup[(split_index, seed)], field) for seed in FIT_SEEDS]
                for split_index in range(N_BLOCK_ALLOCATIONS)
            ]
        )
        split_share, seed_share, interaction_share = balanced_factor_shares(grid)
        summary[f"{field}_median"] = float(np.median(grid))
        summary[f"{field}_minimum"] = float(grid.min())
        summary[f"{field}_maximum"] = float(grid.max())
        summary[f"{field}_split_marginal_sd"] = float(np.std(grid.mean(axis=1), ddof=1))
        summary[f"{field}_seed_marginal_sd"] = float(np.std(grid.mean(axis=0), ddof=1))
        summary[f"{field}_split_share"] = split_share
        summary[f"{field}_seed_share"] = seed_share
        summary[f"{field}_interaction_share"] = interaction_share

    same_seed_spearman = np.array(
        [numeric(record, "same_input_abs_spearman") for record in pairs]
    )
    same_seed_local = np.array(
        [numeric(record, "same_input_local_overlap_adjusted") for record in pairs]
    )
    summary["same_input_seed_abs_spearman_median"] = float(
        np.median(same_seed_spearman)
    )
    summary["same_input_seed_abs_spearman_minimum"] = float(same_seed_spearman.min())
    summary["same_input_seed_abs_spearman_maximum"] = float(same_seed_spearman.max())
    summary["same_input_seed_local_adjusted_median"] = float(np.median(same_seed_local))
    summary["same_input_seed_local_adjusted_minimum"] = float(same_seed_local.min())
    summary["same_input_seed_local_adjusted_maximum"] = float(same_seed_local.max())
    summary_records.append(summary)

save_records(SUMMARY_PATH, summary_records)
print("saved ->", SUMMARY_PATH)


# %% [markdown]
# ## Step 6 — visualization
#
# Dots are individual split×seed cells or same-input seed pairs. Heavy markers
# show session medians and vertical lines show full ranges. The variance-share
# panel describes which perturbation dominates this small factorial grid.

# %%
plot_x = np.arange(len(RECORDINGS))
plot_labels = [name.replace("mouse", "m") for name in RECORDINGS]
condition_colors = np.array(
    [
        "tab:blue" if record["condition"] == "sleep" else "tab:orange"
        for record in summary_records
    ]
)


def plot_cells_with_summary(
    axis,
    field: str,
    summary_stem: str,
    ylabel: str,
    title: str,
) -> None:
    """Plot all factorial cells plus per-session median and range."""
    for index, recording_name in enumerate(RECORDINGS):
        values = np.array(
            [
                numeric(record, field)
                for record in cell_records
                if record["recording"] == recording_name
            ]
        )
        offsets = np.linspace(-0.20, 0.20, values.size)
        axis.scatter(
            index + offsets,
            values,
            color=condition_colors[index],
            s=13,
            alpha=0.34,
            edgecolors="none",
        )
        summary = summary_records[index]
        median = numeric(summary, f"{summary_stem}_median")
        minimum = numeric(summary, f"{summary_stem}_minimum")
        maximum = numeric(summary, f"{summary_stem}_maximum")
        axis.errorbar(
            index,
            median,
            yerr=np.array([[median - minimum], [maximum - median]]),
            fmt="o",
            color=condition_colors[index],
            mec="black",
            mew=0.4,
            capsize=3,
            zorder=3,
        )
    axis.set_ylabel(ylabel)
    axis.set_title(title)


figure, axes = plt.subplots(2, 3, figsize=(17, 9.5), constrained_layout=True)
plot_cells_with_summary(
    axes[0, 0],
    "fold_abs_spearman",
    "fold_abs_spearman",
    "cross-fold |Spearman ρ|",
    "Fine global order across disjoint folds",
)
plot_cells_with_summary(
    axes[0, 1],
    "fold_local_overlap_adjusted",
    "fold_local_overlap_adjusted",
    "adjusted local overlap",
    "Fine local order across disjoint folds",
)

for index, recording_name in enumerate(RECORDINGS):
    pair_values = np.array(
        [
            numeric(record, "same_input_abs_spearman")
            for record in seed_pair_records
            if record["recording"] == recording_name
        ]
    )
    fold_values = np.array(
        [
            numeric(record, "fold_abs_spearman")
            for record in cell_records
            if record["recording"] == recording_name
        ]
    )
    axes[0, 2].plot(
        [index - 0.12, index + 0.12],
        [np.median(pair_values), np.median(fold_values)],
        color="0.75",
        lw=0.8,
        zorder=1,
    )
    axes[0, 2].scatter(
        index - 0.12,
        np.median(pair_values),
        marker="o",
        color=condition_colors[index],
        edgecolor="black",
        linewidth=0.4,
        zorder=2,
    )
    axes[0, 2].scatter(
        index + 0.12,
        np.median(fold_values),
        marker="s",
        color=condition_colors[index],
        edgecolor="black",
        linewidth=0.4,
        zorder=2,
    )
axes[0, 2].scatter([], [], marker="o", color="0.5", label="same input, new seed")
axes[0, 2].scatter([], [], marker="s", color="0.5", label="new time fold")
axes[0, 2].set_ylabel("median |Spearman ρ|")
axes[0, 2].set_title("Same-input seed versus disjoint-time agreement")
axes[0, 2].legend(frameon=False, fontsize=8)

rho_split = np.array(
    [numeric(record, "fold_abs_spearman_split_share") for record in summary_records]
)
rho_seed = np.array(
    [numeric(record, "fold_abs_spearman_seed_share") for record in summary_records]
)
rho_interaction = np.array(
    [
        numeric(record, "fold_abs_spearman_interaction_share")
        for record in summary_records
    ]
)
axes[1, 0].bar(plot_x, rho_split, color="tab:blue", label="block allocation")
axes[1, 0].bar(
    plot_x,
    rho_seed,
    bottom=rho_split,
    color="tab:orange",
    label="fit seed",
)
axes[1, 0].bar(
    plot_x,
    rho_interaction,
    bottom=rho_split + rho_seed,
    color="0.6",
    label="split × seed",
)
axes[1, 0].set_ylim(0, 1.03)
axes[1, 0].set_ylabel("share of 3×3 grid variability")
axes[1, 0].set_title("Descriptive sources of global-order variability")
axes[1, 0].legend(frameon=False, fontsize=8)

coarse_fields = (
    (
        "objective_reciprocal_learned_minus_random_expectation_median",
        "learned − exact random mean",
        "o-",
    ),
    (
        "objective_reciprocal_learned_minus_reversed_median",
        "learned − reversed order",
        "s-",
    ),
    (
        "objective_reciprocal_learned_minus_blockwise_time_reversal_median",
        "learned − reversed time",
        "^-",
    ),
)
for field, label, style in coarse_fields:
    axes[1, 1].plot(
        plot_x,
        [numeric(record, field) for record in summary_records],
        style,
        label=label,
    )
axes[1, 1].axhline(0, color="black", lw=0.8)
axes[1, 1].set_ylabel("median reciprocal objective difference")
axes[1, 1].set_title("Coarse organization versus directional excess")
axes[1, 1].legend(frameon=False, fontsize=8)

plot_cells_with_summary(
    axes[1, 2],
    "pca_subspace_overlap_adjusted",
    "pca_subspace_overlap_adjusted",
    "adjusted subspace overlap",
    "Low-dimensional axes across disjoint folds",
)

for axis in axes.ravel():
    axis.set_xticks(plot_x, plot_labels, rotation=45, ha="right")
    axis.grid(axis="y", color="0.9", lw=0.6)

figure.suptitle(
    "Dataset-active Rastermap resampling · 3 block allocations × 3 fit seeds"
)
figure_path = FIG_DIR / "06_rastermap_resampling.png"
figure.savefig(figure_path, dpi=150, bbox_inches="tight")
print("saved ->", figure_path)
if SHOW_FIGURES:
    plt.show()


# %% [markdown]
# ## Step 7 — mouse-level descriptive readout and guardrails
#
# Mouse 4's two sleep days are averaged before cohort summaries. These ranges
# describe the small observed mouse cohorts and are not inferential intervals.
#
# Interpretation guardrails:
#
# - High same-input seed agreement with low cross-fold agreement means the
#   optimizer can reproduce an order on fixed data, but the order is not stable
#   to which time blocks were observed.
# - A positive learned-minus-random objective supports coarse transferable
#   organization. A much smaller learned-minus-reversal or
#   learned-minus-time-reversal effect does not establish a unique direction.
# - `nonzero_ROI` is a documented dataset-active population, but it is tied to
#   the publication's analysis windows. It is not the Rastermap paper's
#   unpublished Figure 3 firing-rate cutoff.
# - Whole-recording selection, across-all-fold validity, and target-side
#   normalization make this conditional/transductive verification.
# - Only awake+NREM or awake+anesthesia and complete guarded blocks enter the
#   fits. REM, quiet awake, short bouts, and block tails are omitted.

# %%
mouse_records: list[dict[str, object]] = []
for condition in ("sleep", "anesthesia"):
    condition_records = [
        record for record in summary_records if record["condition"] == condition
    ]
    for mouse in sorted({str(record["mouse"]) for record in condition_records}):
        sessions = [record for record in condition_records if record["mouse"] == mouse]
        mouse_records.append(
            {
                "condition": condition,
                "mouse": mouse,
                **{
                    field: float(
                        np.mean([numeric(record, field) for record in sessions])
                    )
                    for field in (
                        "fold_abs_spearman_median",
                        "fold_local_overlap_adjusted_median",
                        "same_input_seed_abs_spearman_median",
                        "same_input_seed_local_adjusted_median",
                        "objective_reciprocal_learned_minus_random_expectation_median",
                        "objective_reciprocal_learned_minus_reversed_median",
                        "objective_reciprocal_learned_minus_blockwise_time_reversal_median",
                        "fold_abs_spearman_split_share",
                        "fold_abs_spearman_seed_share",
                        "fold_abs_spearman_interaction_share",
                    )
                },
            }
        )

for condition in ("sleep", "anesthesia"):
    cohort = [record for record in mouse_records if record["condition"] == condition]
    print(f"\n{condition.capitalize()} mouse-level description (n={len(cohort)}):")
    for field in (
        "same_input_seed_abs_spearman_median",
        "fold_abs_spearman_median",
        "fold_local_overlap_adjusted_median",
        "objective_reciprocal_learned_minus_random_expectation_median",
        "objective_reciprocal_learned_minus_reversed_median",
        "fold_abs_spearman_split_share",
        "fold_abs_spearman_seed_share",
        "fold_abs_spearman_interaction_share",
    ):
        values = np.array([numeric(record, field) for record in cohort])
        print(
            f"  {field}: median={np.median(values):.3f}, "
            f"range={values.min():.3f}…{values.max():.3f}"
        )

print("\nRepeated split × seed Rastermap verification complete.")
