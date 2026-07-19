# %% [markdown]
# # 04 · Does the Rastermap organization generalize?
#
# Tutorial 03 verifies that the official Rastermap implementation receives the
# intended arrays and reproduces its internal normalization, PCA, clustering,
# sorting, and upsampling.  This tutorial asks the harder scientific question:
# **is the resulting organization stable and stronger than appropriate nulls?**
#
# The cells deliberately separate six questions:
#
# 1. Which neurons are active enough to z-score safely?
# 2. Does the result survive different scaled-k-means seeds?
# 3. Does an order learned in one time period describe independent timepoints?
# 4. How sensitive is it to activity threshold and algorithm parameters?
# 5. Is there reproducible low-dimensional shared activity even if the exact
#    one-dimensional neuron order is unstable?
# 6. Can this exact code recover a known synthetic sequence and fail after the
#    coordination is destroyed?
#
# Spatial localization is intentionally outside this tutorial.

# %% Step 0 — imports
from __future__ import annotations

import csv
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
from rastermap import Rastermap
from rastermap.cluster import compute_cc_tdelay
from rastermap.svd import SVD
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import svd
from scipy.stats import kendalltau, rankdata, spearmanr
from sklearn.metrics import adjusted_rand_score

from src.funcnet import dataio, rastermap_tools as rmt, timeseries as ts
from src.funcnet.paths import FIG_DIR, RESULTS_DIR

FIG_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_DIR = RESULTS_DIR / "rastermap_validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Step 1 — paper-motivated working settings
#
# Stringer et al. state that their applications used minimum firing rates of
# 0.1--0.25 Hz because per-neuron z-scoring can amplify low-rate neurons.  Their
# public Figure 3 notebook starts from an already processed 34,086-neuron matrix
# and does not disclose the exact upstream cutoff.
#
# Here `MIN_POSITIVE_BIN_RATE_HZ` means the number of positive OASIS
# deconvolution bins per recorded second.  It is an explicit activity proxy,
# **not a calibrated physiological firing rate**.  The primary 0.10 choice is
# treated as a fixed working criterion in this tutorial, and a later cell shows
# sensitivity to several alternatives. This is exploratory verification, not a
# preregistered threshold choice.

# %%
RECORDING_NAME = "mouse01_sleep"
MIN_POSITIVE_BIN_RATE_HZ = 0.10
THRESHOLD_COUNT_GRID = (0.025, 0.05, 0.075, 0.10, 0.15, 0.25)
THRESHOLD_FIT_GRID = (0.025, 0.05, 0.10, 0.15)

N_CLUSTERS = 100
N_PCS = 128
LOCALITY = 0.0
PAPER_SAMPLING_HZ = 3.2
PAPER_LAG_FRAMES = 5
PRIMARY_LAG_SECONDS = PAPER_LAG_FRAMES / PAPER_SAMPLING_HZ
MEAN_TIME = True
SUPERNEURON_SIZE = 50
REFERENCE_SEED = 0
SEEDS = tuple(range(20))
DISPLAY_SEEDS = (1, 2, 3, 4)

NEIGHBORHOOD_SIZE = 50
N_RANDOM_ORDERS = 25
N_SHIFT_NULLS = 19
N_PAIR_SAMPLES = 100_000
PAIR_CHUNK_SIZE = 512
TIE_PERMUTATIONS = 8
MIN_STATE_HELDOUT_FRAMES = 100
N_OBJECTIVE_PERMUTATIONS = 999

PARAMETER_VARIANTS = (
    ("PCs 64", 64, 100, None, 0.0),
    ("reference", 128, 100, None, 0.0),
    ("PCs 256", 256, 100, None, 0.0),
    ("clusters 50", 128, 50, None, 0.0),
    ("clusters 150", 128, 150, None, 0.0),
    ("lag 0", 128, 100, 0, 0.0),
    ("lag 5", 128, 100, 5, 0.0),
    ("lag 20", 128, 100, 20, 0.0),
    ("locality .25", 128, 100, None, 0.25),
    ("locality .50", 128, 100, None, 0.50),
)

SYNTHETIC_NEURONS = 1_200
SYNTHETIC_FRAMES = 4_096
SYNTHETIC_REPEATS = 5
SYNTHETIC_WAVES = 20
SYNTHETIC_N_CLUSTERS = 50
SYNTHETIC_N_PCS = 64
SYNTHETIC_LOCALITY = 0.75
SYNTHETIC_LAG_FRAMES = 8

SVCA_COMPONENTS = 512
SVCA_REPEATS = 20
SVCA_BLOCK_SECONDS = 15.0
SVCA_GUARD_SECONDS = 2.0

RUN_THRESHOLD_FITS = True
RUN_PARAMETER_SWEEP = True
RUN_SVCA = True
RUN_SYNTHETIC_CONTROL = True
SHOW_FIGURES = True


# %% [markdown]
# ## Helper functions used by the validation cells
#
# `FitSummary` keeps only compact, row-aligned outputs from secondary fits.
# Full normalized matrices are retained only for the reference and two temporal
# fits. Rank/neighborhood comparisons are orientation-free, while the explicit
# lag-aware held-out objective below retains Rastermap's learned direction.


# %%
@dataclass
class FitSummary:
    """Compact Rastermap result aligned to a shared reference population."""

    embedding: np.ndarray
    clusters: np.ndarray
    order: np.ndarray
    runtime_seconds: float


def lag_pair_fraction_crossing_boundaries(
    n_timepoints: int,
    boundaries: np.ndarray,
    maximum_lag: int,
) -> float:
    """Return the fraction of lagged pairs that cross acquisition breaks."""
    split_frames = np.asarray(boundaries, dtype=np.int64).ravel() + 1
    split_frames = split_frames[(split_frames > 0) & (split_frames < n_timepoints)]
    crossing_pairs = 0
    total_pairs = 0
    for lag in range(maximum_lag + 1):
        total_pairs += n_timepoints - lag
        if lag == 0:
            continue
        for split in split_frames:
            crossing_pairs += max(
                0,
                min(split, n_timepoints - lag) - max(0, split - lag),
            )
    return crossing_pairs / total_pairs if total_pairs else 0.0


def summarize_model(
    fitted_model: Rastermap,
    input_rows: np.ndarray,
    population_size: int,
) -> FitSummary:
    """Map one fit back to row IDs in a shared comparison population."""
    input_rows = np.asarray(input_rows, dtype=np.int64)
    good = np.asarray(fitted_model.igood, dtype=bool).ravel()
    if good.size != input_rows.size:
        raise RuntimeError("Rastermap returned an unexpected validity mask")

    embedding = np.full(population_size, np.nan, dtype=np.float32)
    embedding[input_rows] = np.asarray(fitted_model.embedding).ravel()
    clusters = np.full(population_size, -1, dtype=np.int32)
    clusters[input_rows[good]] = np.asarray(
        fitted_model.embedding_clust,
        dtype=np.int32,
    )
    local_order = np.asarray(fitted_model.isort, dtype=np.int64)
    local_order = local_order[good[local_order]]
    order = input_rows[local_order]
    return FitSummary(
        embedding=embedding,
        clusters=clusters,
        order=order,
        runtime_seconds=float(fitted_model.runtime),
    )


def remap_summary(
    summary: FitSummary,
    original_rows: np.ndarray,
    population_size: int,
) -> FitSummary:
    """Map a subset-local fit summary back to original recorded ROI rows."""
    original_rows = np.asarray(original_rows, dtype=np.int64)
    if summary.embedding.size != original_rows.size:
        raise ValueError("summary and original_rows do not describe the same subset")
    embedding = np.full(population_size, np.nan, dtype=np.float32)
    embedding[original_rows] = summary.embedding
    clusters = np.full(population_size, -1, dtype=np.int32)
    clusters[original_rows] = summary.clusters
    return FitSummary(
        embedding=embedding,
        clusters=clusters,
        order=original_rows[summary.order],
        runtime_seconds=summary.runtime_seconds,
    )


def fit_activity(
    activity_matrix: np.ndarray,
    *,
    n_clusters: int,
    n_pcs: int,
    locality: float,
    lag_frames: int,
    seed: int,
    keep_normalized: bool,
) -> tuple[Rastermap, FitSummary]:
    """Fit a fresh model after removing split-specific constant rows."""
    valid = rmt.valid_activity_rows(activity_matrix)
    input_rows = np.flatnonzero(valid)
    working = np.ascontiguousarray(activity_matrix[input_rows], dtype=np.float32)
    fitted_model = Rastermap(
        n_clusters=n_clusters,
        n_PCs=n_pcs,
        locality=locality,
        time_lag_window=lag_frames,
        time_bin=1,
        mean_time=MEAN_TIME,
        bin_size=SUPERNEURON_SIZE,
        random_state=seed,
        keep_norm_X=keep_normalized,
        verbose=False,
    ).fit(working, compute_X_embedding=False)
    summary = summarize_model(fitted_model, input_rows, activity_matrix.shape[0])
    return fitted_model, summary


def fit_from_decomposition(
    Usv: np.ndarray,
    Vsv: np.ndarray,
    *,
    n_clusters: int,
    locality: float,
    lag_frames: int,
    seed: int,
) -> tuple[Rastermap, FitSummary]:
    """Refit clustering/sorting while keeping normalization and PCA fixed."""
    fitted_model = Rastermap(
        n_clusters=n_clusters,
        n_PCs=Usv.shape[1],
        locality=locality,
        time_lag_window=lag_frames,
        mean_time=False,
        normalize=False,
        random_state=seed,
        keep_norm_X=False,
        verbose=False,
    ).fit(
        data=None,
        Usv=np.ascontiguousarray(Usv),
        Vsv=np.ascontiguousarray(Vsv),
        compute_X_embedding=False,
    )
    rows = np.arange(Usv.shape[0], dtype=np.int64)
    return fitted_model, summarize_model(fitted_model, rows, Usv.shape[0])


def common_rows(first: FitSummary, second: FitSummary) -> np.ndarray:
    """Rows with finite embeddings in both fits."""
    return np.flatnonzero(np.isfinite(first.embedding) & np.isfinite(second.embedding))


def subset_embedding(summary: FitSummary, rows: np.ndarray) -> np.ndarray:
    """Return one compact embedding in the supplied common-row order."""
    return np.asarray(summary.embedding[rows], dtype=np.float64)


def compare_fits(first: FitSummary, second: FitSummary) -> dict[str, float]:
    """Compute reversal-safe global, local, and coarse-cluster agreement."""
    rows = common_rows(first, second)
    first_embedding = subset_embedding(first, rows)
    second_embedding = subset_embedding(second, rows)
    tau = kendalltau(first_embedding, second_embedding, variant="b").statistic
    if not np.isfinite(tau):
        raise RuntimeError("Kendall agreement is undefined")
    rank_correlation = rmt.reversal_invariant_rank_correlation(
        first_embedding,
        second_embedding,
    )
    neighborhood = rmt.rank_neighborhood_overlap(
        first_embedding,
        second_embedding,
        neighborhood_size=min(NEIGHBORHOOD_SIZE, rows.size - 1),
        tie_permutations=TIE_PERMUTATIONS,
        random_state=0,
    )
    chance_overlap = min(NEIGHBORHOOD_SIZE, rows.size - 1) / (rows.size - 1)
    adjusted_neighborhood = (neighborhood - chance_overlap) / (1 - chance_overlap)
    ari = adjusted_rand_score(first.clusters[rows], second.clusters[rows])
    return {
        "n_common": float(rows.size),
        "abs_kendall_tau_b": float(abs(tau)),
        "abs_spearman_rank": float(rank_correlation),
        "neighborhood_overlap": float(neighborhood),
        "adjusted_neighborhood_overlap": float(adjusted_neighborhood),
        "cluster_ari": float(ari),
    }


def embedding_distance_correlation(
    first_embedding: np.ndarray,
    second_embedding: np.ndarray,
    *,
    seed: int = 0,
) -> float:
    """Compare pairwise 1-D distance geometry without choosing an orientation."""
    finite = np.isfinite(first_embedding) & np.isfinite(second_embedding)
    first = rankdata(first_embedding[finite], method="average")
    second = rankdata(second_embedding[finite], method="average")
    n_rows = first.size
    rng = np.random.default_rng(seed)
    first_rows = rng.integers(0, n_rows, size=N_PAIR_SAMPLES)
    second_rows = rng.integers(0, n_rows - 1, size=N_PAIR_SAMPLES)
    second_rows += second_rows >= first_rows
    first_distance = np.abs(first[first_rows] - first[second_rows])
    second_distance = np.abs(second[first_rows] - second[second_rows])
    return float(spearmanr(first_distance, second_distance).statistic)


def superneuron_display(
    normalized_activity_matrix: np.ndarray,
    order: np.ndarray,
) -> np.ndarray:
    """Build a display only; fixed bins are not validation statistics."""
    return rmt.ordered_superneurons(
        normalized_activity_matrix,
        np.asarray(order, dtype=np.int64),
        SUPERNEURON_SIZE,
    )


def independently_shift_rows_within_segments(
    activity_matrix: np.ndarray,
    segments: list[tuple[int, int]],
    *,
    min_shift: int,
    seed: int,
) -> np.ndarray:
    """Destroy synchrony while preserving each state/acquisition segment."""
    rng = np.random.default_rng(seed)
    shifted = np.asarray(activity_matrix).copy()
    for start, stop in segments:
        length = stop - start
        if length <= 2 * min_shift:
            continue
        shifts = rng.integers(
            min_shift,
            length - min_shift + 1,
            size=shifted.shape[0],
        )
        for row, shift in enumerate(shifts):
            shifted[row, start:stop] = np.roll(
                activity_matrix[row, start:stop],
                int(shift),
            )
    return shifted


def sample_distinct_pairs(
    n_rows: int,
    n_pairs: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample row pairs without constructing an all-neuron matrix."""
    rng = np.random.default_rng(seed)
    first = rng.integers(0, n_rows, size=n_pairs, dtype=np.int64)
    second = rng.integers(0, n_rows - 1, size=n_pairs, dtype=np.int64)
    second += second >= first
    return first, second


def sampled_pair_correlations(
    activity_matrix: np.ndarray,
    first_rows: np.ndarray,
    second_rows: np.ndarray,
) -> np.ndarray:
    """Compute Pearson correlations for sampled pairs in bounded memory."""
    activity_matrix = np.asarray(activity_matrix, dtype=np.float32)
    n_frames = activity_matrix.shape[1]
    row_mean = np.mean(activity_matrix, axis=1, dtype=np.float64)
    row_norm = np.sqrt(
        np.maximum(
            0,
            np.sum(activity_matrix * activity_matrix, axis=1, dtype=np.float64)
            - n_frames * row_mean**2,
        )
    )
    correlations = np.empty(first_rows.size, dtype=np.float32)
    for start in range(0, first_rows.size, PAIR_CHUNK_SIZE):
        stop = min(first_rows.size, start + PAIR_CHUNK_SIZE)
        first_chunk = activity_matrix[first_rows[start:stop]]
        second_chunk = activity_matrix[second_rows[start:stop]]
        numerator = np.einsum(
            "ij,ij->i",
            first_chunk,
            second_chunk,
            dtype=np.float64,
        )
        numerator -= (
            n_frames
            * row_mean[first_rows[start:stop]]
            * row_mean[second_rows[start:stop]]
        )
        denominator = (
            row_norm[first_rows[start:stop]] * row_norm[second_rows[start:stop]]
        )
        correlations[start:stop] = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
    return correlations


def pair_distance_similarity_score(
    embedding: np.ndarray,
    first_rows: np.ndarray,
    second_rows: np.ndarray,
    pair_correlations: np.ndarray,
) -> float:
    """Score whether nearby embedding positions correlate on held-out data."""
    embedding = np.asarray(embedding, dtype=np.float64)
    if not np.all(np.isfinite(embedding)):
        raise ValueError("pair score requires a finite embedding for every row")
    midranks = rankdata(embedding, method="average")
    distance = np.abs(midranks[first_rows] - midranks[second_rows])
    score = spearmanr(-distance, pair_correlations).statistic
    if not np.isfinite(score):
        raise RuntimeError("pair-distance score is undefined")
    return float(score)


def transferred_node_similarity(
    source_model: Rastermap,
    target_normalized_activity: np.ndarray,
) -> np.ndarray:
    """Evaluate source-fit sorted cluster templates on untouched target time."""
    source_singular_values = np.asarray(source_model.sv, dtype=np.float32)
    source_left = (
        np.asarray(source_model.Usv, dtype=np.float32) / source_singular_values
    )
    target_temporal_scores = target_normalized_activity.T @ source_left
    return compute_cc_tdelay(
        target_temporal_scores / source_singular_values,
        np.asarray(source_model.U_nodes, dtype=np.float32),
        time_lag_window=int(source_model.time_lag_window),
        symmetric=False,
    )


def rastermap_directional_objective(
    node_similarity: np.ndarray,
    matching_target: np.ndarray,
    cluster_order: np.ndarray,
) -> float:
    """Score one directed cluster order with Rastermap's upper-triangle kernel."""
    node_similarity = np.asarray(node_similarity, dtype=np.float64)
    matching_target = np.asarray(matching_target, dtype=np.float64)
    cluster_order = np.asarray(cluster_order, dtype=np.int64)
    if (
        node_similarity.ndim != 2
        or node_similarity.shape[0] != node_similarity.shape[1]
        or matching_target.shape != node_similarity.shape
    ):
        raise ValueError(
            "node similarity and matching target must be square and aligned"
        )
    if (
        cluster_order.ndim != 1
        or cluster_order.size != node_similarity.shape[0]
        or np.unique(cluster_order).size != cluster_order.size
        or np.any(cluster_order < 0)
        or np.any(cluster_order >= cluster_order.size)
    ):
        raise ValueError("cluster_order must be a complete permutation")
    ordered_similarity = node_similarity[np.ix_(cluster_order, cluster_order)]
    weights = np.triu(matching_target, k=1)
    weight_sum = weights.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("matching target must contain positive upper-triangle weight")
    return float(np.sum(weights * ordered_similarity) / weight_sum)


def activity_cluster_order(
    fitted_model: Rastermap,
    neuron_activity_counts: np.ndarray,
) -> np.ndarray:
    """Freeze the better activity-ranked cluster orientation on training data."""
    assignments = np.asarray(fitted_model.embedding_clust, dtype=np.int64)
    neuron_activity_counts = np.asarray(neuron_activity_counts, dtype=np.float64)
    n_clusters = np.asarray(fitted_model.U_nodes).shape[0]
    if assignments.size != neuron_activity_counts.size:
        raise ValueError("cluster assignments and activity counts must align")
    cluster_activity = np.array(
        [
            neuron_activity_counts[assignments == cluster].mean()
            for cluster in range(n_clusters)
        ]
    )
    ascending = np.argsort(cluster_activity, kind="stable")
    descending = ascending[::-1]
    ascending_score = rastermap_directional_objective(
        fitted_model.cc,
        fitted_model.BBt,
        ascending,
    )
    descending_score = rastermap_directional_objective(
        fitted_model.cc,
        fitted_model.BBt,
        descending,
    )
    return ascending if ascending_score >= descending_score else descending


def sum_of_squares(activity_matrix: np.ndarray, chunk_frames: int = 512) -> float:
    """Accumulate matrix energy without making a full float64 copy."""
    total = 0.0
    for start in range(0, activity_matrix.shape[1], chunk_frames):
        total += float(
            np.sum(
                np.square(activity_matrix[:, start : start + chunk_frames]),
                dtype=np.float64,
            )
        )
    return total


def save_records(path: Path, records: list[dict[str, object]]) -> None:
    """Save a list of like-shaped validation records as a readable CSV."""
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("saved ->", path)


# %% [markdown]
# ## Step 2 — load the complete recording and audit active-neuron definitions
#
# Both positive-bin and stricter 0→positive onset rates are displayed.  The
# primary fit uses the former.  Selection is computed once over the complete
# recording before any temporal split, preventing post-hoc neuron selection on
# a later validation block.

# %%
recording = dataio.load_recording(RECORDING_NAME)
recording_name = recording.name
fs = recording.fs
n_frames = recording.n_frames
n_recorded_neurons = recording.n_neurons
duration_seconds = n_frames / fs
state = recording.state.astype(np.float32, copy=True)
state_code_labels = dict(dataio.state_codes(recording))
boundary_ind = np.asarray(recording.boundary_ind, dtype=np.int64).copy()
all_activity = np.ascontiguousarray(recording.spike_deconv, dtype=np.float32)
dataset_nonzero_mask = (
    recording.nonzero_ROI.copy()
    if recording.nonzero_ROI is not None
    else np.ones(n_recorded_neurons, dtype=bool)
)
del recording
gc.collect()

positive_bin_rate_hz = rmt.positive_deconvolution_bin_rates(all_activity, fs)
positive_onset_rate_hz = rmt.positive_deconvolution_onset_rates(all_activity, fs)
valid_recorded_rows = rmt.valid_activity_rows(all_activity)
primary_active_mask = rmt.active_deconvolution_rows(
    all_activity,
    fs,
    min_positive_bin_rate_hz=MIN_POSITIVE_BIN_RATE_HZ,
)
primary_roi_rows = np.flatnonzero(primary_active_mask)
activity = np.ascontiguousarray(all_activity[primary_roi_rows], dtype=np.float32)

threshold_counts = np.array(
    [
        np.count_nonzero(valid_recorded_rows & (positive_bin_rate_hz >= threshold))
        for threshold in THRESHOLD_COUNT_GRID
    ]
)
onset_threshold_counts = np.array(
    [
        np.count_nonzero(valid_recorded_rows & (positive_onset_rate_hz >= threshold))
        for threshold in THRESHOLD_COUNT_GRID
    ]
)

# The whole-session mask can underrepresent neurons active mainly in a short
# state. Audit that dependence explicitly; these alternative masks are not fed
# back into the primary fit.
all_state_codes = np.unique(state[np.isfinite(state)])
state_selection_records: list[dict[str, object]] = []
state_active_masks: dict[float, np.ndarray] = {}
for code in all_state_codes:
    state_frames = np.flatnonzero(state == code)
    state_activity = np.ascontiguousarray(
        all_activity[:, state_frames],
        dtype=np.float32,
    )
    state_active_mask = rmt.active_deconvolution_rows(
        state_activity,
        fs,
        min_positive_bin_rate_hz=MIN_POSITIVE_BIN_RATE_HZ,
    )
    state_active_masks[float(code)] = state_active_mask
    state_and_global = state_active_mask & primary_active_mask
    state_selection_records.append(
        {
            "state_code": float(code),
            "state_label": state_code_labels.get(float(code), str(code)),
            "state_frames": state_frames.size,
            "state_duration_seconds": state_frames.size / fs,
            "selected_within_state": int(state_active_mask.sum()),
            "selected_in_state_and_global": int(state_and_global.sum()),
            "selected_within_state_not_global": int(
                np.count_nonzero(state_active_mask & ~primary_active_mask)
            ),
            "selected_global_not_within_state": int(
                np.count_nonzero(primary_active_mask & ~state_active_mask)
            ),
        }
    )
    del state_activity
    gc.collect()

save_records(
    VALIDATION_DIR / "04_state_activity_selection.csv",
    state_selection_records,
)

print(
    f"{recording_name}: selected {activity.shape[0]:,}/{n_recorded_neurons:,} "
    f"neurons at ≥{MIN_POSITIVE_BIN_RATE_HZ:.2f} positive bins/s"
)
print(f"Dataset nonzero_ROI comparison: {dataset_nonzero_mask.sum():,}")
print(
    "Five paper lag frames correspond to "
    f"{PRIMARY_LAG_SECONDS:.3f} s; this recording needs "
    f"{round(PRIMARY_LAG_SECONDS * fs)} frames for the same duration."
)

activity_figure, activity_axes = plt.subplots(
    2,
    3,
    figsize=(18, 9),
    constrained_layout=True,
)
positive_rates_for_log = np.maximum(positive_bin_rate_hz, 1 / duration_seconds)
activity_axes[0, 0].hist(
    positive_rates_for_log,
    bins=np.geomspace(1 / duration_seconds, positive_bin_rate_hz.max(), 60),
    color="0.35",
)
activity_axes[0, 0].set_xscale("log")
activity_axes[0, 0].axvline(
    MIN_POSITIVE_BIN_RATE_HZ,
    color="tab:red",
    label="primary threshold",
)
activity_axes[0, 0].set_xlabel("positive deconvolution bins per second")
activity_axes[0, 0].set_ylabel("neurons")
activity_axes[0, 0].set_title("Complete-session activity proxy")
activity_axes[0, 0].legend(frameon=False)

activity_axes[0, 1].plot(
    THRESHOLD_COUNT_GRID,
    threshold_counts,
    "o-",
    label="positive bins",
)
activity_axes[0, 1].plot(
    THRESHOLD_COUNT_GRID,
    onset_threshold_counts,
    "o-",
    label="0→positive onsets",
)
activity_axes[0, 1].axvline(MIN_POSITIVE_BIN_RATE_HZ, color="tab:red", lw=1)
activity_axes[0, 1].set_xlabel("minimum proxy rate (s⁻¹)")
activity_axes[0, 1].set_ylabel("retained neurons")
activity_axes[0, 1].set_title("Selection is highly definition-dependent")
activity_axes[0, 1].legend(frameon=False)

activity_axes[1, 0].hexbin(
    positive_bin_rate_hz,
    positive_onset_rate_hz,
    gridsize=45,
    mincnt=1,
    bins="log",
    cmap="viridis",
)
activity_axes[1, 0].axvline(MIN_POSITIVE_BIN_RATE_HZ, color="tab:red", lw=1)
activity_axes[1, 0].set_xlabel("positive-bin rate (s⁻¹)")
activity_axes[1, 0].set_ylabel("positive-onset rate (s⁻¹)")
activity_axes[1, 0].set_title("Two non-equivalent activity proxies")

row_standard_deviation = all_activity.std(axis=1)
activity_axes[1, 1].scatter(
    positive_bin_rate_hz[~primary_active_mask],
    row_standard_deviation[~primary_active_mask],
    s=3,
    alpha=0.25,
    color="0.6",
    label="excluded",
    rasterized=True,
)
activity_axes[1, 1].scatter(
    positive_bin_rate_hz[primary_active_mask],
    row_standard_deviation[primary_active_mask],
    s=3,
    alpha=0.35,
    color="tab:blue",
    label="primary active set",
    rasterized=True,
)
activity_axes[1, 1].set_xlabel("positive-bin rate (s⁻¹)")
activity_axes[1, 1].set_ylabel("deconvolved trace SD")
activity_axes[1, 1].set_title("Activity criterion and signal scale")
activity_axes[1, 1].legend(frameon=False, markerscale=2)

selection_state_labels = [
    str(record["state_label"]).replace("_", " ") for record in state_selection_records
]
selection_state_x = np.arange(len(state_selection_records))
activity_axes[0, 2].bar(
    selection_state_x,
    [record["state_duration_seconds"] / 60 for record in state_selection_records],
    color="0.45",
)
activity_axes[0, 2].set_xticks(
    selection_state_x,
    selection_state_labels,
    rotation=20,
)
activity_axes[0, 2].set_ylabel("recorded duration (min)")
activity_axes[0, 2].set_title("Short states contribute fewer selection frames")

state_global_overlap = np.array(
    [record["selected_in_state_and_global"] for record in state_selection_records]
)
state_only = np.array(
    [record["selected_within_state_not_global"] for record in state_selection_records]
)
activity_axes[1, 2].bar(
    selection_state_x,
    state_global_overlap,
    label="also in global set",
    color="tab:blue",
)
activity_axes[1, 2].bar(
    selection_state_x,
    state_only,
    bottom=state_global_overlap,
    label="state-active but absent globally",
    color="tab:orange",
)
activity_axes[1, 2].axhline(
    primary_roi_rows.size,
    color="tab:red",
    lw=1,
    ls="--",
    label="global active-set size",
)
activity_axes[1, 2].set_xticks(
    selection_state_x,
    selection_state_labels,
    rotation=20,
)
activity_axes[1, 2].set_ylabel("neurons")
activity_axes[1, 2].set_title("Whole-session selection is state-composition-dependent")
activity_axes[1, 2].legend(frameon=False, fontsize=8)

activity_figure.suptitle(
    f"{recording_name} · paper-motivated active-neuron selection audit"
)
activity_path = FIG_DIR / "04_rastermap_01_active_selection.png"
activity_figure.savefig(activity_path, dpi=150, bbox_inches="tight")
print("saved ->", activity_path)
if SHOW_FIGURES:
    plt.show()


# %% [markdown]
# %% [markdown]
# ## Step 3 — primary active-neuron fit
#
# The primary lag is matched in physical seconds to the paper's five-frame,
# 3.2-Hz sensorimotor example.  This fit retains `model.X`, PCA scores, and
# temporal components for the later held-out and parameter diagnostics.

# %%
primary_lag_frames = int(round(PRIMARY_LAG_SECONDS * fs))
lag_boundary_pair_fraction = lag_pair_fraction_crossing_boundaries(
    n_frames,
    boundary_ind,
    primary_lag_frames,
)
print(
    f"Acquisition breaks affect {100 * lag_boundary_pair_fraction:.4f}% of "
    f"the lag-0…{primary_lag_frames} frame pairs used by Rastermap; version 1.0 "
    "has no segment-mask argument."
)
primary_model, primary_summary = fit_activity(
    activity,
    n_clusters=N_CLUSTERS,
    n_pcs=N_PCS,
    locality=LOCALITY,
    lag_frames=primary_lag_frames,
    seed=REFERENCE_SEED,
    keep_normalized=True,
)
assert primary_summary.order.size == activity.shape[0]

primary_energy = float(np.sum(primary_model.sv.astype(np.float64) ** 2))
primary_total_energy = sum_of_squares(primary_model.X)
primary_energy_fraction = primary_energy / primary_total_energy
primary_embedding_values = primary_summary.embedding[
    np.isfinite(primary_summary.embedding)
]
_, primary_position_counts = np.unique(
    primary_embedding_values,
    return_counts=True,
)
primary_unique_positions = primary_position_counts.size
primary_tied_neuron_fraction = float(
    primary_position_counts[primary_position_counts > 1].sum()
    / primary_embedding_values.size
)
print(
    f"Primary fit: N={activity.shape[0]:,}, T={n_frames:,}, "
    f"lag={primary_lag_frames} frames ({primary_lag_frames / fs:.2f} s), "
    f"128-PC training energy={100 * primary_energy_fraction:.2f}%"
)
print(
    f"Final grid: {primary_unique_positions:,} occupied positions; "
    f"{100 * primary_tied_neuron_fraction:.1f}% of neurons share a position"
)


# %% [markdown]
# ## Step 4 — scaled-k-means seed stability
#
# The paper used 20 seeds on simulations and identified scaled k-means as the
# main run-to-run source of variation.  We therefore use the same 20-seed count
# here.  Normalization and PCA are held exactly fixed, isolating only clustering,
# sorting, and upsampling.  Exact cluster IDs are secondary; the final 1-D
# embedding and its local neighborhoods are the primary quantities.

# %%
seed_summaries: dict[int, FitSummary] = {REFERENCE_SEED: primary_summary}
for seed in SEEDS:
    if seed == REFERENCE_SEED:
        continue
    seed_model, seed_summary = fit_from_decomposition(
        primary_model.Usv,
        primary_model.Vsv,
        n_clusters=N_CLUSTERS,
        locality=LOCALITY,
        lag_frames=primary_lag_frames,
        seed=seed,
    )
    seed_summaries[seed] = seed_summary
    del seed_model

seed_records: list[dict[str, object]] = []
for seed in SEEDS:
    if seed == REFERENCE_SEED:
        continue
    metrics = compare_fits(primary_summary, seed_summaries[seed])
    metrics["distance_geometry"] = embedding_distance_correlation(
        primary_summary.embedding,
        seed_summaries[seed].embedding,
        seed=seed,
    )
    record: dict[str, object] = {"reference_seed": REFERENCE_SEED, "seed": seed}
    record.update(metrics)
    seed_records.append(record)

seed_tau_matrix = np.eye(len(SEEDS))
seed_overlap_matrix = np.eye(len(SEEDS))
for row, first_seed in enumerate(SEEDS):
    for column, second_seed in enumerate(SEEDS):
        if column <= row:
            continue
        metrics = compare_fits(
            seed_summaries[first_seed],
            seed_summaries[second_seed],
        )
        seed_tau_matrix[row, column] = metrics["abs_kendall_tau_b"]
        seed_tau_matrix[column, row] = seed_tau_matrix[row, column]
        seed_overlap_matrix[row, column] = metrics["adjusted_neighborhood_overlap"]
        seed_overlap_matrix[column, row] = seed_overlap_matrix[row, column]

save_records(VALIDATION_DIR / "04_seed_stability.csv", seed_records)
print(
    "Seed stability versus seed 0: "
    f"|tau-b| {min(row['abs_kendall_tau_b'] for row in seed_records):.3f}–"
    f"{max(row['abs_kendall_tau_b'] for row in seed_records):.3f}; "
    "adjusted local overlap "
    f"{min(row['adjusted_neighborhood_overlap'] for row in seed_records):.3f}–"
    f"{max(row['adjusted_neighborhood_overlap'] for row in seed_records):.3f}"
)

seed_figure, seed_axes = plt.subplots(
    2,
    2,
    figsize=(12, 10),
    constrained_layout=True,
)
seed_tau_image = seed_axes[0, 0].imshow(
    seed_tau_matrix,
    vmin=0,
    vmax=1,
    cmap="viridis",
    interpolation="nearest",
)
seed_axes[0, 0].set_xticks(np.arange(len(SEEDS)), SEEDS)
seed_axes[0, 0].set_yticks(np.arange(len(SEEDS)), SEEDS)
seed_axes[0, 0].set_xlabel("seed")
seed_axes[0, 0].set_ylabel("seed")
seed_axes[0, 0].set_title("Pairwise |Kendall tau-b| (ties retained)")
seed_figure.colorbar(seed_tau_image, ax=seed_axes[0, 0])

seed_overlap_image = seed_axes[0, 1].imshow(
    seed_overlap_matrix,
    vmin=0,
    vmax=1,
    cmap="magma",
    interpolation="nearest",
)
seed_axes[0, 1].set_xticks(np.arange(len(SEEDS)), SEEDS)
seed_axes[0, 1].set_yticks(np.arange(len(SEEDS)), SEEDS)
seed_axes[0, 1].set_xlabel("seed")
seed_axes[0, 1].set_ylabel("seed")
seed_axes[0, 1].set_title("Chance-adjusted rank-50 neighborhood overlap")
seed_figure.colorbar(seed_overlap_image, ax=seed_axes[0, 1])

for seed in DISPLAY_SEEDS:
    seed_axes[1, 0].scatter(
        primary_summary.embedding,
        seed_summaries[seed].embedding,
        s=2,
        alpha=0.2,
        label=f"seed {seed}",
        rasterized=True,
    )
seed_axes[1, 0].set_xlabel("seed 0 embedding")
seed_axes[1, 0].set_ylabel("comparison embedding")
seed_axes[1, 0].set_title("Final grid positions (reversal may be equivalent)")
seed_axes[1, 0].legend(frameon=False, markerscale=3, ncol=2)

metric_names = (
    "abs_kendall_tau_b",
    "adjusted_neighborhood_overlap",
    "cluster_ari",
    "distance_geometry",
)
metric_labels = ("|tau-b|", "local overlap", "cluster ARI", "distance geometry")
seed_metric_values = np.array(
    [[float(record[name]) for name in metric_names] for record in seed_records]
)
for seed_index, _record in enumerate(seed_records):
    seed_axes[1, 1].plot(
        np.arange(len(metric_names)),
        seed_metric_values[seed_index],
        "-",
        color="0.65",
        alpha=0.45,
        lw=0.8,
    )
seed_axes[1, 1].plot(
    np.arange(len(metric_names)),
    np.median(seed_metric_values, axis=0),
    "o-",
    color="black",
    lw=2,
    label="median of seeds 1–19",
)
seed_axes[1, 1].set_xticks(np.arange(len(metric_names)), metric_labels, rotation=20)
seed_axes[1, 1].set_ylim(0, 1)
seed_axes[1, 1].set_ylabel("agreement with seed 0")
seed_axes[1, 1].set_title("Different notions of reproducibility")
seed_axes[1, 1].legend(frameon=False)

seed_figure.suptitle(f"{recording_name} · active-neuron Rastermap seed sensitivity")
seed_path = FIG_DIR / "04_rastermap_02_seed_stability.png"
seed_figure.savefig(seed_path, dpi=150, bbox_inches="tight")
print("saved ->", seed_path)
if SHOW_FIGURES:
    plt.show()

gc.collect()


# %% [markdown]
# ## Step 5 — independent-time validation on a common active population
#
# Version 1.0's `itrain` argument is not leakage-free for sorting: PCA starts on
# `itrain`, but temporal components used for cluster similarity are reconstructed
# over all supplied columns. We therefore fit two **fresh models on two separate
# contiguous blocks**. The split is the state-bout boundary nearest the session
# midpoint, preventing one uninterrupted bout from leaking across both blocks.
# This symmetric stability analysis is explicitly
# conditional on neurons exceeding the activity criterion in both blocks.  The
# held-out activity values do not enter the opposite fit, although marginal
# activity in both blocks does enter population QC.
#
# Different brain-state composition between the blocks is part of this
# nonstationarity test.  A weak result does not mean the software failed; it
# means the full-session ordering should not automatically be called a stable
# module assignment.  The primary held-out statistic samples neuron pairs and
# asks whether embedding proximity learned in block A predicts zero-lag
# correlation in block B.  It does not depend on arbitrary 50-neuron display bins.

# %%
midpoint_frame = n_frames // 2
state_split_candidates = (
    np.flatnonzero(
        (state[1:] != state[:-1]) | ~np.isfinite(state[1:]) | ~np.isfinite(state[:-1])
    )
    + 1
)
if state_split_candidates.size:
    half_frame = int(
        state_split_candidates[
            np.argmin(np.abs(state_split_candidates - midpoint_frame))
        ]
    )
else:
    half_frame = midpoint_frame
print(
    f"Temporal split at frame {half_frame:,} "
    f"({(half_frame - midpoint_frame) / fs:+.1f} s from midpoint), "
    "on a state-bout boundary"
)
first_half_mask = rmt.active_deconvolution_rows(
    all_activity[:, :half_frame],
    fs,
    min_positive_bin_rate_hz=MIN_POSITIVE_BIN_RATE_HZ,
)
second_half_mask = rmt.active_deconvolution_rows(
    all_activity[:, half_frame:],
    fs,
    min_positive_bin_rate_hz=MIN_POSITIVE_BIN_RATE_HZ,
)
crossfit_mask = primary_active_mask & first_half_mask & second_half_mask
crossfit_roi_rows = np.flatnonzero(crossfit_mask)
first_activity = np.ascontiguousarray(
    all_activity[crossfit_roi_rows, :half_frame],
    dtype=np.float32,
)
second_activity = np.ascontiguousarray(
    all_activity[crossfit_roi_rows, half_frame:],
    dtype=np.float32,
)

first_model, first_summary = fit_activity(
    first_activity,
    n_clusters=N_CLUSTERS,
    n_pcs=N_PCS,
    locality=LOCALITY,
    lag_frames=primary_lag_frames,
    seed=REFERENCE_SEED,
    keep_normalized=True,
)
second_model, second_summary = fit_activity(
    second_activity,
    n_clusters=N_CLUSTERS,
    n_pcs=N_PCS,
    locality=LOCALITY,
    lag_frames=primary_lag_frames,
    seed=REFERENCE_SEED,
    keep_normalized=True,
)
temporal_metrics = compare_fits(first_summary, second_summary)
temporal_metrics["distance_geometry"] = embedding_distance_correlation(
    first_summary.embedding,
    second_summary.embedding,
)

first_U = first_model.Usv / first_model.sv
second_U = second_model.Usv / second_model.sv
subspace_overlap = float(np.linalg.norm(first_U.T @ second_U, ord="fro") ** 2 / N_PCS)
subspace_chance = N_PCS / crossfit_roi_rows.size
adjusted_subspace_overlap = (subspace_overlap - subspace_chance) / (1 - subspace_chance)

first_to_second_energy = float(
    np.sum((first_U.T @ second_model.X) ** 2, dtype=np.float64)
)
second_to_first_energy = float(
    np.sum((second_U.T @ first_model.X) ** 2, dtype=np.float64)
)
first_total = sum_of_squares(first_model.X)
second_total = sum_of_squares(second_model.X)
first_optimal = float(np.sum(first_model.sv.astype(np.float64) ** 2)) / first_total
second_optimal = float(np.sum(second_model.sv.astype(np.float64) ** 2)) / second_total
first_to_second_fraction = first_to_second_energy / second_total
second_to_first_fraction = second_to_first_energy / first_total
first_to_second_efficiency = first_to_second_fraction / second_optimal
second_to_first_efficiency = second_to_first_fraction / first_optimal

print(f"Cross-fit population: {crossfit_roi_rows.size:,} neurons active in both blocks")
print(
    f"Independent-block |tau-b|={temporal_metrics['abs_kendall_tau_b']:.3f}, "
    f"adjusted local overlap="
    f"{temporal_metrics['adjusted_neighborhood_overlap']:.3f}, "
    f"ARI={temporal_metrics['cluster_ari']:.3f}"
)
print(
    f"PCA subspace overlap={subspace_overlap:.3f} "
    f"(dimension-only chance {subspace_chance:.3f}); "
    f"cross-energy efficiency={first_to_second_efficiency:.3f}/"
    f"{second_to_first_efficiency:.3f}"
)

pair_first_rows, pair_second_rows = sample_distinct_pairs(
    crossfit_roi_rows.size,
    N_PAIR_SAMPLES,
    seed=REFERENCE_SEED,
)
first_pair_correlations = sampled_pair_correlations(
    first_model.X,
    pair_first_rows,
    pair_second_rows,
)
second_pair_correlations = sampled_pair_correlations(
    second_model.X,
    pair_first_rows,
    pair_second_rows,
)

first_own_pair_score = pair_distance_similarity_score(
    first_summary.embedding,
    pair_first_rows,
    pair_second_rows,
    first_pair_correlations,
)
second_own_pair_score = pair_distance_similarity_score(
    second_summary.embedding,
    pair_first_rows,
    pair_second_rows,
    second_pair_correlations,
)
first_to_second_pair_score = pair_distance_similarity_score(
    first_summary.embedding,
    pair_first_rows,
    pair_second_rows,
    second_pair_correlations,
)
second_to_first_pair_score = pair_distance_similarity_score(
    second_summary.embedding,
    pair_first_rows,
    pair_second_rows,
    first_pair_correlations,
)
first_activity_score = np.count_nonzero(first_activity > 0, axis=1)
second_activity_score = np.count_nonzero(second_activity > 0, axis=1)
activity_to_second_pair_score = pair_distance_similarity_score(
    first_activity_score,
    pair_first_rows,
    pair_second_rows,
    second_pair_correlations,
)
activity_to_first_pair_score = pair_distance_similarity_score(
    second_activity_score,
    pair_first_rows,
    pair_second_rows,
    first_pair_correlations,
)

# The zero-lag pair score above is orientation-free and secondary. Rastermap's
# actual lagged objective is directional: for an ascending cluster order, upper
# positions are expected to lead lower positions. Transfer each source fit's
# sorted cluster templates to the untouched opposite block and score that exact
# upper-triangle matching target.
first_nodes_on_second = transferred_node_similarity(first_model, second_model.X)
second_nodes_on_first = transferred_node_similarity(second_model, first_model.X)
first_cluster_identity = np.arange(first_model.U_nodes.shape[0], dtype=np.int64)
second_cluster_identity = np.arange(second_model.U_nodes.shape[0], dtype=np.int64)
lagged_objective_first_to_second = rastermap_directional_objective(
    first_nodes_on_second,
    first_model.BBt,
    first_cluster_identity,
)
lagged_objective_first_reverse_on_second = rastermap_directional_objective(
    first_nodes_on_second,
    first_model.BBt,
    first_cluster_identity[::-1],
)
lagged_objective_second_to_first = rastermap_directional_objective(
    second_nodes_on_first,
    second_model.BBt,
    second_cluster_identity,
)
lagged_objective_second_reverse_on_first = rastermap_directional_objective(
    second_nodes_on_first,
    second_model.BBt,
    second_cluster_identity[::-1],
)
first_activity_cluster_order = activity_cluster_order(
    first_model,
    first_activity_score,
)
second_activity_cluster_order = activity_cluster_order(
    second_model,
    second_activity_score,
)
lagged_objective_activity_first_to_second = rastermap_directional_objective(
    first_nodes_on_second,
    first_model.BBt,
    first_activity_cluster_order,
)
lagged_objective_activity_second_to_first = rastermap_directional_objective(
    second_nodes_on_first,
    second_model.BBt,
    second_activity_cluster_order,
)
objective_rng = np.random.default_rng(70_000)
lagged_objective_random_first_to_second = np.array(
    [
        rastermap_directional_objective(
            first_nodes_on_second,
            first_model.BBt,
            objective_rng.permutation(first_cluster_identity),
        )
        for _ in range(N_OBJECTIVE_PERMUTATIONS)
    ]
)
lagged_objective_random_second_to_first = np.array(
    [
        rastermap_directional_objective(
            second_nodes_on_first,
            second_model.BBt,
            objective_rng.permutation(second_cluster_identity),
        )
        for _ in range(N_OBJECTIVE_PERMUTATIONS)
    ]
)
lagged_objective_exceedance_first_to_second = (
    1
    + np.count_nonzero(
        lagged_objective_random_first_to_second >= lagged_objective_first_to_second
    )
) / (N_OBJECTIVE_PERMUTATIONS + 1)
lagged_objective_exceedance_second_to_first = (
    1
    + np.count_nonzero(
        lagged_objective_random_second_to_first >= lagged_objective_second_to_first
    )
) / (N_OBJECTIVE_PERMUTATIONS + 1)
print(
    "Lag-aware cluster objective A→B: learned="
    f"{lagged_objective_first_to_second:.4f}, reversed="
    f"{lagged_objective_first_reverse_on_second:.4f}, activity-rank="
    f"{lagged_objective_activity_first_to_second:.4f}, permutation "
    f"exceedance={lagged_objective_exceedance_first_to_second:.3f}"
)
print(
    "Lag-aware cluster objective B→A: learned="
    f"{lagged_objective_second_to_first:.4f}, reversed="
    f"{lagged_objective_second_reverse_on_first:.4f}, activity-rank="
    f"{lagged_objective_activity_second_to_first:.4f}, permutation "
    f"exceedance={lagged_objective_exceedance_second_to_first:.3f}"
)

random_rng = np.random.default_rng(REFERENCE_SEED)
random_second_pair_scores = np.empty(N_RANDOM_ORDERS)
first_random_order = None
for repetition in range(N_RANDOM_ORDERS):
    random_order = random_rng.permutation(crossfit_roi_rows.size)
    if first_random_order is None:
        first_random_order = random_order.copy()
    random_second_pair_scores[repetition] = pair_distance_similarity_score(
        random_order,
        pair_first_rows,
        pair_second_rows,
        second_pair_correlations,
    )

state_changes = (
    np.flatnonzero(
        (state[1:] != state[:-1]) | ~np.isfinite(state[1:]) | ~np.isfinite(state[:-1])
    )
    + 1
)
complete_segments = ts.acquisition_segments(
    n_frames,
    boundary_ind,
    extra_splits=(*state_changes, half_frame),
)
first_segments = [
    (start, stop)
    for start, stop in complete_segments
    if start < half_frame and stop <= half_frame
]

shift_null_pair_scores = np.empty(N_SHIFT_NULLS)
shift_null_lagged_objectives = np.empty(N_SHIFT_NULLS)
shift_null_embeddings = np.empty(
    (N_SHIFT_NULLS, crossfit_roi_rows.size),
    dtype=np.float32,
)
for repetition in range(N_SHIFT_NULLS):
    shifted_first = independently_shift_rows_within_segments(
        first_activity,
        first_segments,
        min_shift=primary_lag_frames + 1,
        seed=10_000 + repetition,
    )
    shifted_model, shifted_summary = fit_activity(
        shifted_first,
        n_clusters=N_CLUSTERS,
        n_pcs=N_PCS,
        locality=LOCALITY,
        lag_frames=primary_lag_frames,
        seed=REFERENCE_SEED,
        keep_normalized=False,
    )
    shift_null_pair_scores[repetition] = pair_distance_similarity_score(
        shifted_summary.embedding,
        pair_first_rows,
        pair_second_rows,
        second_pair_correlations,
    )
    shift_null_embeddings[repetition] = shifted_summary.embedding
    shifted_nodes_on_second = transferred_node_similarity(
        shifted_model,
        second_model.X,
    )
    shift_null_lagged_objectives[repetition] = rastermap_directional_objective(
        shifted_nodes_on_second,
        shifted_model.BBt,
        np.arange(shifted_model.U_nodes.shape[0], dtype=np.int64),
    )
    del shifted_first, shifted_model, shifted_summary
    gc.collect()

shift_null_exceedance = (
    1 + np.count_nonzero(shift_null_pair_scores >= first_to_second_pair_score)
) / (N_SHIFT_NULLS + 1)
lagged_objective_shift_exceedance = (
    1
    + np.count_nonzero(shift_null_lagged_objectives >= lagged_objective_first_to_second)
) / (N_SHIFT_NULLS + 1)
print(
    "Held-out pair-distance score A→B="
    f"{first_to_second_pair_score:.3f}; random mean="
    f"{random_second_pair_scores.mean():.3f}; state/bout-shift mean="
    f"{shift_null_pair_scores.mean():.3f}"
)
print(
    "Descriptive shift-null exceedance="
    f"{shift_null_exceedance:.3f} ({N_SHIFT_NULLS} null fits; minimum "
    f"{1 / (N_SHIFT_NULLS + 1):.2f})"
)
print(
    "Lag-aware A→B state/bout-shift objective mean="
    f"{shift_null_lagged_objectives.mean():.4f}; descriptive exceedance="
    f"{lagged_objective_shift_exceedance:.3f}"
)

state_first = state[:half_frame]
state_second = state[half_frame:]
state_codes = np.unique(state[np.isfinite(state)])
state_fraction_first = np.array([np.mean(state_first == code) for code in state_codes])
state_fraction_second = np.array(
    [np.mean(state_second == code) for code in state_codes]
)
state_labels = [state_code_labels.get(float(code), str(code)) for code in state_codes]

state_stratified_records: list[dict[str, object]] = []
state_random_rng = np.random.default_rng(80_000)
state_random_embeddings = np.stack(
    [
        state_random_rng.permutation(crossfit_roi_rows.size)
        for _ in range(N_RANDOM_ORDERS)
    ]
)
for code, label in zip(state_codes, state_labels, strict=True):
    first_state_frames = np.flatnonzero(state_first == code)
    second_state_frames = np.flatnonzero(state_second == code)
    state_record: dict[str, object] = {
        "state_code": code,
        "state_label": label,
        "first_block_frames": first_state_frames.size,
        "second_block_frames": second_state_frames.size,
        "evaluable": False,
        "not_evaluable_reason": "",
        "first_order_on_first_state": np.nan,
        "second_order_on_second_state": np.nan,
        "first_order_on_second_state": np.nan,
        "second_order_on_first_state": np.nan,
        "activity_rank_first_to_second_state": np.nan,
        "activity_rank_second_to_first_state": np.nan,
        "rastermap_minus_activity_rank_first_to_second": np.nan,
        "random_order_on_second_state_mean": np.nan,
        "random_order_on_second_state_min": np.nan,
        "random_order_on_second_state_max": np.nan,
        "shift_fit_on_second_state_mean": np.nan,
        "shift_fit_on_second_state_min": np.nan,
        "shift_fit_on_second_state_max": np.nan,
        "shift_null_exceedance_descriptive": np.nan,
    }
    if (
        first_state_frames.size < MIN_STATE_HELDOUT_FRAMES
        or second_state_frames.size < MIN_STATE_HELDOUT_FRAMES
    ):
        state_record["not_evaluable_reason"] = (
            f"requires at least {MIN_STATE_HELDOUT_FRAMES} frames in both "
            f"blocks; observed {first_state_frames.size} and "
            f"{second_state_frames.size}"
        )
        state_stratified_records.append(state_record)
        continue
    first_state_correlations = sampled_pair_correlations(
        first_model.X[:, first_state_frames],
        pair_first_rows,
        pair_second_rows,
    )
    second_state_correlations = sampled_pair_correlations(
        second_model.X[:, second_state_frames],
        pair_first_rows,
        pair_second_rows,
    )
    state_random_scores = np.array(
        [
            pair_distance_similarity_score(
                random_embedding,
                pair_first_rows,
                pair_second_rows,
                second_state_correlations,
            )
            for random_embedding in state_random_embeddings
        ]
    )
    state_shift_scores = np.array(
        [
            pair_distance_similarity_score(
                shifted_embedding,
                pair_first_rows,
                pair_second_rows,
                second_state_correlations,
            )
            for shifted_embedding in shift_null_embeddings
        ]
    )
    first_order_on_second_state = pair_distance_similarity_score(
        first_summary.embedding,
        pair_first_rows,
        pair_second_rows,
        second_state_correlations,
    )
    activity_rank_first_to_second_state = pair_distance_similarity_score(
        first_activity_score,
        pair_first_rows,
        pair_second_rows,
        second_state_correlations,
    )
    state_record.update(
        {
            "evaluable": True,
            "first_order_on_first_state": pair_distance_similarity_score(
                first_summary.embedding,
                pair_first_rows,
                pair_second_rows,
                first_state_correlations,
            ),
            "second_order_on_second_state": pair_distance_similarity_score(
                second_summary.embedding,
                pair_first_rows,
                pair_second_rows,
                second_state_correlations,
            ),
            "first_order_on_second_state": first_order_on_second_state,
            "second_order_on_first_state": pair_distance_similarity_score(
                second_summary.embedding,
                pair_first_rows,
                pair_second_rows,
                first_state_correlations,
            ),
            "activity_rank_first_to_second_state": (
                activity_rank_first_to_second_state
            ),
            "activity_rank_second_to_first_state": pair_distance_similarity_score(
                second_activity_score,
                pair_first_rows,
                pair_second_rows,
                first_state_correlations,
            ),
            "rastermap_minus_activity_rank_first_to_second": (
                first_order_on_second_state - activity_rank_first_to_second_state
            ),
            "random_order_on_second_state_mean": state_random_scores.mean(),
            "random_order_on_second_state_min": state_random_scores.min(),
            "random_order_on_second_state_max": state_random_scores.max(),
            "shift_fit_on_second_state_mean": state_shift_scores.mean(),
            "shift_fit_on_second_state_min": state_shift_scores.min(),
            "shift_fit_on_second_state_max": state_shift_scores.max(),
            "shift_null_exceedance_descriptive": (
                1 + np.count_nonzero(state_shift_scores >= first_order_on_second_state)
            )
            / (N_SHIFT_NULLS + 1),
        }
    )
    state_stratified_records.append(state_record)

evaluable_state_records = [
    record for record in state_stratified_records if bool(record["evaluable"])
]
save_records(
    VALIDATION_DIR / "04_state_stratified_generalization.csv",
    state_stratified_records,
)
for record in state_stratified_records:
    if bool(record["evaluable"]):
        print(
            f"  {record['state_label']}: Rastermap A→B="
            f"{float(record['first_order_on_second_state']):.4f}, B→A="
            f"{float(record['second_order_on_first_state']):.4f}, "
            f"activity-rank A→B="
            f"{float(record['activity_rank_first_to_second_state']):.4f}"
        )
    else:
        print(
            f"  {record['state_label']}: not evaluable — "
            f"{record['not_evaluable_reason']}"
        )

temporal_record: dict[str, object] = {
    "recording": recording_name,
    "population_definition": "active_above_threshold_in_both_blocks",
    "n_crossfit": crossfit_roi_rows.size,
    **temporal_metrics,
    "pca_subspace_overlap": subspace_overlap,
    "pca_subspace_chance": subspace_chance,
    "pca_subspace_overlap_adjusted": adjusted_subspace_overlap,
    "pca_transfer_first_to_second": first_to_second_efficiency,
    "pca_transfer_second_to_first": second_to_first_efficiency,
    "pair_score_own_first": first_own_pair_score,
    "pair_score_own_second": second_own_pair_score,
    "pair_score_first_to_second": first_to_second_pair_score,
    "pair_score_second_to_first": second_to_first_pair_score,
    "pair_score_activity_rank_first_to_second": activity_to_second_pair_score,
    "pair_score_activity_rank_second_to_first": activity_to_first_pair_score,
    "pair_score_random_mean": float(random_second_pair_scores.mean()),
    "pair_score_state_bout_shift_mean": float(shift_null_pair_scores.mean()),
    "shift_null_exceedance_descriptive": shift_null_exceedance,
    "lagged_cluster_objective_first_to_second": (lagged_objective_first_to_second),
    "lagged_cluster_objective_first_reverse_on_second": (
        lagged_objective_first_reverse_on_second
    ),
    "lagged_cluster_objective_activity_first_to_second": (
        lagged_objective_activity_first_to_second
    ),
    "lagged_cluster_objective_permutation_exceedance_first_to_second": (
        lagged_objective_exceedance_first_to_second
    ),
    "lagged_cluster_objective_second_to_first": (lagged_objective_second_to_first),
    "lagged_cluster_objective_second_reverse_on_first": (
        lagged_objective_second_reverse_on_first
    ),
    "lagged_cluster_objective_activity_second_to_first": (
        lagged_objective_activity_second_to_first
    ),
    "lagged_cluster_objective_permutation_exceedance_second_to_first": (
        lagged_objective_exceedance_second_to_first
    ),
    "lagged_cluster_objective_shift_mean_first_to_second": float(
        shift_null_lagged_objectives.mean()
    ),
    "lagged_cluster_objective_shift_exceedance_descriptive": (
        lagged_objective_shift_exceedance
    ),
}
for code, label, fraction_first, fraction_second in zip(
    state_codes,
    state_labels,
    state_fraction_first,
    state_fraction_second,
    strict=True,
):
    safe_label = label.lower().replace(" ", "_")
    temporal_record[f"state_{safe_label}_{code:g}_fraction_first"] = fraction_first
    temporal_record[f"state_{safe_label}_{code:g}_fraction_second"] = fraction_second
save_records(
    VALIDATION_DIR / "04_temporal_generalization.csv",
    [temporal_record],
)

temporal_figure, temporal_axes = plt.subplots(
    2,
    3,
    figsize=(17, 10),
    constrained_layout=True,
)
temporal_axes[0, 0].scatter(
    first_summary.embedding,
    second_summary.embedding,
    s=3,
    alpha=0.3,
    color="tab:blue",
    linewidths=0,
    rasterized=True,
)
temporal_axes[0, 0].set_xlabel("block-A embedding")
temporal_axes[0, 0].set_ylabel("block-B embedding")
temporal_axes[0, 0].set_title(
    "Independent embeddings · " f"|tau-b|={temporal_metrics['abs_kendall_tau_b']:.3f}"
)

comparison_labels = ("|tau-b|", "local overlap", "distance", "cluster ARI")
comparison_values = (
    temporal_metrics["abs_kendall_tau_b"],
    temporal_metrics["adjusted_neighborhood_overlap"],
    temporal_metrics["distance_geometry"],
    temporal_metrics["cluster_ari"],
)
temporal_axes[0, 1].bar(
    comparison_labels,
    comparison_values,
    color=("tab:blue", "tab:orange", "tab:green", "tab:purple"),
)
temporal_axes[0, 1].axhline(0, color="black", lw=0.8)
temporal_axes[0, 1].set_ylim(min(-0.05, min(comparison_values) - 0.03), 1)
temporal_axes[0, 1].tick_params(axis="x", rotation=20)
temporal_axes[0, 1].set_ylabel("independent-block agreement")
temporal_axes[0, 1].set_title("Tie-aware order and coarse clusters")

pca_labels = ("subspace", "A→B energy", "B→A energy")
pca_values = (
    adjusted_subspace_overlap,
    first_to_second_efficiency,
    second_to_first_efficiency,
)
temporal_axes[0, 2].bar(pca_labels, pca_values, color="tab:cyan")
temporal_axes[0, 2].axhline(0, color="black", lw=0.8)
temporal_axes[0, 2].tick_params(axis="x", rotation=15)
temporal_axes[0, 2].set_ylabel("adjusted overlap / transfer efficiency")
temporal_axes[0, 2].set_title("Held-out PCA structure")

pair_labels = ("A own", "B own", "A→B", "B→A", "activity A→B", "activity B→A")
pair_values = (
    first_own_pair_score,
    second_own_pair_score,
    first_to_second_pair_score,
    second_to_first_pair_score,
    activity_to_second_pair_score,
    activity_to_first_pair_score,
)
temporal_axes[1, 0].bar(
    pair_labels,
    pair_values,
    color=(
        "tab:green",
        "tab:green",
        "tab:blue",
        "tab:blue",
        "tab:orange",
        "tab:orange",
    ),
)
temporal_axes[1, 0].axhline(0, color="black", lw=0.8)
temporal_axes[1, 0].tick_params(axis="x", rotation=20)
temporal_axes[1, 0].set_ylabel("Spearman(correlation, −embedding distance)")
temporal_axes[1, 0].set_title("Sampled-pair structure (primary held-out score)")

temporal_axes[1, 1].boxplot(
    (random_second_pair_scores, shift_null_pair_scores),
    positions=(0, 1),
    widths=0.55,
    patch_artist=True,
    boxprops={"facecolor": "0.8"},
    medianprops={"color": "black"},
)
temporal_axes[1, 1].axhline(
    first_to_second_pair_score,
    color="tab:blue",
    lw=2,
    label="observed A→B",
)
temporal_axes[1, 1].set_xticks((0, 1), ("random order", "state/bout shifts"))
temporal_axes[1, 1].set_ylabel("held-out pair-distance score")
temporal_axes[1, 1].set_title("Null calibrations (descriptive)")
temporal_axes[1, 1].legend(frameon=False)

state_x = np.arange(len(state_codes))
state_width = 0.38
temporal_axes[1, 2].bar(
    state_x - state_width / 2,
    state_fraction_first,
    state_width,
    label="first block",
)
temporal_axes[1, 2].bar(
    state_x + state_width / 2,
    state_fraction_second,
    state_width,
    label="second block",
)
temporal_axes[1, 2].set_xticks(state_x, state_labels, rotation=20)
temporal_axes[1, 2].set_ylabel("fraction of frames")
temporal_axes[1, 2].set_title("State composition differs between blocks")
temporal_axes[1, 2].legend(frameon=False)

temporal_figure.suptitle(
    f"{recording_name} · separate bout-boundary blocks with different state composition"
)
temporal_path = FIG_DIR / "04_rastermap_03_temporal_generalization.png"
temporal_figure.savefig(temporal_path, dpi=150, bbox_inches="tight")
print("saved ->", temporal_path)
if SHOW_FIGURES:
    plt.show()

lagged_figure, lagged_axes = plt.subplots(
    1,
    2,
    figsize=(14, 5),
    constrained_layout=True,
)
for axis, null_values, learned, reversed_score, activity_score, direction in (
    (
        lagged_axes[0],
        lagged_objective_random_first_to_second,
        lagged_objective_first_to_second,
        lagged_objective_first_reverse_on_second,
        lagged_objective_activity_first_to_second,
        "A→B",
    ),
    (
        lagged_axes[1],
        lagged_objective_random_second_to_first,
        lagged_objective_second_to_first,
        lagged_objective_second_reverse_on_first,
        lagged_objective_activity_second_to_first,
        "B→A",
    ),
):
    axis.hist(
        null_values,
        bins=35,
        color="0.75",
        label=f"{N_OBJECTIVE_PERMUTATIONS} cluster-order permutations",
    )
    axis.axvline(learned, color="tab:blue", lw=2, label="learned order")
    axis.axvline(
        reversed_score,
        color="tab:cyan",
        lw=2,
        label="reversed learned order",
    )
    axis.axvline(
        activity_score,
        color="tab:green",
        lw=2,
        label="training activity-cluster order",
    )
    if direction == "A→B":
        axis.axvline(
            shift_null_lagged_objectives.mean(),
            color="tab:orange",
            lw=2,
            label=f"A-side shift-fit mean (n={N_SHIFT_NULLS})",
        )
    axis.set_xlabel("held-out directed Rastermap cluster objective")
    axis.set_ylabel("permutations")
    axis.set_title(
        f"{direction}: exact lag-{primary_lag_frames} cluster-template transfer"
    )
    axis.legend(frameon=False, fontsize=8)
lagged_figure.suptitle(
    "Lag-aware held-out objective; coarse 100-cluster order, not final-neuron resolution"
)
lagged_path = FIG_DIR / "04_rastermap_03d_lagged_objective.png"
lagged_figure.savefig(lagged_path, dpi=150, bbox_inches="tight")
print("saved ->", lagged_path)
if SHOW_FIGURES:
    plt.show()

if evaluable_state_records:
    stratified_labels = [
        str(record["state_label"]) for record in evaluable_state_records
    ]
    stratified_x = np.arange(len(evaluable_state_records))
    stratified_figure, stratified_axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        constrained_layout=True,
    )
    stratified_width = 0.2
    for offset, field, label, color in (
        (-1.5, "first_order_on_first_state", "A own", "tab:green"),
        (-0.5, "second_order_on_second_state", "B own", "tab:olive"),
        (0.5, "first_order_on_second_state", "A→B", "tab:blue"),
        (1.5, "second_order_on_first_state", "B→A", "tab:cyan"),
    ):
        stratified_axes[0].bar(
            stratified_x + offset * stratified_width,
            [record[field] for record in evaluable_state_records],
            stratified_width,
            label=label,
            color=color,
        )
    stratified_axes[0].set_xticks(stratified_x, stratified_labels, rotation=20)
    stratified_axes[0].axhline(0, color="black", lw=0.8)
    stratified_axes[0].set_ylabel("pair correlation vs −embedding distance")
    stratified_axes[0].set_title("Fixed full-block orders, evaluated within state")
    stratified_axes[0].legend(frameon=False, ncol=2)

    heldout_state_scores = np.array(
        [
            float(record["first_order_on_second_state"])
            for record in evaluable_state_records
        ]
    )
    activity_rank_state_scores = np.array(
        [
            float(record["activity_rank_first_to_second_state"])
            for record in evaluable_state_records
        ]
    )
    random_state_mean = np.array(
        [
            float(record["random_order_on_second_state_mean"])
            for record in evaluable_state_records
        ]
    )
    shift_state_mean = np.array(
        [
            float(record["shift_fit_on_second_state_mean"])
            for record in evaluable_state_records
        ]
    )
    stratified_axes[1].plot(
        stratified_x,
        heldout_state_scores,
        "o-",
        label="observed A→B",
        color="tab:blue",
    )
    stratified_axes[1].plot(
        stratified_x,
        activity_rank_state_scores,
        "o-",
        label="activity-count rank A→B",
        color="tab:green",
    )
    stratified_axes[1].plot(
        stratified_x,
        random_state_mean,
        "o-",
        label="random-order mean",
        color="0.5",
    )
    stratified_axes[1].plot(
        stratified_x,
        shift_state_mean,
        "o-",
        label=f"A-side state/bout-shift mean (min–max; n={N_SHIFT_NULLS})",
        color="tab:orange",
    )
    for index, record in enumerate(evaluable_state_records):
        stratified_axes[1].vlines(
            index,
            record["shift_fit_on_second_state_min"],
            record["shift_fit_on_second_state_max"],
            color="tab:orange",
            alpha=0.45,
        )
    stratified_axes[1].set_xticks(stratified_x, stratified_labels, rotation=20)
    stratified_axes[1].axhline(0, color="black", lw=0.8)
    stratified_axes[1].set_ylabel("held-out pair-distance score")
    stratified_axes[1].set_title(
        "Evaluation is conditioned on state; training composition remains confounded"
    )
    stratified_axes[1].legend(frameon=False)
    omitted_state_text = "; ".join(
        f"{record['state_label']} not evaluable "
        f"({record['first_block_frames']}/{record['second_block_frames']} frames)"
        for record in state_stratified_records
        if not bool(record["evaluable"])
    )
    omission_suffix = f"\n{omitted_state_text}" if omitted_state_text else ""
    stratified_figure.suptitle(
        "Descriptive state-conditioned evaluation; not a state-specific refit"
        f"{omission_suffix}"
    )
    stratified_path = FIG_DIR / "04_rastermap_03c_state_stratified.png"
    stratified_figure.savefig(stratified_path, dpi=150, bbox_inches="tight")
    print("saved ->", stratified_path)
    if SHOW_FIGURES:
        plt.show()

assert first_random_order is not None
second_own_superneurons = superneuron_display(
    second_model.X,
    second_summary.order,
)
cross_superneurons = superneuron_display(
    second_model.X,
    first_summary.order,
)
random_superneurons = superneuron_display(
    second_model.X,
    first_random_order,
)
display_frames = min(1_500, second_activity.shape[1])
heldout_raster_figure, heldout_raster_axes = plt.subplots(
    1,
    3,
    figsize=(15, 5),
    constrained_layout=True,
    sharey=True,
)
for axis, values, title in (
    (heldout_raster_axes[0], second_own_superneurons, "B order on B (in-sample)"),
    (heldout_raster_axes[1], cross_superneurons, "A order on B (held-out)"),
    (heldout_raster_axes[2], random_superneurons, "Random order on B"),
):
    axis.imshow(
        values[:, :display_frames],
        aspect="auto",
        cmap="gray_r",
        vmin=0,
        vmax=1.5,
        interpolation="nearest",
        rasterized=True,
    )
    axis.set_title(title)
    axis.set_xlabel("block-B frame")
heldout_raster_axes[0].set_ylabel("50-neuron superneurons")
heldout_raster_figure.suptitle(
    "Display only: fixed bins depend on unresolved within-grid ties"
)
heldout_raster_path = FIG_DIR / "04_rastermap_03b_heldout_rasters.png"
heldout_raster_figure.savefig(
    heldout_raster_path,
    dpi=150,
    bbox_inches="tight",
)
print("saved ->", heldout_raster_path)
if SHOW_FIGURES:
    plt.show()

del first_activity, second_activity, first_model, second_model
gc.collect()


# %% [markdown]
# ## Step 6 — activity-cutoff and proxy-definition sensitivity
#
# The paper reports a 0.1--0.25 Hz minimum but does not disclose the exact
# Figure 3 cutoff.  OASIS amplitudes here are not calibrated spike counts, so
# this cell does not search for the cutoff that makes the prettiest result.
# Instead it asks two fixed working questions: how many neurons survive, and how
# much seed/reference agreement changes under nearby positive-bin cutoffs.  A
# separate 0→positive onset definition exposes the ambiguity in the activity
# proxy.  Every cross-cutoff comparison is first mapped back to original ROI
# IDs and evaluated only among neurons common to both selected populations;
# subset-local row numbers must never be compared directly.

# %%
primary_global_summary = remap_summary(
    primary_summary,
    primary_roi_rows,
    n_recorded_neurons,
)
threshold_records: list[dict[str, object]] = []
for threshold, n_selected in zip(
    THRESHOLD_COUNT_GRID,
    threshold_counts,
    strict=True,
):
    record: dict[str, object] = {
        "selection_definition": "positive_bins_per_second",
        "threshold_positive_bins_per_second": threshold,
        "n_selected": int(n_selected),
        "fitted": False,
        "seed_abs_kendall_tau_b": np.nan,
        "seed_adjusted_neighborhood_overlap": np.nan,
        "seed_distance_geometry": np.nan,
        "primary_abs_kendall_tau_b": np.nan,
        "primary_adjusted_neighborhood_overlap": np.nan,
        "primary_distance_geometry": np.nan,
        "runtime_seed0_seconds": np.nan,
        "runtime_seed1_seconds": np.nan,
    }
    if RUN_THRESHOLD_FITS and threshold in THRESHOLD_FIT_GRID:
        threshold_mask = valid_recorded_rows & (positive_bin_rate_hz >= threshold)
        threshold_roi_rows = np.flatnonzero(threshold_mask)
        threshold_activity = np.ascontiguousarray(
            all_activity[threshold_roi_rows],
            dtype=np.float32,
        )
        if np.isclose(threshold, MIN_POSITIVE_BIN_RATE_HZ):
            threshold_summary0 = primary_summary
            threshold_summary1 = seed_summaries[1]
            runtime0 = primary_summary.runtime_seconds
            runtime1 = seed_summaries[1].runtime_seconds
        else:
            threshold_model0, threshold_summary0 = fit_activity(
                threshold_activity,
                n_clusters=N_CLUSTERS,
                n_pcs=N_PCS,
                locality=LOCALITY,
                lag_frames=primary_lag_frames,
                seed=REFERENCE_SEED,
                keep_normalized=False,
            )
            threshold_model1, threshold_summary1 = fit_activity(
                threshold_activity,
                n_clusters=N_CLUSTERS,
                n_pcs=N_PCS,
                locality=LOCALITY,
                lag_frames=primary_lag_frames,
                seed=1,
                keep_normalized=False,
            )
            runtime0 = threshold_summary0.runtime_seconds
            runtime1 = threshold_summary1.runtime_seconds
            del threshold_model0, threshold_model1

        seed_metrics = compare_fits(threshold_summary0, threshold_summary1)
        seed_distance = embedding_distance_correlation(
            threshold_summary0.embedding,
            threshold_summary1.embedding,
            seed=1,
        )
        threshold_global_summary = remap_summary(
            threshold_summary0,
            threshold_roi_rows,
            n_recorded_neurons,
        )
        primary_metrics = compare_fits(
            primary_global_summary,
            threshold_global_summary,
        )
        primary_distance = embedding_distance_correlation(
            primary_global_summary.embedding,
            threshold_global_summary.embedding,
        )
        record.update(
            {
                "fitted": True,
                "seed_abs_kendall_tau_b": seed_metrics["abs_kendall_tau_b"],
                "seed_adjusted_neighborhood_overlap": seed_metrics[
                    "adjusted_neighborhood_overlap"
                ],
                "seed_distance_geometry": seed_distance,
                "primary_abs_kendall_tau_b": primary_metrics["abs_kendall_tau_b"],
                "primary_adjusted_neighborhood_overlap": primary_metrics[
                    "adjusted_neighborhood_overlap"
                ],
                "primary_distance_geometry": primary_distance,
                "runtime_seed0_seconds": runtime0,
                "runtime_seed1_seconds": runtime1,
            }
        )
        del threshold_activity, threshold_summary0, threshold_summary1
        gc.collect()
    threshold_records.append(record)

onset_mask = valid_recorded_rows & (positive_onset_rate_hz >= MIN_POSITIVE_BIN_RATE_HZ)
onset_roi_rows = np.flatnonzero(onset_mask)
onset_record: dict[str, object] = {
    "selection_definition": "positive_onsets_per_second",
    "threshold_positive_bins_per_second": MIN_POSITIVE_BIN_RATE_HZ,
    "n_selected": int(onset_roi_rows.size),
    "fitted": False,
    "seed_abs_kendall_tau_b": np.nan,
    "seed_adjusted_neighborhood_overlap": np.nan,
    "seed_distance_geometry": np.nan,
    "primary_abs_kendall_tau_b": np.nan,
    "primary_adjusted_neighborhood_overlap": np.nan,
    "primary_distance_geometry": np.nan,
    "runtime_seed0_seconds": np.nan,
    "runtime_seed1_seconds": np.nan,
}
if RUN_THRESHOLD_FITS:
    onset_activity = np.ascontiguousarray(
        all_activity[onset_roi_rows],
        dtype=np.float32,
    )
    onset_model, onset_summary = fit_activity(
        onset_activity,
        n_clusters=N_CLUSTERS,
        n_pcs=N_PCS,
        locality=LOCALITY,
        lag_frames=primary_lag_frames,
        seed=REFERENCE_SEED,
        keep_normalized=False,
    )
    onset_global_summary = remap_summary(
        onset_summary,
        onset_roi_rows,
        n_recorded_neurons,
    )
    onset_primary_metrics = compare_fits(
        primary_global_summary,
        onset_global_summary,
    )
    onset_record.update(
        {
            "fitted": True,
            "primary_abs_kendall_tau_b": onset_primary_metrics["abs_kendall_tau_b"],
            "primary_adjusted_neighborhood_overlap": onset_primary_metrics[
                "adjusted_neighborhood_overlap"
            ],
            "primary_distance_geometry": embedding_distance_correlation(
                primary_global_summary.embedding,
                onset_global_summary.embedding,
            ),
            "runtime_seed0_seconds": onset_summary.runtime_seconds,
        }
    )
    del onset_activity, onset_model, onset_summary, onset_global_summary
    gc.collect()
threshold_records.append(onset_record)
save_records(
    VALIDATION_DIR / "04_activity_threshold_sensitivity.csv",
    threshold_records,
)

fitted_bin_records = [
    record
    for record in threshold_records
    if record["selection_definition"] == "positive_bins_per_second" and record["fitted"]
]
threshold_figure, threshold_axes = plt.subplots(
    2,
    2,
    figsize=(13, 9),
    constrained_layout=True,
)
threshold_axes[0, 0].plot(
    THRESHOLD_COUNT_GRID,
    threshold_counts,
    "o-",
    label="positive bins",
)
threshold_axes[0, 0].plot(
    THRESHOLD_COUNT_GRID,
    onset_threshold_counts,
    "o-",
    label="positive onsets",
)
threshold_axes[0, 0].axvline(MIN_POSITIVE_BIN_RATE_HZ, color="tab:red", lw=1)
threshold_axes[0, 0].set_xlabel("minimum activity proxy (s⁻¹)")
threshold_axes[0, 0].set_ylabel("retained neurons")
threshold_axes[0, 0].set_title("Population size is cutoff- and proxy-dependent")
threshold_axes[0, 0].legend(frameon=False)

if fitted_bin_records:
    fitted_thresholds = np.array(
        [
            float(record["threshold_positive_bins_per_second"])
            for record in fitted_bin_records
        ]
    )
    threshold_axes[0, 1].plot(
        fitted_thresholds,
        [record["seed_abs_kendall_tau_b"] for record in fitted_bin_records],
        "o-",
        label="|tau-b|",
    )
    threshold_axes[0, 1].plot(
        fitted_thresholds,
        [record["seed_adjusted_neighborhood_overlap"] for record in fitted_bin_records],
        "o-",
        label="tie-randomized local overlap",
    )
    threshold_axes[0, 1].plot(
        fitted_thresholds,
        [record["seed_distance_geometry"] for record in fitted_bin_records],
        "o-",
        label="distance geometry",
    )
    threshold_axes[0, 1].set_xlabel("positive-bin cutoff (s⁻¹)")
    threshold_axes[0, 1].set_ylabel("seed 0 versus seed 1")
    threshold_axes[0, 1].set_title("Filtering does not guarantee seed stability")
    threshold_axes[0, 1].legend(frameon=False)

    threshold_axes[1, 0].plot(
        fitted_thresholds,
        [record["primary_abs_kendall_tau_b"] for record in fitted_bin_records],
        "o-",
        label="|tau-b|",
    )
    threshold_axes[1, 0].plot(
        fitted_thresholds,
        [
            record["primary_adjusted_neighborhood_overlap"]
            for record in fitted_bin_records
        ],
        "o-",
        label="local overlap",
    )
    threshold_axes[1, 0].plot(
        fitted_thresholds,
        [record["primary_distance_geometry"] for record in fitted_bin_records],
        "o-",
        label="distance geometry",
    )
    threshold_axes[1, 0].axvline(
        MIN_POSITIVE_BIN_RATE_HZ,
        color="tab:red",
        lw=1,
    )
    threshold_axes[1, 0].set_xlabel("positive-bin cutoff (s⁻¹)")
    threshold_axes[1, 0].set_ylabel("common-neuron agreement with primary 0.10 fit")
    threshold_axes[1, 0].set_title(
        "Changing the included population changes common rows"
    )
    threshold_axes[1, 0].legend(frameon=False)

proxy_metrics = (
    onset_record["primary_abs_kendall_tau_b"],
    onset_record["primary_adjusted_neighborhood_overlap"],
    onset_record["primary_distance_geometry"],
)
threshold_axes[1, 1].bar(
    ("|tau-b|", "local overlap", "distance"),
    proxy_metrics,
    color=("tab:blue", "tab:orange", "tab:green"),
)
threshold_axes[1, 1].set_ylim(0, 1)
threshold_axes[1, 1].set_ylabel("common-neuron agreement with positive-bin fit")
threshold_axes[1, 1].set_title(f"0.10 onset/s alternative · N={onset_roi_rows.size:,}")

threshold_figure.suptitle(f"{recording_name} · active-neuron definition sensitivity")
threshold_path = FIG_DIR / "04_rastermap_04_threshold_sensitivity.png"
threshold_figure.savefig(threshold_path, dpi=150, bbox_inches="tight")
print("saved ->", threshold_path)
if SHOW_FIGURES:
    plt.show()


# %% [markdown]
# ## Step 7 — one-factor-at-a-time Rastermap parameter sensitivity
#
# The final order is algorithm-dependent, so a single parameter setting is not
# sufficient verification.  All PC-count variants below come from one 256-PC
# decomposition; this avoids conflating the requested PC count with differences
# between separate randomized SVD calls.  Other variants change one setting at
# a time.  The lag labels are in frames: the primary/reference lag is the
# physical-duration match (about 12 frames here), five frames is the numerical
# value used in the paper's 3.2-Hz recording, and 20 is the upper paper grid.
# Seed variability from Step 4 remains the relevant baseline for judging the
# magnitude of any parameter effect.  We also show the original 128-PC SVD
# against the first 128 axes of a separately computed 256-PC SVD: this exposes
# numerical decomposition sensitivity instead of silently folding it into the
# PC-count comparison.  Tie-randomized local overlap is below one even for a
# self-comparison because Rastermap leaves within-grid ties unresolved.

# %%
parameter_records: list[dict[str, object]] = []
parameter_reference_summary = primary_summary
parameter_reference_model = None
if RUN_PARAMETER_SWEEP:
    max_parameter_pcs = max(variant[1] for variant in PARAMETER_VARIANTS)
    max_Usv = SVD(primary_model.X, n_components=max_parameter_pcs)
    max_sv = np.sqrt(np.sum(max_Usv * max_Usv, axis=0, dtype=np.float64))
    max_U = max_Usv / max_sv
    max_Vsv = primary_model.X.T @ max_U
    parameter_reference_model, parameter_reference_summary = fit_from_decomposition(
        max_Usv[:, :N_PCS],
        max_Vsv[:, :N_PCS],
        n_clusters=N_CLUSTERS,
        locality=LOCALITY,
        lag_frames=primary_lag_frames,
        seed=REFERENCE_SEED,
    )
    decomposition_replay = compare_fits(
        primary_summary,
        parameter_reference_summary,
    )
    print(
        "256-PC decomposition's 128-PC reference versus primary: "
        f"|tau-b|={decomposition_replay['abs_kendall_tau_b']:.3f}"
    )
    parameter_records.append(
        {
            "variant": "original SVD 128",
            "n_pcs": N_PCS,
            "n_clusters": N_CLUSTERS,
            "lag_frames": primary_lag_frames,
            "lag_seconds": primary_lag_frames / fs,
            "locality": LOCALITY,
            "abs_kendall_tau_b": decomposition_replay["abs_kendall_tau_b"],
            "adjusted_neighborhood_overlap": decomposition_replay[
                "adjusted_neighborhood_overlap"
            ],
            "distance_geometry": embedding_distance_correlation(
                parameter_reference_summary.embedding,
                primary_summary.embedding,
            ),
            "cluster_ari_when_comparable_and_refit": decomposition_replay[
                "cluster_ari"
            ],
            "runtime_seconds": primary_summary.runtime_seconds,
        }
    )

    for label, n_pcs, n_clusters, lag_frames, locality in PARAMETER_VARIANTS:
        effective_lag = primary_lag_frames if lag_frames is None else lag_frames
        if label == "reference":
            variant_summary = parameter_reference_summary
            runtime_seconds = parameter_reference_summary.runtime_seconds
            variant_model = None
        else:
            variant_model, variant_summary = fit_from_decomposition(
                max_Usv[:, :n_pcs],
                max_Vsv[:, :n_pcs],
                n_clusters=n_clusters,
                locality=locality,
                lag_frames=effective_lag,
                seed=REFERENCE_SEED,
            )
            runtime_seconds = variant_summary.runtime_seconds
        metrics = compare_fits(parameter_reference_summary, variant_summary)
        distance_geometry = embedding_distance_correlation(
            parameter_reference_summary.embedding,
            variant_summary.embedding,
        )
        cluster_ari = (
            metrics["cluster_ari"]
            if n_clusters == N_CLUSTERS and (label == "reference" or n_pcs != N_PCS)
            else np.nan
        )
        parameter_records.append(
            {
                "variant": label,
                "n_pcs": n_pcs,
                "n_clusters": n_clusters,
                "lag_frames": effective_lag,
                "lag_seconds": effective_lag / fs,
                "locality": locality,
                "abs_kendall_tau_b": metrics["abs_kendall_tau_b"],
                "adjusted_neighborhood_overlap": metrics[
                    "adjusted_neighborhood_overlap"
                ],
                "distance_geometry": distance_geometry,
                "cluster_ari_when_comparable_and_refit": cluster_ari,
                "runtime_seconds": runtime_seconds,
            }
        )
        if variant_model is not None:
            del variant_model, variant_summary
        gc.collect()

    save_records(
        VALIDATION_DIR / "04_parameter_sensitivity.csv",
        parameter_records,
    )

    parameter_labels = [str(record["variant"]) for record in parameter_records]
    parameter_x = np.arange(len(parameter_records))
    parameter_figure, parameter_axes = plt.subplots(
        2,
        1,
        figsize=(15, 9),
        constrained_layout=True,
        sharex=True,
    )
    parameter_width = 0.25
    parameter_axes[0].bar(
        parameter_x - parameter_width,
        [record["abs_kendall_tau_b"] for record in parameter_records],
        parameter_width,
        label="|tau-b|",
    )
    parameter_axes[0].bar(
        parameter_x,
        [record["adjusted_neighborhood_overlap"] for record in parameter_records],
        parameter_width,
        label="tie-randomized local overlap",
    )
    parameter_axes[0].bar(
        parameter_x + parameter_width,
        [record["distance_geometry"] for record in parameter_records],
        parameter_width,
        label="distance geometry",
    )
    parameter_axes[0].set_ylabel("agreement with decomposition reference")
    parameter_axes[0].set_title("One-factor changes can exceed seed variability")
    parameter_axes[0].legend(frameon=False, ncol=3)

    seed_tau_values = np.array(
        [float(record["abs_kendall_tau_b"]) for record in seed_records]
    )
    seed_local_values = np.array(
        [float(record["adjusted_neighborhood_overlap"]) for record in seed_records]
    )
    parameter_axes[1].axhspan(
        seed_tau_values.min(),
        seed_tau_values.max(),
        color="tab:blue",
        alpha=0.16,
        label="seed |tau-b| range",
    )
    parameter_axes[1].axhspan(
        seed_local_values.min(),
        seed_local_values.max(),
        color="tab:orange",
        alpha=0.16,
        label="seed local-overlap range",
    )
    parameter_axes[1].plot(
        parameter_x,
        [record["abs_kendall_tau_b"] for record in parameter_records],
        "o-",
        color="tab:blue",
    )
    parameter_axes[1].plot(
        parameter_x,
        [record["adjusted_neighborhood_overlap"] for record in parameter_records],
        "o-",
        color="tab:orange",
    )
    parameter_axes[1].set_xticks(
        parameter_x,
        parameter_labels,
        rotation=25,
        ha="right",
    )
    parameter_axes[1].set_ylabel("agreement")
    parameter_axes[1].set_title("Shaded bands: seed 0 versus seeds 1–19")
    parameter_axes[1].legend(frameon=False, ncol=2)
    parameter_figure.suptitle(f"{recording_name} · Rastermap parameter sensitivity")
    parameter_path = FIG_DIR / "04_rastermap_05_parameter_sensitivity.png"
    parameter_figure.savefig(parameter_path, dpi=150, bbox_inches="tight")
    print("saved ->", parameter_path)
    if SHOW_FIGURES:
        plt.show()

    del max_Usv, max_sv, max_U, max_Vsv
    if parameter_reference_model is not None:
        del parameter_reference_model
    gc.collect()


# %% [markdown]
# ## Step 8 — shared-variance components with state-matched blocked folds
#
# Ordinary PCA always finds axes of maximal variance in the same data used to
# fit them; it does not tell us whether those axes are reproducible.  The
# Rastermap paper uses shared variance component analysis (SVCA), which splits
# neurons into two groups, learns cross-population axes on training frames, and
# measures shared covariance on held-out frames.
#
# Paper-style random division of individual frames is optimistic for
# autocorrelated calcium data because adjacent samples from one event can enter
# both folds.  We therefore compare a state-matched random-frame control with a
# stronger primary split: non-overlapping 15-s blocks inside each constant-state
# and acquisition segment, with at least 2-s guards discarded at block edges.
# Each train/test fold receives matching block counts within every state (not
# equal counts across states). Negative held-out shared covariance is retained;
# taking its absolute value would hide unstable components.
#
# A third condition uses the same blocked folds after independently removing
# each labeled state's per-neuron mean within train and test. This removes fixed
# between-state offsets while preserving the global per-neuron scale. It asks
# whether reproducible covariance remains *within* labeled states; it does not
# prove that one universal axis is shared across states.


# %%
def guarded_blocks_by_state(
    state_vector: np.ndarray,
    boundaries: np.ndarray,
    *,
    block_frames: int,
    guard_frames: int,
) -> dict[float, list[np.ndarray]]:
    """Build guarded blocks that never cross a state or acquisition boundary."""
    state_splits = (
        np.flatnonzero(
            (state_vector[1:] != state_vector[:-1])
            | ~np.isfinite(state_vector[1:])
            | ~np.isfinite(state_vector[:-1])
        )
        + 1
    )
    segments = ts.acquisition_segments(
        state_vector.size,
        boundaries,
        extra_splits=state_splits,
    )
    blocks: dict[float, list[np.ndarray]] = {}
    for start, stop in segments:
        code = float(state_vector[start])
        if not np.isfinite(code):
            continue
        for block_start in range(start, stop - block_frames + 1, block_frames):
            inner_start = block_start + guard_frames
            inner_stop = block_start + block_frames - guard_frames
            if inner_stop > inner_start:
                blocks.setdefault(code, []).append(
                    np.arange(inner_start, inner_stop, dtype=np.int64)
                )
    return blocks


def balanced_block_fold(
    blocks: dict[float, list[np.ndarray]],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[float, int]]:
    """Assign equal block counts per state to train and test folds."""
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    frame_counts: dict[float, int] = {}
    for code in sorted(blocks):
        order = rng.permutation(len(blocks[code]))
        blocks_per_fold = order.size // 2
        if blocks_per_fold == 0:
            continue
        train = np.concatenate(
            [blocks[code][index] for index in order[:blocks_per_fold]]
        )
        test = np.concatenate(
            [
                blocks[code][index]
                for index in order[blocks_per_fold : 2 * blocks_per_fold]
            ]
        )
        train_parts.append(train)
        test_parts.append(test)
        frame_counts[code] = train.size
    return (
        np.sort(np.concatenate(train_parts)),
        np.sort(np.concatenate(test_parts)),
        frame_counts,
    )


def matched_random_frame_fold(
    blocks: dict[float, list[np.ndarray]],
    frame_counts: dict[float, int],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomize frames while matching the blocked fold's per-state counts."""
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for code, n_frames_per_fold in frame_counts.items():
        eligible = np.concatenate(blocks[code])
        order = rng.permutation(eligible.size)
        train_parts.append(eligible[order[:n_frames_per_fold]])
        test_parts.append(eligible[order[n_frames_per_fold : 2 * n_frames_per_fold]])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(test_parts))


def center_rows_within_labels(
    activity_matrix: np.ndarray,
    frame_labels: np.ndarray,
) -> np.ndarray:
    """Remove per-neuron state means without rescaling rare-state activity."""
    centered = np.array(activity_matrix, dtype=np.float32, order="C", copy=True)
    frame_labels = np.asarray(frame_labels)
    if frame_labels.ndim != 1 or frame_labels.size != centered.shape[1]:
        raise ValueError("frame_labels must match the activity columns")
    for code in np.unique(frame_labels[np.isfinite(frame_labels)]):
        columns = frame_labels == code
        centered[:, columns] -= centered[:, columns].mean(
            axis=1,
            keepdims=True,
            dtype=np.float32,
        )
    return centered


def svca_split(
    normalized_activity_matrix: np.ndarray,
    first_neurons: np.ndarray,
    second_neurons: np.ndarray,
    train_frames: np.ndarray,
    test_frames: np.ndarray,
    *,
    n_components: int,
    frame_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit exact cross-covariance axes and evaluate them on held-out frames."""
    first_train = normalized_activity_matrix[np.ix_(first_neurons, train_frames)]
    second_train = normalized_activity_matrix[np.ix_(second_neurons, train_frames)]
    first_test = normalized_activity_matrix[np.ix_(first_neurons, test_frames)]
    second_test = normalized_activity_matrix[np.ix_(second_neurons, test_frames)]
    if frame_labels is not None:
        frame_labels = np.asarray(frame_labels)
        if (
            frame_labels.ndim != 1
            or frame_labels.size != normalized_activity_matrix.shape[1]
        ):
            raise ValueError("frame_labels must match normalized_activity_matrix")
        first_train = center_rows_within_labels(
            first_train,
            frame_labels[train_frames],
        )
        second_train = center_rows_within_labels(
            second_train,
            frame_labels[train_frames],
        )
        # Test-fold centering deliberately estimates within-state covariance;
        # the projection axes themselves still come only from training frames.
        first_test = center_rows_within_labels(
            first_test,
            frame_labels[test_frames],
        )
        second_test = center_rows_within_labels(
            second_test,
            frame_labels[test_frames],
        )
    cross_covariance = first_train @ second_train.T / train_frames.size
    left, _, right_transpose = svd(
        cross_covariance,
        full_matrices=False,
        overwrite_a=True,
        check_finite=False,
        lapack_driver="gesdd",
    )
    n_components = min(n_components, left.shape[1])
    first_test_projection = left[:, :n_components].T @ first_test
    second_test_projection = right_transpose[:n_components] @ second_test
    shared_covariance = np.mean(
        first_test_projection * second_test_projection,
        axis=1,
        dtype=np.float64,
    )
    total_variance = np.mean(
        (first_test_projection**2 + second_test_projection**2) / 2,
        axis=1,
        dtype=np.float64,
    )
    return shared_covariance, total_variance


svca_component_records: list[dict[str, object]] = []
svca_aggregate_records: list[dict[str, object]] = []
svca_split_records: list[dict[str, object]] = []
svca_reliable_prefix = 0
centered_svca_reliable_prefix = 0
if RUN_SVCA:
    block_frames = int(round(SVCA_BLOCK_SECONDS * fs))
    guard_frames = int(np.ceil(SVCA_GUARD_SECONDS * fs))
    if block_frames <= 2 * guard_frames:
        raise ValueError("SVCA block duration must exceed twice the guard duration")
    state_blocks = guarded_blocks_by_state(
        state,
        boundary_ind,
        block_frames=block_frames,
        guard_frames=guard_frames,
    )
    zscored_activity = activity.copy()
    zscored_activity -= zscored_activity.mean(
        axis=1,
        keepdims=True,
        dtype=np.float32,
    )
    zscored_scale = zscored_activity.std(
        axis=1,
        keepdims=True,
        dtype=np.float32,
    )
    zscored_activity /= zscored_scale

    blocked_shared = np.empty((SVCA_REPEATS, SVCA_COMPONENTS))
    blocked_total = np.empty_like(blocked_shared)
    centered_blocked_shared = np.empty_like(blocked_shared)
    centered_blocked_total = np.empty_like(blocked_shared)
    random_shared = np.empty_like(blocked_shared)
    random_total = np.empty_like(blocked_shared)
    retained_state_counts: dict[float, int] = {}
    for repetition in range(SVCA_REPEATS):
        neuron_rng = np.random.default_rng(30_000 + repetition)
        neuron_order = neuron_rng.permutation(activity.shape[0])
        neurons_per_group = activity.shape[0] // 2
        first_neurons = neuron_order[:neurons_per_group]
        second_neurons = neuron_order[neurons_per_group : 2 * neurons_per_group]
        blocked_train, blocked_test, retained_state_counts = balanced_block_fold(
            state_blocks,
            seed=40_000 + repetition,
        )
        random_train, random_test = matched_random_frame_fold(
            state_blocks,
            retained_state_counts,
            seed=50_000 + repetition,
        )
        blocked_shared[repetition], blocked_total[repetition] = svca_split(
            zscored_activity,
            first_neurons,
            second_neurons,
            blocked_train,
            blocked_test,
            n_components=SVCA_COMPONENTS,
        )
        (
            centered_blocked_shared[repetition],
            centered_blocked_total[repetition],
        ) = svca_split(
            zscored_activity,
            first_neurons,
            second_neurons,
            blocked_train,
            blocked_test,
            n_components=SVCA_COMPONENTS,
            frame_labels=state,
        )
        random_shared[repetition], random_total[repetition] = svca_split(
            zscored_activity,
            first_neurons,
            second_neurons,
            random_train,
            random_test,
            n_components=SVCA_COMPONENTS,
        )
        print(
            f"  SVCA split {repetition + 1:02d}/{SVCA_REPEATS}: "
            f"{blocked_train.size:,} train + {blocked_test.size:,} test frames",
            flush=True,
        )

    blocked_lower = np.percentile(blocked_shared, 2.5, axis=0)
    blocked_median = np.median(blocked_shared, axis=0)
    blocked_upper = np.percentile(blocked_shared, 97.5, axis=0)
    centered_blocked_lower = np.percentile(
        centered_blocked_shared,
        2.5,
        axis=0,
    )
    centered_blocked_median = np.median(centered_blocked_shared, axis=0)
    centered_blocked_upper = np.percentile(
        centered_blocked_shared,
        97.5,
        axis=0,
    )
    random_median = np.median(random_shared, axis=0)
    positive_lower = blocked_lower > 0
    first_failure = np.flatnonzero(~positive_lower)
    svca_reliable_prefix = (
        int(first_failure[0]) if first_failure.size else SVCA_COMPONENTS
    )
    centered_positive_lower = centered_blocked_lower > 0
    centered_first_failure = np.flatnonzero(~centered_positive_lower)
    centered_svca_reliable_prefix = (
        int(centered_first_failure[0])
        if centered_first_failure.size
        else SVCA_COMPONENTS
    )

    for repetition in range(SVCA_REPEATS):
        for split_type, shared_values, total_values in (
            ("state_matched_blocks", blocked_shared, blocked_total),
            (
                "state_matched_blocks_within_state_centered",
                centered_blocked_shared,
                centered_blocked_total,
            ),
            ("state_matched_random_frames", random_shared, random_total),
        ):
            for component in range(SVCA_COMPONENTS):
                svca_split_records.append(
                    {
                        "repetition": repetition,
                        "split_type": split_type,
                        "component": component + 1,
                        "shared_covariance": shared_values[repetition, component],
                        "total_variance": total_values[repetition, component],
                    }
                )

    for component in range(SVCA_COMPONENTS):
        svca_component_records.append(
            {
                "component": component + 1,
                "blocked_shared_median": blocked_median[component],
                "blocked_shared_percentile025": blocked_lower[component],
                "blocked_shared_percentile975": blocked_upper[component],
                "blocked_fraction_median": np.median(
                    blocked_shared[:, component] / blocked_total[:, component]
                ),
                "within_state_centered_blocked_shared_median": (
                    centered_blocked_median[component]
                ),
                "within_state_centered_blocked_shared_percentile025": (
                    centered_blocked_lower[component]
                ),
                "within_state_centered_blocked_shared_percentile975": (
                    centered_blocked_upper[component]
                ),
                "within_state_centered_blocked_fraction_median": np.median(
                    centered_blocked_shared[:, component]
                    / centered_blocked_total[:, component]
                ),
                "random_frame_shared_median": random_median[component],
                "random_frame_fraction_median": np.median(
                    random_shared[:, component] / random_total[:, component]
                ),
            }
        )

    aggregate_components = (10, 30, 64, 128, 256, 512)
    blocked_aggregate_values: list[np.ndarray] = []
    centered_blocked_aggregate_values: list[np.ndarray] = []
    random_aggregate_values: list[np.ndarray] = []
    for n_components in aggregate_components:
        blocked_fraction = blocked_shared[:, :n_components].sum(axis=1) / blocked_total[
            :, :n_components
        ].sum(axis=1)
        random_fraction = random_shared[:, :n_components].sum(axis=1) / random_total[
            :, :n_components
        ].sum(axis=1)
        centered_blocked_fraction = centered_blocked_shared[:, :n_components].sum(
            axis=1
        ) / centered_blocked_total[:, :n_components].sum(axis=1)
        blocked_aggregate_values.append(blocked_fraction)
        centered_blocked_aggregate_values.append(centered_blocked_fraction)
        random_aggregate_values.append(random_fraction)
        svca_aggregate_records.append(
            {
                "top_components": n_components,
                "blocked_fraction_median": np.median(blocked_fraction),
                "blocked_fraction_percentile025": np.percentile(blocked_fraction, 2.5),
                "blocked_fraction_percentile975": np.percentile(blocked_fraction, 97.5),
                "within_state_centered_blocked_fraction_median": np.median(
                    centered_blocked_fraction
                ),
                "within_state_centered_blocked_fraction_percentile025": np.percentile(
                    centered_blocked_fraction,
                    2.5,
                ),
                "within_state_centered_blocked_fraction_percentile975": np.percentile(
                    centered_blocked_fraction,
                    97.5,
                ),
                "random_frame_fraction_median": np.median(random_fraction),
                "random_frame_fraction_percentile025": np.percentile(
                    random_fraction, 2.5
                ),
                "random_frame_fraction_percentile975": np.percentile(
                    random_fraction, 97.5
                ),
            }
        )
    save_records(
        VALIDATION_DIR / "04_svca_components.csv",
        svca_component_records,
    )
    save_records(
        VALIDATION_DIR / "04_svca_aggregate.csv",
        svca_aggregate_records,
    )
    save_records(
        VALIDATION_DIR / "04_svca_split_values.csv",
        svca_split_records,
    )

    svca_x = np.arange(1, SVCA_COMPONENTS + 1)
    svca_figure, svca_axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        constrained_layout=True,
    )
    svca_axes[0, 0].plot(
        svca_x,
        blocked_median,
        color="tab:blue",
        label="state-matched blocked",
    )
    svca_axes[0, 0].fill_between(
        svca_x,
        blocked_lower,
        blocked_upper,
        color="tab:blue",
        alpha=0.18,
        label="2.5–97.5% split-to-split range",
    )
    svca_axes[0, 0].plot(
        svca_x,
        centered_blocked_median,
        color="tab:purple",
        label="blocked + within-state centered",
    )
    svca_axes[0, 0].plot(
        svca_x,
        random_median,
        color="tab:orange",
        alpha=0.8,
        label="matched random frames",
    )
    svca_axes[0, 0].axhline(0, color="black", lw=0.8)
    svca_axes[0, 0].set_xscale("log")
    svca_axes[0, 0].set_yscale("symlog", linthresh=0.01)
    svca_axes[0, 0].set_xlabel("SVCA component")
    svca_axes[0, 0].set_ylabel("held-out shared covariance")
    svca_axes[0, 0].set_title("Random-frame splitting is optimistic")
    svca_axes[0, 0].legend(frameon=False)

    blocked_fraction_components = blocked_shared / blocked_total
    centered_blocked_fraction_components = (
        centered_blocked_shared / centered_blocked_total
    )
    random_fraction_components = random_shared / random_total
    svca_axes[0, 1].plot(
        svca_x,
        np.median(blocked_fraction_components, axis=0),
        color="tab:blue",
        label="blocked",
    )
    svca_axes[0, 1].plot(
        svca_x,
        np.median(centered_blocked_fraction_components, axis=0),
        color="tab:purple",
        label="blocked + state-centered",
    )
    svca_axes[0, 1].plot(
        svca_x,
        np.median(random_fraction_components, axis=0),
        color="tab:orange",
        label="random frames",
    )
    svca_axes[0, 1].axhline(0, color="black", lw=0.8)
    svca_axes[0, 1].set_xscale("log")
    svca_axes[0, 1].set_ylim(-0.25, 1.05)
    svca_axes[0, 1].set_xlabel("SVCA component")
    svca_axes[0, 1].set_ylabel("shared / total variance")
    svca_axes[0, 1].set_title(
        "Positive-range prefixes: "
        f"raw={svca_reliable_prefix}, state-centered="
        f"{centered_svca_reliable_prefix}"
    )
    svca_axes[0, 1].legend(frameon=False)

    aggregate_x = np.arange(len(aggregate_components))
    blocked_medians = np.array(
        [np.median(values) for values in blocked_aggregate_values]
    )
    random_medians = np.array([np.median(values) for values in random_aggregate_values])
    centered_blocked_medians = np.array(
        [np.median(values) for values in centered_blocked_aggregate_values]
    )
    blocked_error = np.vstack(
        (
            blocked_medians
            - [np.percentile(values, 2.5) for values in blocked_aggregate_values],
            [np.percentile(values, 97.5) for values in blocked_aggregate_values]
            - blocked_medians,
        )
    )
    svca_axes[1, 0].errorbar(
        aggregate_x,
        blocked_medians,
        yerr=blocked_error,
        fmt="o-",
        capsize=3,
        label="blocked",
    )
    centered_blocked_error = np.vstack(
        (
            centered_blocked_medians
            - [
                np.percentile(values, 2.5)
                for values in centered_blocked_aggregate_values
            ],
            [
                np.percentile(values, 97.5)
                for values in centered_blocked_aggregate_values
            ]
            - centered_blocked_medians,
        )
    )
    svca_axes[1, 0].errorbar(
        aggregate_x,
        centered_blocked_medians,
        yerr=centered_blocked_error,
        fmt="o-",
        color="tab:purple",
        capsize=3,
        label="blocked + state-centered",
    )
    svca_axes[1, 0].plot(
        aggregate_x,
        random_medians,
        "o-",
        label="random frames",
    )
    svca_axes[1, 0].set_xticks(aggregate_x, aggregate_components)
    svca_axes[1, 0].set_xlabel("first k components")
    svca_axes[1, 0].set_ylabel("sum(shared covariance) / sum(total variance)")
    svca_axes[1, 0].set_title("Aggregate reproducible variance")
    svca_axes[1, 0].legend(frameon=False)

    state_count_codes = np.array(sorted(retained_state_counts))
    state_count_values = np.array(
        [retained_state_counts[code] for code in state_count_codes]
    )
    state_count_labels = [
        state_code_labels.get(float(code), str(code)) for code in state_count_codes
    ]
    svca_axes[1, 1].bar(state_count_labels, state_count_values, color="tab:green")
    svca_axes[1, 1].tick_params(axis="x", rotation=20)
    svca_axes[1, 1].set_ylabel("frames in each train/test fold")
    svca_axes[1, 1].set_title(
        f"Matched states · {SVCA_BLOCK_SECONDS:g}-s blocks, "
        f"≥{SVCA_GUARD_SECONDS:g}-s guards"
    )
    svca_figure.suptitle(
        f"{recording_name} · blocked shared variance before/after state-mean removal"
    )
    svca_path = FIG_DIR / "04_rastermap_06_blocked_svca.png"
    svca_figure.savefig(svca_path, dpi=150, bbox_inches="tight")
    print("saved ->", svca_path)
    if SHOW_FIGURES:
        plt.show()

    print(
        f"Blocked SVCA positive-range contiguous prefix in this {SVCA_REPEATS}-split run: "
        f"{svca_reliable_prefix}/{SVCA_COMPONENTS} components (not a precise "
        "dimensionality estimate)"
    )
    print(
        "Within-state-centered blocked prefix in the same folds: "
        f"{centered_svca_reliable_prefix}/{SVCA_COMPONENTS} components"
    )
    del zscored_activity
    gc.collect()


# %% [markdown]
# ## Step 9 — end-to-end synthetic positive and negative controls
#
# Real data have no known correct one-dimensional neuron order.  This cell
# therefore generates five independent datasets with a known latent order:
# twenty noisy traveling waves traverse 1,200 neurons, the input rows are
# randomly permuted, and Rastermap must recover the hidden position.  A paired
# negative control independently circular-shifts every neuron, preserving its
# values and autocorrelation while destroying coordination.
#
# The explicit synthetic settings (64 PCs, 50 clusters, locality 0.75, lag 8)
# are appropriate for this clean sequence-recovery smoke test; they are not a
# post-hoc recommendation for the sleep data.  No activity cutoff is needed
# because the simulator gives every neuron comparable event counts.


# %%
def simulate_traveling_waves(
    *,
    seed: int,
    n_neurons: int,
    n_frames: int,
    n_waves: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return shuffled neuron rows and their aligned known latent positions."""
    rng = np.random.default_rng(seed)
    latent_position = np.linspace(0, 1, n_neurons, dtype=np.float32)
    rate = np.full((n_neurons, n_frames), 0.003, dtype=np.float32)
    frame_axis = np.arange(n_frames, dtype=np.float32)[None, :]
    wave_onsets = np.linspace(100, n_frames - 400, n_waves).astype(np.int64)
    wave_onsets += rng.integers(-40, 41, size=n_waves)
    for onset in wave_onsets:
        duration = int(rng.integers(120, 261))
        width = float(rng.uniform(5, 10))
        amplitude = rng.uniform(0.2, 0.5, n_neurons).astype(np.float32)
        centers = onset + duration * latent_position
        rate += amplitude[:, None] * np.exp(
            -0.5 * ((frame_axis - centers[:, None]) / width) ** 2
        )

    global_nuisance = gaussian_filter1d(rng.standard_normal(n_frames), 40)
    global_nuisance -= global_nuisance.min()
    global_nuisance /= np.ptp(global_nuisance) + 1e-6
    rate += 0.01 * global_nuisance
    simulated_activity = rng.poisson(rate).astype(np.float32)
    input_permutation = rng.permutation(n_neurons)
    return (
        simulated_activity[input_permutation],
        latent_position[input_permutation],
    )


def known_order_metrics(
    latent_position: np.ndarray,
    fitted_embedding: np.ndarray,
) -> dict[str, float]:
    """Compare a fitted one-dimensional embedding with known ground truth."""
    tau = kendalltau(latent_position, fitted_embedding, variant="b").statistic
    raw_overlap = rmt.rank_neighborhood_overlap(
        latent_position,
        fitted_embedding,
        neighborhood_size=NEIGHBORHOOD_SIZE,
        tie_permutations=TIE_PERMUTATIONS,
        random_state=0,
    )
    chance = NEIGHBORHOOD_SIZE / (latent_position.size - 1)
    return {
        "abs_kendall_tau_b": float(abs(tau)),
        "adjusted_neighborhood_overlap": float((raw_overlap - chance) / (1 - chance)),
        "distance_geometry": embedding_distance_correlation(
            latent_position,
            fitted_embedding,
        ),
    }


synthetic_records: list[dict[str, object]] = []
synthetic_display = None
if RUN_SYNTHETIC_CONTROL:
    for repetition in range(SYNTHETIC_REPEATS):
        simulated_activity, latent_position = simulate_traveling_waves(
            seed=60_000 + repetition,
            n_neurons=SYNTHETIC_NEURONS,
            n_frames=SYNTHETIC_FRAMES,
            n_waves=SYNTHETIC_WAVES,
        )
        keep_display = repetition == 0
        signal_model, signal_summary = fit_activity(
            simulated_activity,
            n_clusters=SYNTHETIC_N_CLUSTERS,
            n_pcs=SYNTHETIC_N_PCS,
            locality=SYNTHETIC_LOCALITY,
            lag_frames=SYNTHETIC_LAG_FRAMES,
            seed=REFERENCE_SEED,
            keep_normalized=keep_display,
        )
        shifted_activity = independently_shift_rows_within_segments(
            simulated_activity,
            [(0, SYNTHETIC_FRAMES)],
            min_shift=SYNTHETIC_LAG_FRAMES + 1,
            seed=70_000 + repetition,
        )
        null_model, null_summary = fit_activity(
            shifted_activity,
            n_clusters=SYNTHETIC_N_CLUSTERS,
            n_pcs=SYNTHETIC_N_PCS,
            locality=SYNTHETIC_LOCALITY,
            lag_frames=SYNTHETIC_LAG_FRAMES,
            seed=REFERENCE_SEED,
            keep_normalized=keep_display,
        )
        signal_metrics = known_order_metrics(
            latent_position,
            signal_summary.embedding,
        )
        null_metrics = known_order_metrics(
            latent_position,
            null_summary.embedding,
        )
        positive_fraction = float(np.mean(simulated_activity > 0))
        for control, metrics, runtime_seconds in (
            ("coordinated_waves", signal_metrics, signal_summary.runtime_seconds),
            ("independent_row_shifts", null_metrics, null_summary.runtime_seconds),
        ):
            synthetic_records.append(
                {
                    "repetition": repetition,
                    "control": control,
                    "n_neurons": SYNTHETIC_NEURONS,
                    "n_frames": SYNTHETIC_FRAMES,
                    "n_waves": SYNTHETIC_WAVES,
                    "positive_bin_fraction": positive_fraction,
                    **metrics,
                    "runtime_seconds": runtime_seconds,
                }
            )

        if keep_display:
            true_order = np.argsort(latent_position, kind="stable")
            signal_embedding = signal_summary.embedding.copy()
            if spearmanr(latent_position, signal_embedding).statistic < 0:
                signal_embedding *= -1
            null_embedding = null_summary.embedding.copy()
            if spearmanr(latent_position, null_embedding).statistic < 0:
                null_embedding *= -1
            synthetic_display = {
                "true_superneurons": superneuron_display(
                    signal_model.X,
                    true_order,
                ),
                "signal_superneurons": superneuron_display(
                    signal_model.X,
                    signal_summary.order,
                ),
                "null_superneurons": superneuron_display(
                    null_model.X,
                    null_summary.order,
                ),
                "latent_position": latent_position.copy(),
                "signal_embedding": signal_embedding,
                "null_embedding": null_embedding,
            }

        del (
            simulated_activity,
            shifted_activity,
            signal_model,
            null_model,
            signal_summary,
            null_summary,
        )
        gc.collect()

    save_records(
        VALIDATION_DIR / "04_synthetic_controls.csv",
        synthetic_records,
    )
    signal_records = [
        record
        for record in synthetic_records
        if record["control"] == "coordinated_waves"
    ]
    null_records = [
        record
        for record in synthetic_records
        if record["control"] == "independent_row_shifts"
    ]
    assert synthetic_display is not None
    synthetic_figure = plt.figure(figsize=(16, 10), constrained_layout=True)
    synthetic_grid = synthetic_figure.add_gridspec(2, 3, height_ratios=(1.25, 1))
    synthetic_display_frames = min(1_500, SYNTHETIC_FRAMES)
    for column, (values, title) in enumerate(
        (
            (synthetic_display["true_superneurons"], "Known latent order"),
            (synthetic_display["signal_superneurons"], "Rastermap: coordinated waves"),
            (synthetic_display["null_superneurons"], "Rastermap: row-shift null"),
        )
    ):
        axis = synthetic_figure.add_subplot(synthetic_grid[0, column])
        axis.imshow(
            values[:, :synthetic_display_frames],
            aspect="auto",
            cmap="gray_r",
            vmin=0,
            vmax=1.5,
            interpolation="nearest",
            rasterized=True,
        )
        axis.set_xlabel("frame")
        axis.set_title(title)
        if column == 0:
            axis.set_ylabel("50-neuron superneurons")

    scatter_axis = synthetic_figure.add_subplot(synthetic_grid[1, 0])
    scatter_axis.scatter(
        synthetic_display["latent_position"],
        synthetic_display["signal_embedding"],
        s=4,
        alpha=0.35,
        label="coordinated",
        rasterized=True,
    )
    scatter_axis.scatter(
        synthetic_display["latent_position"],
        synthetic_display["null_embedding"],
        s=4,
        alpha=0.25,
        label="row-shift null",
        rasterized=True,
    )
    scatter_axis.set_xlabel("known latent position")
    scatter_axis.set_ylabel("aligned Rastermap embedding")
    scatter_axis.set_title("Ground truth is never supplied to Rastermap")
    scatter_axis.legend(frameon=False)

    metric_names = (
        "abs_kendall_tau_b",
        "adjusted_neighborhood_overlap",
        "distance_geometry",
    )
    metric_labels = ("|tau-b|", "local overlap", "distance geometry")
    metric_axis = synthetic_figure.add_subplot(synthetic_grid[1, 1:])
    metric_x = np.arange(len(metric_names))
    for repetition in range(SYNTHETIC_REPEATS):
        signal_values = [signal_records[repetition][metric] for metric in metric_names]
        null_values = [null_records[repetition][metric] for metric in metric_names]
        for metric_index in range(len(metric_names)):
            metric_axis.plot(
                [metric_x[metric_index] - 0.12, metric_x[metric_index] + 0.12],
                [signal_values[metric_index], null_values[metric_index]],
                color="0.65",
                lw=1,
                zorder=1,
            )
        metric_axis.scatter(
            metric_x - 0.12,
            signal_values,
            color="tab:blue",
            label="coordinated" if repetition == 0 else None,
            zorder=2,
        )
        metric_axis.scatter(
            metric_x + 0.12,
            null_values,
            color="tab:orange",
            label="row-shift null" if repetition == 0 else None,
            zorder=2,
        )
    metric_axis.set_xticks(metric_x, metric_labels)
    metric_axis.set_ylim(-0.05, 1.05)
    metric_axis.set_ylabel("recovery of known order")
    metric_axis.set_title(f"Paired controls across {SYNTHETIC_REPEATS} simulations")
    metric_axis.legend(frameon=False)

    synthetic_figure.suptitle(
        "End-to-end Rastermap positive/negative control (algorithm smoke test)"
    )
    synthetic_path = FIG_DIR / "04_rastermap_07_synthetic_controls.png"
    synthetic_figure.savefig(synthetic_path, dpi=150, bbox_inches="tight")
    print("saved ->", synthetic_path)
    if SHOW_FIGURES:
        plt.show()


# %% [markdown]
# ## Step 10 — evidence summary
#
# Passing normalization/replay checks in Tutorial 03 establishes that the code
# calls Rastermap correctly.  The checks here answer a different question:
# whether this dataset supports a stable biological interpretation.  A clean
# synthetic recovery is necessary but not sufficient; real-data seed,
# independent-time, cutoff, parameter, and shared-variance results remain the
# relevant limitations.

# %%
seed_tau_median = float(
    np.median([record["abs_kendall_tau_b"] for record in seed_records])
)
seed_local_median = float(
    np.median([record["adjusted_neighborhood_overlap"] for record in seed_records])
)
summary_record: dict[str, object] = {
    "recording": recording_name,
    "recorded_neurons": n_recorded_neurons,
    "primary_active_neurons": primary_roi_rows.size,
    "primary_positive_bin_rate_threshold_per_second": MIN_POSITIVE_BIN_RATE_HZ,
    "primary_lag_frames": primary_lag_frames,
    "primary_lag_seconds": primary_lag_frames / fs,
    "primary_lag_pair_fraction_crossing_acquisition_breaks": (
        lag_boundary_pair_fraction
    ),
    "primary_occupied_embedding_positions": primary_unique_positions,
    "primary_fraction_neurons_in_tied_positions": primary_tied_neuron_fraction,
    "seed_abs_kendall_tau_b_median": seed_tau_median,
    "seed_adjusted_local_overlap_median": seed_local_median,
    "temporal_split_frame": half_frame,
    "separate_bout_boundary_block_abs_kendall_tau_b": temporal_metrics[
        "abs_kendall_tau_b"
    ],
    "separate_bout_boundary_block_adjusted_local_overlap": temporal_metrics[
        "adjusted_neighborhood_overlap"
    ],
    "heldout_pair_score_first_to_second": first_to_second_pair_score,
    "heldout_pair_score_second_to_first": second_to_first_pair_score,
    "heldout_activity_rank_score_first_to_second": activity_to_second_pair_score,
    "heldout_activity_rank_score_second_to_first": activity_to_first_pair_score,
    "heldout_shift_null_mean": float(shift_null_pair_scores.mean()),
    "lagged_cluster_objective_first_to_second": (lagged_objective_first_to_second),
    "lagged_cluster_objective_activity_first_to_second": (
        lagged_objective_activity_first_to_second
    ),
    "lagged_cluster_objective_second_to_first": (lagged_objective_second_to_first),
    "lagged_cluster_objective_activity_second_to_first": (
        lagged_objective_activity_second_to_first
    ),
    "blocked_svca_positive_range_prefix_this_run": (
        svca_reliable_prefix if RUN_SVCA else np.nan
    ),
    "within_state_centered_blocked_svca_positive_range_prefix_this_run": (
        centered_svca_reliable_prefix if RUN_SVCA else np.nan
    ),
}
for record in state_stratified_records:
    safe_state = str(record["state_label"]).lower().replace(" ", "_")
    summary_record[f"state_{safe_state}_evaluable"] = record["evaluable"]
    summary_record[f"state_{safe_state}_not_evaluable_reason"] = record[
        "not_evaluable_reason"
    ]
    summary_record[f"state_{safe_state}_rastermap_first_to_second"] = record[
        "first_order_on_second_state"
    ]
    summary_record[f"state_{safe_state}_rastermap_second_to_first"] = record[
        "second_order_on_first_state"
    ]
    summary_record[f"state_{safe_state}_activity_rank_first_to_second"] = record[
        "activity_rank_first_to_second_state"
    ]
for record in state_selection_records:
    safe_state = str(record["state_label"]).lower().replace(" ", "_")
    summary_record[f"state_{safe_state}_active_by_within_state_proxy"] = record[
        "selected_within_state"
    ]
    summary_record[f"state_{safe_state}_active_within_state_not_global"] = record[
        "selected_within_state_not_global"
    ]
if synthetic_records:
    summary_record["synthetic_signal_tau_median"] = float(
        np.median(
            [
                record["abs_kendall_tau_b"]
                for record in synthetic_records
                if record["control"] == "coordinated_waves"
            ]
        )
    )
    summary_record["synthetic_null_tau_median"] = float(
        np.median(
            [
                record["abs_kendall_tau_b"]
                for record in synthetic_records
                if record["control"] == "independent_row_shifts"
            ]
        )
    )
save_records(
    VALIDATION_DIR / "04_validation_summary.csv",
    [summary_record],
)

print("Rastermap robustness verification complete.")
print(
    f"  active population: {primary_roi_rows.size:,}/{n_recorded_neurons:,} "
    f"at ≥{MIN_POSITIVE_BIN_RATE_HZ:.2f} positive bins/s"
)
print(
    f"  embedding resolution: {primary_unique_positions:,} occupied positions; "
    f"{100 * primary_tied_neuron_fraction:.1f}% of neurons are in tied positions"
)
print(
    f"  seed stability: median |tau-b|={seed_tau_median:.3f}, "
    f"tie-randomized local overlap={seed_local_median:.3f}"
)
print(
    "  separate bout-boundary blocks: "
    f"|tau-b|={temporal_metrics['abs_kendall_tau_b']:.3f}, "
    f"local overlap={temporal_metrics['adjusted_neighborhood_overlap']:.3f}"
)
print(
    f"  held-out pair score A→B={first_to_second_pair_score:.3f} "
    f"versus activity-rank={activity_to_second_pair_score:.3f} and "
    f"shift-null mean={shift_null_pair_scores.mean():.3f}"
)
print(
    "  lag-aware cluster objective A→B="
    f"{lagged_objective_first_to_second:.4f} versus activity-rank="
    f"{lagged_objective_activity_first_to_second:.4f}; B→A="
    f"{lagged_objective_second_to_first:.4f} versus activity-rank="
    f"{lagged_objective_activity_second_to_first:.4f}"
)
if RUN_SVCA:
    print(
        "  blocked SVCA positive-range prefixes: raw="
        f"{svca_reliable_prefix}, after within-state centering="
        f"{centered_svca_reliable_prefix} (run-specific, not dimensionality estimates)"
    )
if synthetic_records:
    print(
        "  synthetic |tau-b|: coordinated median="
        f"{summary_record['synthetic_signal_tau_median']:.3f}, row-shift median="
        f"{summary_record['synthetic_null_tau_median']:.3f}"
    )
print(
    "Conclusion: the implementation can recover known sequences, but the real-data "
    "ordering must be presented with its seed dependence and poor transport across "
    "these state-composition-shifted blocks—not as a unique, state-independent set "
    "of modules."
)
