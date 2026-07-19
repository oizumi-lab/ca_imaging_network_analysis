# %% [markdown]
# # 08 · Does Rastermap depend on how an “active neuron” is defined?
#
# Tutorials 05–07 established that coarse Rastermap structure can transfer even
# when the fine neuron order is unstable. This tutorial asks whether that result
# survives a more defensible active-neuron audit across all six sleep and four
# anesthesia recordings.
#
# The primary population is the dataset's own published-window population:
# finite, nonconstant rows intersected with `nonzero_ROI`. The supplied mask is
# reconstructed before any fit by requiring at least one positive OASIS sample
# in every complete `frame.used_frame` qualification window. Those windows are
# 1,500 frames for sleep and 2,900 frames for anesthesia; incomplete remainders
# are deliberately excluded, as they were when the mask was made.
#
# After one common A/B numerical-validity screen, two nested sensitivity families
# retain the top 90%, 75%, and 50% of the fit-eligible primary population within
# each recording:
#
# - positive-run-onset rate: transitions from zero to positive, reset at every
#   qualification window, `used_frame` gap, and microscope acquisition break;
# - positive-bin rate: recorded samples whose OASIS estimate is positive.
#
# Ties at a cutoff are all retained, so the actual population can be slightly
# larger than its nominal fraction. Neither quantity is a calibrated spike or
# physiological firing rate. No neuron is randomly sampled.

# %% Step 0 — imports
import csv
import gc
import hashlib
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
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from src.funcnet import dataio, rastermap_tools as rmt, timeseries as ts
from src.funcnet.paths import FIG_DIR, RESULTS_DIR

FIG_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_DIR = RESULTS_DIR / "rastermap_validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Step 1 — settings and execution switch
#
# - `RUN_FULL_ANALYSIS=True` runs the 140 official fits (seven populations ×
#   two folds × ten recordings). Set it to `False` for a quick selection-only
#   audit; completed fits are reused from exact checkpoints.
# - `RETAINED_FRACTIONS` creates nested, recording-relative sensitivity arms.
#   Relative ranks avoid pretending that an absolute OASIS proxy threshold has
#   the same physiological meaning across recordings.
# - One state-matched A/B block allocation is fixed for each recording. Blocks
#   are 30 s long, 3 s are removed from both edges, and lag-wide row-mean seams
#   prevent Rastermap from correlating unrelated block endpoints.
# - Tie-aware local overlap uses `TIE_METRIC_SEED_BASE + recording_index`, shared
#   across all populations of a recording so cutoff comparisons receive common
#   tie-breaking noise. That seed rule is part of checkpoint provenance.
# - `REUSE_COMPLETE_RESULTS=True` reuses a population only when its source-file
#   signature, exact selected-row hash, block allocation, package version, and
#   every method setting match the checkpoint.
# - `SHOW_FIGURES=True` opens each completed figure interactively after saving.

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
RETAINED_FRACTIONS = (0.90, 0.75, 0.50)
QUALIFICATION_WINDOW_FRAMES = {"sleep": 1500, "anesthesia": 2900}

N_CLUSTERS = 100
N_PCS = 128
LOCALITY = 0.0
MEAN_TIME = True
PAPER_LAG_SECONDS = 5 / 3.2
FIT_SEED = 0

SPLIT_SEED_BASE = 80_000
BLOCK_SECONDS = 30.0
BLOCK_EDGE_GUARD_SECONDS = 3.0
NEIGHBORHOOD_SIZE = 50
TIE_PERMUTATIONS = 8
TIE_METRIC_SEED_BASE = 900_000
TIE_METRIC_SEED_SCHEME = "base_plus_recording_index_shared_across_populations"
MIN_NEURONS_FOR_FIT = 256

RUN_FULL_ANALYSIS = True
REUSE_COMPLETE_RESULTS = True
SHOW_FIGURES = True

SELECTION_PATH = VALIDATION_DIR / "08_active_selection_populations.csv"
OVERLAP_PATH = VALIDATION_DIR / "08_active_selection_overlap.csv"
FIT_PATH = VALIDATION_DIR / "08_active_sensitivity_fits.csv"
SUMMARY_PATH = VALIDATION_DIR / "08_active_sensitivity_summary.csv"
RASTERMAP_VERSION = version("rastermap")


# %% [markdown]
# ## Step 2 — lightweight loading and exact qualification-window proxies
#
# `load_analysis_arrays` reads only the OASIS estimate, state annotations,
# acquisition boundaries, `used_frame`, and `nonzero_ROI`; raw ΔF/F and the
# smoothed matrix are not loaded.
#
# `qualification_proxy_counts` has two important safeguards. First, it analyzes
# only complete published qualification windows and discards their remainders.
# Second, an onset is restarted at each window boundary, whenever two neighboring
# entries in `used_frame` are not neighboring original frames, and after every
# `boundary_ind` microscope break even if stored frame numbers remain adjacent.
# This prevents concatenated sleep bouts and acquisition segments from being
# treated as continuous time.


# %%
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
    raise ValueError(f"Cannot infer condition from {recording_name!r}")


def load_analysis_arrays(recording_name: str):
    """Load only arrays required for selection and Rastermap verification."""
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
    """Per-neuron proxy totals and qualification-window provenance."""

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
    """Count bins/onsets in complete windows, resetting at gaps and breaks."""
    n_neurons = activity.shape[0]
    boundary_ind = np.asarray(boundary_ind, dtype=np.int64).ravel()
    valid_boundaries = boundary_ind[
        (boundary_ind >= 0) & (boundary_ind < activity.shape[1])
    ]
    break_after_frame = np.zeros(activity.shape[1], dtype=bool)
    break_after_frame[valid_boundaries] = True
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
            values = activity[:, frames]
            positive = values > 0
            bin_counts = np.count_nonzero(positive, axis=1)
            positive_bins += bin_counts
            reconstructed &= bin_counts > 0

            # Every new window starts a run. A selected-frame gap or a
            # microscope break after the preceding frame does the same.
            starts = np.ones(window_frames, dtype=bool)
            starts[1:] = (np.diff(frames) != 1) | break_after_frame[frames[:-1]]
            onset = positive.copy()
            onset[:, 1:] &= (~positive[:, :-1]) | starts[np.newaxis, 1:]
            positive_run_onsets += np.count_nonzero(onset, axis=1)
            analyzed_frames += window_frames

    if analyzed_frames == 0 or any(count == 0 for count in windows_by_state):
        raise ValueError("Each state must provide at least one complete window")
    return QualificationSummary(
        positive_bins=positive_bins,
        positive_run_onsets=positive_run_onsets,
        analyzed_frames=analyzed_frames,
        windows_by_state=(windows_by_state[0], windows_by_state[1]),
        reconstructed_nonzero_roi=reconstructed,
    )


def primary_segments(
    state: np.ndarray,
    boundary_ind: np.ndarray,
    allowed_codes: tuple[float, ...],
) -> list[tuple[int, int, float]]:
    """Return constant-state segments that do not cross acquisition breaks."""
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
    """Join blocks with lag-safe row-mean seams and return data slices."""
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
# ## Step 3 — nested populations and selection diagnostics
#
# The cutoff is the lowest proxy value needed to reach the requested population
# size. Every neuron tied at that value is retained. The nestedness assertion is
# deliberate: a nominally stricter population must never gain a neuron that was
# absent from a more permissive population in the same proxy family.


# %%
@dataclass(frozen=True)
class PopulationSpec:
    """One fitted active-neuron population."""

    key: str
    label: str
    ranking_proxy: str
    nominal_fraction: float
    cutoff: float
    local_rows: np.ndarray


def top_fraction_with_ties(
    values: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, float, int]:
    """Retain a top fraction and every row tied at the boundary."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a nonempty finite vector")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    target = int(np.ceil(fraction * values.size))
    cutoff = float(np.sort(values)[-target])
    return values >= cutoff, cutoff, target


def make_population_specs(
    onset_rates: np.ndarray,
    bin_rates: np.ndarray,
) -> list[PopulationSpec]:
    """Create the primary population and two nested proxy families."""
    n_rows = onset_rates.size
    if bin_rates.shape != (n_rows,):
        raise ValueError("Proxy vectors must align")
    specs = [
        PopulationSpec(
            key="dataset_active",
            label="dataset active (100%)",
            ranking_proxy="dataset_nonzero_ROI",
            nominal_fraction=1.0,
            cutoff=np.nan,
            local_rows=np.arange(n_rows, dtype=np.int64),
        )
    ]
    for proxy_name, values, short_label in (
        ("positive_run_onset_rate", onset_rates, "onset"),
        ("positive_bin_rate", bin_rates, "positive-bin"),
    ):
        previous_mask = np.ones(n_rows, dtype=bool)
        for fraction in RETAINED_FRACTIONS:
            mask, cutoff, _target = top_fraction_with_ties(values, fraction)
            if np.any(mask & ~previous_mask):
                raise RuntimeError(f"{proxy_name} populations are not nested")
            previous_mask = mask
            percent = round(100 * fraction)
            specs.append(
                PopulationSpec(
                    key=f"{short_label.replace('-', '_')}_top_{percent:02d}",
                    label=f"{short_label} top {percent}%",
                    ranking_proxy=proxy_name,
                    nominal_fraction=fraction,
                    cutoff=cutoff,
                    local_rows=np.flatnonzero(mask),
                )
            )
    return specs


def row_hash(rows: np.ndarray) -> str:
    """Hash exact original ROI indices for checkpoint provenance."""
    canonical = np.ascontiguousarray(rows, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def block_hash(
    fold_a_blocks: list[tuple[int, int, float]],
    fold_b_blocks: list[tuple[int, int, float]],
) -> str:
    """Hash the ordered A/B allocation, including state codes."""
    values = []
    for fold_index, blocks in enumerate((fold_a_blocks, fold_b_blocks)):
        for start, stop, code in blocks:
            values.extend((fold_index, start, stop, int(round(code * 2))))
    canonical = np.ascontiguousarray(values, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


# %% [markdown]
# ## Step 4 — official fits, tie-aware fine recurrence, and coarse transfer
#
# Fine A/B recurrence uses reversal-invariant Spearman correlation and local
# neighborhood overlap with independent random ordering inside exact Rastermap
# ties. The adjusted local score subtracts finite-population chance overlap.
#
# For coarse transfer, source PCA axes and sorted cluster templates are evaluated
# on the held-out fold in both directions. The learned order is compared with:
#
# - the exact mean objective over uniformly random complete cluster orders;
# - an activity-ranked cluster order whose orientation is chosen on source data
#   only.
#
# The training replay assertion verifies that the transfer equations reconstruct
# the fitted model's own `cc`. It catches projection or normalization drift before
# held-out scores are interpreted.


# %%
@dataclass
class FitSummary:
    """Compact aligned output retained after an official Rastermap fit."""

    embedding: np.ndarray
    clusters: np.ndarray
    runtime_seconds: float


def fit_matrix(
    activity: np.ndarray,
    lag_frames: int,
) -> tuple[Rastermap, FitSummary]:
    """Fit official Rastermap to one prescreened population and fold."""
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
        random_state=FIT_SEED,
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


def tie_fraction(embedding: np.ndarray) -> tuple[int, float]:
    """Return occupied positions and fraction of neurons in tied positions."""
    _positions, counts = np.unique(embedding, return_counts=True)
    return counts.size, float(counts[counts > 1].sum() / embedding.size)


def fine_order_record(
    first: FitSummary,
    second: FitSummary,
    metric_seed: int,
) -> dict[str, float]:
    """Return reversal-invariant and tie-aware aligned A/B diagnostics."""
    neighborhood = min(NEIGHBORHOOD_SIZE, first.embedding.size - 1)
    local_raw = rmt.rank_neighborhood_overlap(
        first.embedding,
        second.embedding,
        neighborhood_size=neighborhood,
        tie_permutations=TIE_PERMUTATIONS,
        random_state=metric_seed,
    )
    chance = neighborhood / (first.embedding.size - 1)
    occupied_a, tied_a = tie_fraction(first.embedding)
    occupied_b, tied_b = tie_fraction(second.embedding)
    return {
        "abs_spearman": rmt.reversal_invariant_rank_correlation(
            first.embedding,
            second.embedding,
        ),
        "local_overlap_raw": local_raw,
        "local_overlap_adjusted": (local_raw - chance) / (1 - chance),
        "cluster_ari": adjusted_rand_score(first.clusters, second.clusters),
        "occupied_positions_a": occupied_a,
        "occupied_positions_b": occupied_b,
        "fraction_tied_a": tied_a,
        "fraction_tied_b": tied_b,
    }


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
    """Project held-out normalized activity through source PCA axes."""
    singular_values = np.asarray(source_model.sv, dtype=np.float32)
    source_left = np.asarray(source_model.Usv, dtype=np.float32) / singular_values
    return (target_normalized_activity.T @ source_left) / singular_values


def node_similarity_from_temporal_scores(
    source_model: Rastermap,
    temporal_scores: np.ndarray,
) -> np.ndarray:
    """Apply Rastermap's installed directed lag-similarity calculation."""
    return compute_cc_tdelay(
        temporal_scores,
        np.asarray(source_model.U_nodes, dtype=np.float32),
        time_lag_window=int(source_model.time_lag_window),
        symmetric=False,
    )


def verify_training_replay(model: Rastermap) -> None:
    """Assert that transfer projection exactly replays training similarity."""
    replay = node_similarity_from_temporal_scores(
        model,
        transferred_temporal_scores(model, model.X),
    )
    if not np.allclose(replay, model.cc, atol=2e-5, rtol=2e-5):
        raise RuntimeError("Transfer math did not replay training cc")


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
    weights = np.triu(target, k=1)
    ordered = similarity[np.ix_(order, order)]
    return float(np.sum(weights * ordered) / weights.sum())


def exact_random_order_expectation(node_similarity: np.ndarray) -> float:
    """Return the exact objective mean under a uniform complete order."""
    similarity = np.asarray(node_similarity, dtype=np.float64)
    n_clusters = similarity.shape[0]
    return float(
        (similarity.sum() - np.trace(similarity)) / (n_clusters * (n_clusters - 1))
    )


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


def coarse_transfer_record(
    source_model: Rastermap,
    target_model: Rastermap,
    source_positive_counts: np.ndarray,
) -> dict[str, float]:
    """Compare learned held-out order with exact-random and activity order."""
    temporal_scores = transferred_temporal_scores(source_model, target_model.X)
    transferred = node_similarity_from_temporal_scores(source_model, temporal_scores)
    identity = np.arange(source_model.U_nodes.shape[0], dtype=np.int64)
    activity_order = activity_cluster_order(source_model, source_positive_counts)
    return {
        "learned": directional_objective(transferred, source_model.BBt, identity),
        "activity": directional_objective(
            transferred,
            source_model.BBt,
            activity_order,
        ),
        "random_expectation": exact_random_order_expectation(transferred),
    }


# %% [markdown]
# ## Step 5 — checkpoint schema and exact reuse
#
# Selection tables are recomputed because they are inexpensive relative to the
# fits. Fit rows are reusable only when `checkpoint_matches` verifies every
# provenance field, including hashes of the exact original ROI rows and A/B
# block allocation, sampling rate, derived lag/block/guard frame counts, and the
# tie-metric random-seed rule. A partial or mismatched population is recomputed
# as one unit; models are never mixed across configurations.


# %%
def save_records(path: Path, records: list[dict[str, object]]) -> None:
    """Atomically write same-schema records as a restartable CSV checkpoint."""
    if not records:
        return
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, path)


def load_records(path: Path) -> list[dict[str, str]]:
    """Load a checkpoint table if it exists."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def numeric(record: dict[str, object], field: str) -> float:
    """Read a numeric field from fresh or CSV-loaded records."""
    return float(record[field])


def configuration_record(
    source_path: Path,
    spec: PopulationSpec,
    selected_original_rows: np.ndarray,
    allocation_hash: str,
    qualification: QualificationSummary,
    fs: float,
    lag_frames: int,
    block_frames: int,
    guard_frames: int,
    split_seed: int,
    tie_metric_seed: int,
) -> dict[str, object]:
    """Return source, selection, allocation, and model provenance."""
    source_stat = source_path.stat()
    return {
        "analysis_schema": ANALYSIS_SCHEMA,
        "source_size_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "rastermap_version": RASTERMAP_VERSION,
        "population_key": spec.key,
        "ranking_proxy": spec.ranking_proxy,
        "nominal_retained_fraction": spec.nominal_fraction,
        "proxy_cutoff_per_second": spec.cutoff,
        "selected_neurons": selected_original_rows.size,
        "selected_row_sha256": row_hash(selected_original_rows),
        "qualification_window_frames": (
            qualification.analyzed_frames // sum(qualification.windows_by_state)
        ),
        "qualification_state0_windows": qualification.windows_by_state[0],
        "qualification_state1_windows": qualification.windows_by_state[1],
        "qualification_analyzed_frames": qualification.analyzed_frames,
        "fs": fs,
        "lag_frames": lag_frames,
        "block_frames": block_frames,
        "guard_frames": guard_frames,
        "split_seed_base": SPLIT_SEED_BASE,
        "split_seed": split_seed,
        "block_allocation_sha256": allocation_hash,
        "block_seconds": BLOCK_SECONDS,
        "block_edge_guard_seconds": BLOCK_EDGE_GUARD_SECONDS,
        "paper_lag_seconds": PAPER_LAG_SECONDS,
        "fit_seed": FIT_SEED,
        "n_clusters": N_CLUSTERS,
        "n_pcs": N_PCS,
        "locality": LOCALITY,
        "mean_time": int(MEAN_TIME),
        "neighborhood_size": NEIGHBORHOOD_SIZE,
        "tie_permutations": TIE_PERMUTATIONS,
        "tie_metric_seed_base": TIE_METRIC_SEED_BASE,
        "tie_metric_seed_scheme": TIE_METRIC_SEED_SCHEME,
        "tie_metric_seed": tie_metric_seed,
    }


def checkpoint_matches(
    record: dict[str, object],
    expected: dict[str, object],
) -> bool:
    """Return whether one checkpoint row exactly matches current provenance."""
    exact_fields = (
        "analysis_schema",
        "source_size_bytes",
        "source_mtime_ns",
        "rastermap_version",
        "population_key",
        "ranking_proxy",
        "selected_neurons",
        "selected_row_sha256",
        "qualification_window_frames",
        "qualification_state0_windows",
        "qualification_state1_windows",
        "qualification_analyzed_frames",
        "lag_frames",
        "block_frames",
        "guard_frames",
        "split_seed_base",
        "split_seed",
        "block_allocation_sha256",
        "fit_seed",
        "n_clusters",
        "n_pcs",
        "mean_time",
        "neighborhood_size",
        "tie_permutations",
        "tie_metric_seed_base",
        "tie_metric_seed_scheme",
        "tie_metric_seed",
    )
    float_fields = (
        "nominal_retained_fraction",
        "proxy_cutoff_per_second",
        "block_seconds",
        "block_edge_guard_seconds",
        "paper_lag_seconds",
        "locality",
        "fs",
    )
    try:
        if any(str(record[field]) != str(expected[field]) for field in exact_fields):
            return False
        return all(
            (np.isnan(float(record[field])) and np.isnan(float(expected[field])))
            or np.isclose(float(record[field]), float(expected[field]))
            for field in float_fields
        )
    except (KeyError, TypeError, ValueError):
        return False


# %% [markdown]
# ## Step 6 — audit all recordings and optionally run the official fits
#
# The selection audit always verifies `nonzero_ROI` before making populations.
# One common A/B validity mask is then fixed on the primary population and used
# by every sensitivity arm. This prevents a proxy-specific change in numerical
# eligibility from masquerading as an activity-definition effect.
#
# The mask remains conditional/transductive: `nonzero_ROI` and common validity
# inspect both states/folds. This tutorial tests robustness of a descriptive
# embedding; it is not a prospective prediction analysis.

# %%
selection_records: list[dict[str, object]] = []
overlap_records: list[dict[str, object]] = []
stored_fit_records: list[dict[str, object]] = list(load_records(FIT_PATH))
stored_fit_records = [
    record
    for record in stored_fit_records
    if record.get("recording") in RECORDINGS
    and str(record.get("analysis_schema")) == str(ANALYSIS_SCHEMA)
]
fit_records: list[dict[str, object]] = []

for recording_index, recording_name in enumerate(RECORDINGS):
    print(
        f"\n[{recording_index + 1}/{len(RECORDINGS)}] {recording_name}",
        flush=True,
    )
    condition, mouse, allowed_codes, code_labels = condition_and_states(recording_name)
    (
        source_path,
        activity,
        state,
        nonzero_roi,
        boundary_ind,
        used_frame,
    ) = load_analysis_arrays(recording_name)
    n_recorded = activity.shape[0]
    fs = dataio.FS_HZ
    qualification_window = QUALIFICATION_WINDOW_FRAMES[condition]
    qualification = qualification_proxy_counts(
        activity,
        used_frame,
        boundary_ind,
        qualification_window,
    )
    if not np.array_equal(
        qualification.reconstructed_nonzero_roi,
        nonzero_roi,
    ):
        mismatch = int(
            np.count_nonzero(qualification.reconstructed_nonzero_roi != nonzero_roi)
        )
        raise RuntimeError(
            f"{recording_name}: qualification windows disagree with "
            f"nonzero_ROI for {mismatch} neurons"
        )

    full_valid = rmt.valid_activity_rows(activity)
    primary_mask = full_valid & nonzero_roi
    primary_original_rows = np.flatnonzero(primary_mask)
    primary_activity = np.ascontiguousarray(
        activity[primary_original_rows],
        dtype=np.float32,
    )

    lag_frames = round(PAPER_LAG_SECONDS * fs)
    block_frames = round(BLOCK_SECONDS * fs)
    guard_frames = round(BLOCK_EDGE_GUARD_SECONDS * fs)
    segments = primary_segments(state, boundary_ind, allowed_codes)
    blocks = guarded_blocks(segments, block_frames, guard_frames)
    split_seed = SPLIT_SEED_BASE + recording_index
    tie_metric_seed = TIE_METRIC_SEED_BASE + recording_index
    fold_a_blocks, fold_b_blocks, blocks_per_fold = matched_fold_blocks(
        blocks,
        allowed_codes,
        seed=split_seed,
    )
    allocation_hash = block_hash(fold_a_blocks, fold_b_blocks)

    common_valid = valid_rows_in_blocks(primary_activity, fold_a_blocks)
    common_valid &= valid_rows_in_blocks(primary_activity, fold_b_blocks)
    fit_base_original_rows = primary_original_rows[common_valid]
    del activity
    gc.collect()
    fit_base_activity = np.ascontiguousarray(
        primary_activity[common_valid],
        dtype=np.float32,
    )
    del primary_activity
    gc.collect()
    onset_rates = (
        qualification.positive_run_onsets[fit_base_original_rows]
        * fs
        / qualification.analyzed_frames
    )
    bin_rates = (
        qualification.positive_bins[fit_base_original_rows]
        * fs
        / qualification.analyzed_frames
    )
    population_specs = make_population_specs(onset_rates, bin_rates)

    # Joined fold matrices are allocated lazily on the first missing fit. With
    # RUN_FULL_ANALYSIS=False makes the audit perform only loading,
    # qualification-window counting, and row-selection checks.
    fold_a_matrix = None
    fold_b_matrix = None
    fold_a_counts = None
    fold_b_counts = None
    fold_a_slices = None
    fold_b_slices = None

    spec_lookup = {spec.key: spec for spec in population_specs}
    proxy_rho = float(spearmanr(onset_rates, bin_rates).statistic)
    bins_per_onset = qualification.positive_bins[fit_base_original_rows] / np.maximum(
        qualification.positive_run_onsets[fit_base_original_rows],
        1,
    )
    for spec in population_specs:
        selected_original_rows = fit_base_original_rows[spec.local_rows]
        target_neurons = int(
            np.ceil(spec.nominal_fraction * fit_base_activity.shape[0])
        )
        selection_records.append(
            {
                "recording": recording_name,
                "condition": condition,
                "mouse": mouse,
                "population_key": spec.key,
                "population_label": spec.label,
                "ranking_proxy": spec.ranking_proxy,
                "nominal_retained_fraction": spec.nominal_fraction,
                "proxy_cutoff_per_second": spec.cutoff,
                "recorded_neurons": n_recorded,
                "dataset_nonzero_roi_neurons": int(nonzero_roi.sum()),
                "finite_nonconstant_dataset_active_neurons": primary_original_rows.size,
                "common_fold_eligible_neurons": fit_base_original_rows.size,
                "target_neurons_before_ties": target_neurons,
                "actual_selected_neurons": spec.local_rows.size,
                "actual_retained_fraction": (
                    spec.local_rows.size / fit_base_original_rows.size
                ),
                "qualification_window_frames": qualification_window,
                "qualification_state0_windows": qualification.windows_by_state[0],
                "qualification_state1_windows": qualification.windows_by_state[1],
                "qualification_analyzed_seconds": (qualification.analyzed_frames / fs),
                "onset_bin_proxy_spearman": proxy_rho,
                "median_positive_bins_per_onset": float(np.median(bins_per_onset)),
                "q90_positive_bins_per_onset": float(np.quantile(bins_per_onset, 0.90)),
                "selected_row_sha256": row_hash(selected_original_rows),
            }
        )

    for fraction in RETAINED_FRACTIONS:
        percent = round(100 * fraction)
        onset_spec = spec_lookup[f"onset_top_{percent:02d}"]
        bin_spec = spec_lookup[f"positive_bin_top_{percent:02d}"]
        onset_set = set(onset_spec.local_rows.tolist())
        bin_set = set(bin_spec.local_rows.tolist())
        intersection = len(onset_set & bin_set)
        union = len(onset_set | bin_set)
        overlap_records.append(
            {
                "recording": recording_name,
                "condition": condition,
                "mouse": mouse,
                "nominal_retained_fraction": fraction,
                "onset_selected_neurons": len(onset_set),
                "positive_bin_selected_neurons": len(bin_set),
                "intersection_neurons": intersection,
                "union_neurons": union,
                "jaccard": intersection / union,
                "smaller_set_recovery": intersection
                / min(len(onset_set), len(bin_set)),
                "onset_bin_proxy_spearman": proxy_rho,
            }
        )

    print(
        f"  primary population: {fit_base_original_rows.size:,} fit-eligible / "
        f"{int(nonzero_roi.sum()):,} dataset-active; proxy rank ρ={proxy_rho:.3f}",
        flush=True,
    )

    for spec in population_specs:
        selected_original_rows = fit_base_original_rows[spec.local_rows]
        expected = configuration_record(
            source_path,
            spec,
            selected_original_rows,
            allocation_hash,
            qualification,
            fs,
            lag_frames,
            block_frames,
            guard_frames,
            split_seed,
            tie_metric_seed,
        )
        existing = [
            record
            for record in stored_fit_records
            if record.get("recording") == recording_name
            and record.get("population_key") == spec.key
            and checkpoint_matches(record, expected)
        ]
        if REUSE_COMPLETE_RESULTS and len(existing) == 1:
            fit_records.append(existing[0])
            print(f"  {spec.key}: reusing exact checkpoint", flush=True)
            continue
        if not RUN_FULL_ANALYSIS:
            print(
                f"  {spec.key}: pending {spec.local_rows.size:,}-neuron A/B fits",
                flush=True,
            )
            continue

        if spec.local_rows.size < MIN_NEURONS_FOR_FIT:
            raise RuntimeError(
                f"{recording_name}/{spec.key}: only {spec.local_rows.size} neurons"
            )
        if fold_a_matrix is None:
            fold_a_matrix, fold_a_counts, fold_a_slices = matrix_from_blocks(
                fit_base_activity,
                fold_a_blocks,
                separator_frames=lag_frames,
            )
            fold_b_matrix, fold_b_counts, fold_b_slices = matrix_from_blocks(
                fit_base_activity,
                fold_b_blocks,
                separator_frames=lag_frames,
            )
        if (
            fold_b_matrix is None
            or fold_a_counts is None
            or fold_b_counts is None
            or fold_a_slices is None
            or fold_b_slices is None
        ):
            raise RuntimeError("Fold matrices were not initialized together")
        pop_a = np.ascontiguousarray(fold_a_matrix[spec.local_rows], dtype=np.float32)
        pop_b = np.ascontiguousarray(fold_b_matrix[spec.local_rows], dtype=np.float32)
        model_a, summary_a = fit_matrix(pop_a, lag_frames)
        model_b, summary_b = fit_matrix(pop_b, lag_frames)
        verify_normalized_separators(model_a, fold_a_slices)
        verify_normalized_separators(model_b, fold_b_slices)
        verify_training_replay(model_a)
        verify_training_replay(model_b)

        fine = fine_order_record(
            summary_a,
            summary_b,
            metric_seed=tie_metric_seed,
        )
        transfer_a_to_b = coarse_transfer_record(
            model_a,
            model_b,
            fold_a_counts[spec.local_rows],
        )
        transfer_b_to_a = coarse_transfer_record(
            model_b,
            model_a,
            fold_b_counts[spec.local_rows],
        )
        fit_record: dict[str, object] = {
            "recording": recording_name,
            "condition": condition,
            "mouse": mouse,
            **expected,
            "population_label": spec.label,
            "recorded_neurons": n_recorded,
            "dataset_nonzero_roi_neurons": int(nonzero_roi.sum()),
            "primary_finite_nonconstant_neurons": primary_original_rows.size,
            "common_fold_eligible_neurons": fit_base_original_rows.size,
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
            "fold_a_occupied_embedding_positions": fine["occupied_positions_a"],
            "fold_b_occupied_embedding_positions": fine["occupied_positions_b"],
            "fold_a_fraction_neurons_in_ties": fine["fraction_tied_a"],
            "fold_b_fraction_neurons_in_ties": fine["fraction_tied_b"],
            "fold_a_runtime_seconds": summary_a.runtime_seconds,
            "fold_b_runtime_seconds": summary_b.runtime_seconds,
        }
        for prefix, transfer in (
            ("objective_a_to_b", transfer_a_to_b),
            ("objective_b_to_a", transfer_b_to_a),
        ):
            for field, value in transfer.items():
                fit_record[f"{prefix}_{field}"] = value
        for comparator in ("random_expectation", "activity"):
            fit_record[f"objective_reciprocal_learned_minus_{comparator}"] = float(
                np.mean(
                    [
                        transfer_a_to_b["learned"] - transfer_a_to_b[comparator],
                        transfer_b_to_a["learned"] - transfer_b_to_a[comparator],
                    ]
                )
            )
        fit_records.append(fit_record)
        stored_fit_records = [
            record
            for record in stored_fit_records
            if not (
                record.get("recording") == recording_name
                and record.get("population_key") == spec.key
            )
        ]
        stored_fit_records.append(fit_record)
        save_records(FIT_PATH, stored_fit_records)
        print(
            f"  {spec.key}: |ρ|={fine['abs_spearman']:.3f}, "
            f"local={fine['local_overlap_adjusted']:.3f}, "
            "learned−random="
            f"{fit_record['objective_reciprocal_learned_minus_random_expectation']:.4f}",
            flush=True,
        )
        del model_a, model_b, pop_a, pop_b
        gc.collect()

    del (
        state,
        nonzero_roi,
        used_frame,
        fit_base_activity,
        fold_a_matrix,
        fold_b_matrix,
        fold_a_counts,
        fold_b_counts,
    )
    gc.collect()

save_records(SELECTION_PATH, selection_records)
save_records(OVERLAP_PATH, overlap_records)
print("saved ->", SELECTION_PATH)
print("saved ->", OVERLAP_PATH)
if RUN_FULL_ANALYSIS:
    print("saved ->", FIT_PATH)
else:
    print(
        "Official fits were not started. Set RUN_FULL_ANALYSIS=True and rerun "
        "Step 6 when ready."
    )


# %% [markdown]
# ## Step 7 — selection and proxy-dependence figure
#
# The first figure is available before any Rastermap fit. It shows the actual
# retained fractions after keeping cutoff ties, agreement between the two proxy
# populations, and how closely their within-recording ranks correspond.

# %%
recording_x = np.arange(len(RECORDINGS))
recording_labels = [name.replace("mouse", "m") for name in RECORDINGS]
condition_colors = [
    "tab:blue" if name.endswith("sleep") else "tab:orange" for name in RECORDINGS
]
proxy_styles = {
    "positive_run_onset_rate": ("o-", "positive-run onset rank"),
    "positive_bin_rate": ("s--", "positive-bin rank"),
}

selection_figure, selection_axes = plt.subplots(
    2,
    2,
    figsize=(15, 9),
    constrained_layout=True,
)
for proxy, (style, label) in proxy_styles.items():
    for recording_index, recording_name in enumerate(RECORDINGS):
        records = [
            record
            for record in selection_records
            if record["recording"] == recording_name
            and record["ranking_proxy"] == proxy
        ]
        records.sort(key=lambda record: float(record["nominal_retained_fraction"]))
        selection_axes[0, 0].plot(
            [float(record["nominal_retained_fraction"]) for record in records],
            [float(record["actual_retained_fraction"]) for record in records],
            style,
            color=condition_colors[recording_index],
            alpha=0.35,
            ms=3,
        )
selection_axes[0, 0].plot([0.45, 0.95], [0.45, 0.95], color="black", lw=0.8)
selection_axes[0, 0].set(
    xlabel="nominal retained fraction",
    ylabel="actual fraction after keeping ties",
    title="Cutoff ties can enlarge nested populations",
)

for fraction in RETAINED_FRACTIONS:
    values = [
        float(record["jaccard"])
        for record in overlap_records
        if np.isclose(float(record["nominal_retained_fraction"]), fraction)
    ]
    selection_axes[0, 1].scatter(
        recording_x,
        values,
        label=f"top {round(100 * fraction)}%",
        s=28,
    )
selection_axes[0, 1].set(
    ylabel="onset/bin population Jaccard",
    title="Proxy choice changes neuron membership",
)
selection_axes[0, 1].legend(frameon=False, fontsize=8)

primary_selection = [
    next(
        record
        for record in selection_records
        if record["recording"] == recording_name
        and record["population_key"] == "dataset_active"
    )
    for recording_name in RECORDINGS
]
selection_axes[1, 0].bar(
    recording_x,
    [
        float(record["common_fold_eligible_neurons"])
        / float(record["recorded_neurons"])
        for record in primary_selection
    ],
    color=condition_colors,
)
selection_axes[1, 0].set(
    ylabel="fit-eligible dataset-active / recorded",
    title="Primary dataset-active population",
)

selection_axes[1, 1].scatter(
    recording_x,
    [float(record["onset_bin_proxy_spearman"]) for record in primary_selection],
    color=condition_colors,
    edgecolor="black",
    linewidth=0.4,
)
selection_axes[1, 1].set_ylim(0, 1.02)
selection_axes[1, 1].set(
    ylabel="within-recording Spearman ρ",
    title="Positive-bin and onset ranks are related, not identical",
)

for axis in selection_axes.ravel():
    axis.grid(axis="y", color="0.9", lw=0.6)
for axis in selection_axes.flat[1:]:
    axis.set_xticks(recording_x, recording_labels, rotation=45, ha="right")
selection_figure.suptitle(
    "Dataset qualification-window activity proxies · neither is a firing rate"
)
selection_figure_path = FIG_DIR / "08_rastermap_01_active_selection_sensitivity.png"
selection_figure.savefig(selection_figure_path, dpi=150, bbox_inches="tight")
print("saved ->", selection_figure_path)
if SHOW_FIGURES:
    plt.show()


# %% [markdown]
# ## Step 8 — completed-fit summaries and figure
#
# The fit figure is made only when all 70 population rows are available from new
# fits or exact checkpoints. Each line begins at the common dataset-active fit
# and follows one proxy family toward stricter populations. A robust conclusion
# should not depend on one proxy or one retained fraction.

# %%
expected_population_keys = {
    spec.key for spec in make_population_specs(np.ones(8), np.ones(8))
}
expected_fit_keys = {
    (recording_name, population_key)
    for recording_name in RECORDINGS
    for population_key in expected_population_keys
}
actual_fit_keys = {
    (str(record["recording"]), str(record["population_key"])) for record in fit_records
}
expected_fit_rows = len(expected_fit_keys)
complete_fit_table = (
    len(fit_records) == expected_fit_rows and actual_fit_keys == expected_fit_keys
)

if complete_fit_table:
    metric_fields = (
        "fold_abs_spearman",
        "fold_local_overlap_adjusted",
        "objective_reciprocal_learned_minus_random_expectation",
        "objective_reciprocal_learned_minus_activity",
    )
    summary_records: list[dict[str, object]] = []
    population_keys = [
        "dataset_active",
        "onset_top_90",
        "onset_top_75",
        "onset_top_50",
        "positive_bin_top_90",
        "positive_bin_top_75",
        "positive_bin_top_50",
    ]
    for condition in ("sleep", "anesthesia"):
        for population_key in population_keys:
            population_rows = [
                record
                for record in fit_records
                if record["condition"] == condition
                and record["population_key"] == population_key
            ]
            mouse_values = {}
            for mouse in sorted({str(record["mouse"]) for record in population_rows}):
                sessions = [
                    record for record in population_rows if record["mouse"] == mouse
                ]
                mouse_values[mouse] = {
                    field: float(
                        np.mean([numeric(record, field) for record in sessions])
                    )
                    for field in metric_fields
                }
            first = population_rows[0]
            summary: dict[str, object] = {
                "condition": condition,
                "population_key": population_key,
                "population_label": first["population_label"],
                "ranking_proxy": first["ranking_proxy"],
                "nominal_retained_fraction": numeric(
                    first,
                    "nominal_retained_fraction",
                ),
                "n_mice": len(mouse_values),
            }
            for field in metric_fields:
                values = np.array(
                    [mouse_value[field] for mouse_value in mouse_values.values()]
                )
                summary[f"{field}_mouse_median"] = float(np.median(values))
                summary[f"{field}_mouse_minimum"] = float(values.min())
                summary[f"{field}_mouse_maximum"] = float(values.max())
            summary_records.append(summary)
    save_records(SUMMARY_PATH, summary_records)
    print("saved ->", SUMMARY_PATH)

    fit_figure, fit_axes = plt.subplots(
        2,
        2,
        figsize=(14, 9),
        constrained_layout=True,
    )
    figure_metrics = (
        ("fold_abs_spearman", "A/B |Spearman ρ|", "Fine global order"),
        (
            "fold_local_overlap_adjusted",
            "adjusted local overlap",
            "Tie-aware local order",
        ),
        (
            "objective_reciprocal_learned_minus_random_expectation",
            "learned − exact random mean",
            "Coarse transfer beyond random order",
        ),
        (
            "objective_reciprocal_learned_minus_activity",
            "learned − activity-ranked",
            "Coarse transfer beyond activity rank",
        ),
    )
    for axis, (field, ylabel, title) in zip(
        fit_axes.ravel(),
        figure_metrics,
        strict=True,
    ):
        for recording_index, recording_name in enumerate(RECORDINGS):
            baseline = next(
                record
                for record in fit_records
                if record["recording"] == recording_name
                and record["population_key"] == "dataset_active"
            )
            for proxy, (style, proxy_label) in proxy_styles.items():
                family = [
                    record
                    for record in fit_records
                    if record["recording"] == recording_name
                    and record["ranking_proxy"] == proxy
                ]
                family.append(baseline)
                family.sort(
                    key=lambda record: numeric(record, "nominal_retained_fraction")
                )
                axis.plot(
                    [numeric(record, "nominal_retained_fraction") for record in family],
                    [numeric(record, field) for record in family],
                    style,
                    color=condition_colors[recording_index],
                    alpha=0.32,
                    ms=3,
                )
        axis.axhline(0, color="black", lw=0.7)
        axis.set(xlabel="nominal retained fraction", ylabel=ylabel, title=title)
        axis.grid(axis="y", color="0.9", lw=0.6)
    fit_axes[0, 0].plot([], [], "o-", color="0.4", label="onset-ranked")
    fit_axes[0, 0].plot([], [], "s--", color="0.4", label="positive-bin-ranked")
    fit_axes[0, 0].plot([], [], "-", color="tab:blue", label="sleep")
    fit_axes[0, 0].plot([], [], "-", color="tab:orange", label="anesthesia")
    fit_axes[0, 0].legend(frameon=False, fontsize=8)
    fit_figure.suptitle(
        "Rastermap sensitivity to active-neuron definition · one fixed A/B split"
    )
    fit_figure_path = FIG_DIR / "08_rastermap_02_active_order_sensitivity.png"
    fit_figure.savefig(fit_figure_path, dpi=150, bbox_inches="tight")
    print("saved ->", fit_figure_path)
    if SHOW_FIGURES:
        plt.show()
else:
    print(
        f"Fit summary deferred: {len(fit_records)}/{expected_fit_rows} exact "
        "population checkpoints are currently available."
    )


# %% [markdown]
# ## Step 9 — interpretation guardrails
#
# - `nonzero_ROI` is the primary dataset-active population because it is exactly
#   reconstructable from the publication's state-specific qualification windows.
#   It is not the unpublished calcium-specific threshold used before Rastermap's
#   Figure 3.
# - Positive-run onset rate and positive-bin rate are numerical support proxies.
#   An onset can merge adjacent inferred events; a bin count can count several
#   adjacent positive samples. Neither should be called a firing rate.
# - Stable metrics across both proxy families and all retained fractions support
#   robustness to low-activity neurons. Divergence between families is evidence
#   that the active-neuron definition remains a material analysis choice.
# - Fine recurrence and coarse held-out transfer answer different questions. A
#   stable coarse objective does not validate a unique single-neuron order.
# - Only per-population A/B metrics are checkpointed. Embeddings are not cached
#   for a direct dataset-active-versus-refit order comparison, so this tutorial
#   quantifies recurrence/transfer sensitivity rather than baseline order
#   displacement. That remains an explicit follow-up analysis.
# - Whole-session `nonzero_ROI`, common A/B validity, within-recording ranking,
#   and target-side normalization make this a conditional/transductive audit.
# - REM, quiet awake, short state bouts, and block tails do not enter the fits,
#   although they remain available for the complete-sequence visualizations in
#   tutorial 02.

# %%
print("\nActive-neuron sensitivity tutorial is configured.")
print(f"  expected official fits: {expected_fit_rows * 2}")
print("  estimated full-run time: approximately 10–15 minutes on this workstation")
print("  heavy fits enabled:", RUN_FULL_ANALYSIS)
