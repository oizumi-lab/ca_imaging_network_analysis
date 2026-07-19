"""Small, testable helpers for applying Rastermap to complete recordings.

Rastermap itself is provided by the official ``rastermap`` package.  This
module only adapts that API to this dataset: it validates neuron rows, maps the
one-dimensional ordering back to original ROI indices, and caches the compact
display result.  Selection is always explicit: callers may fit all usable rows
or pass a prespecified active-neuron population.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import version
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import rankdata


CACHE_SCHEMA_VERSION = 3


@dataclass
class RastermapResult:
    """Compact output retained after a Rastermap fit to explicit ROI rows.

    ``isort`` contains original zero-based ROI row indices. ``embedding`` is
    aligned to every recorded ROI; a NaN denotes either a nonselected ROI or a
    mathematically unusable non-finite/constant trace. ``X_embedding`` is the
    paper-style normalized display matrix, in which adjacent Rastermap-ordered
    neurons are averaged into superneurons rather than randomly selected.
    """

    X_embedding: np.ndarray
    embedding: np.ndarray
    isort: np.ndarray
    valid_rows: np.ndarray
    runtime_seconds: float


def installed_rastermap_version() -> str:
    """Return the installed distribution version without relying on package globals."""
    return version("rastermap")


def valid_activity_rows(
    activity: np.ndarray,
    chunk_frames: int = 2048,
) -> np.ndarray:
    """Return rows that are finite and non-constant over the complete session.

    Rastermap z-scores each neuron, so a constant row has zero standard
    deviation and cannot be embedded.  The scan is chunked along time to avoid
    allocating a second full neuron-by-frame boolean matrix.
    """
    activity = np.asarray(activity)
    if activity.ndim != 2:
        raise ValueError("activity must be a neuron-by-frame matrix")
    n_neurons, n_frames = activity.shape
    if n_neurons == 0 or n_frames == 0:
        raise ValueError("activity must contain at least one neuron and frame")
    if isinstance(chunk_frames, (bool, np.bool_)) or not isinstance(
        chunk_frames, (int, np.integer)
    ):
        raise TypeError("chunk_frames must be an integer")
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")

    finite = np.ones(n_neurons, dtype=bool)
    row_min = np.full(n_neurons, np.inf, dtype=np.float64)
    row_max = np.full(n_neurons, -np.inf, dtype=np.float64)
    for start in range(0, n_frames, int(chunk_frames)):
        chunk = activity[:, start : start + int(chunk_frames)]
        finite &= np.all(np.isfinite(chunk), axis=1)
        # NaNs propagate through min/max and therefore also fail the final
        # comparison. The explicit finite mask additionally catches infinities.
        row_min = np.minimum(row_min, np.min(chunk, axis=1))
        row_max = np.maximum(row_max, np.max(chunk, axis=1))
    return finite & (row_max > row_min)


def _validate_activity_rate_inputs(
    activity: np.ndarray,
    fs: float,
    chunk_frames: int,
) -> tuple[np.ndarray, float, int]:
    """Validate shared inputs for the chunked deconvolution-rate helpers."""
    activity = np.asarray(activity)
    if activity.ndim != 2:
        raise ValueError("activity must be a neuron-by-frame matrix")
    if activity.shape[0] == 0 or activity.shape[1] == 0:
        raise ValueError("activity must contain at least one neuron and frame")
    if isinstance(fs, (bool, np.bool_)) or not isinstance(fs, Real):
        raise TypeError("fs must be a real number")
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be finite and positive")
    if isinstance(chunk_frames, (bool, np.bool_)) or not isinstance(
        chunk_frames,
        (int, np.integer),
    ):
        raise TypeError("chunk_frames must be an integer")
    chunk_frames = int(chunk_frames)
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    return activity, fs, chunk_frames


def positive_deconvolution_bin_counts(
    activity: np.ndarray,
    chunk_frames: int = 2048,
) -> np.ndarray:
    """Count positive deconvolution bins for every neuron in bounded memory."""
    activity, _, chunk_frames = _validate_activity_rate_inputs(
        activity,
        1.0,
        chunk_frames,
    )
    positive_counts = np.zeros(activity.shape[0], dtype=np.int64)
    for start in range(0, activity.shape[1], chunk_frames):
        positive_counts += np.count_nonzero(
            activity[:, start : start + chunk_frames] > 0,
            axis=1,
        )
    return positive_counts


def positive_deconvolution_bin_rates(
    activity: np.ndarray,
    fs: float,
    chunk_frames: int = 2048,
) -> np.ndarray:
    """Count positive deconvolution bins per second for every neuron.

    A positive OASIS bin is an activity proxy, not a calibrated physiological
    spike count.  The time scan is chunked so the function never constructs a
    second full neuron-by-frame boolean matrix.  Non-finite or constant traces
    are deliberately not rejected here; combine this summary with
    :func:`valid_activity_rows`, as :func:`active_deconvolution_rows` does.
    """
    activity, fs, chunk_frames = _validate_activity_rate_inputs(
        activity,
        fs,
        chunk_frames,
    )
    positive_counts = positive_deconvolution_bin_counts(
        activity,
        chunk_frames=chunk_frames,
    )
    return positive_counts.astype(np.float64) * fs / activity.shape[1]


def positive_deconvolution_onset_rates(
    activity: np.ndarray,
    fs: float,
    chunk_frames: int = 2048,
) -> np.ndarray:
    """Count transitions into positive deconvolution activity per second.

    Consecutive positive bins count as one onset.  State is carried between
    chunks, preventing an event that crosses a chunk boundary from being
    counted twice.  This stricter proxy is useful as a sensitivity analysis for
    :func:`positive_deconvolution_bin_rates`.
    """
    activity, fs, chunk_frames = _validate_activity_rate_inputs(
        activity,
        fs,
        chunk_frames,
    )
    n_neurons, n_frames = activity.shape
    onset_counts = np.zeros(n_neurons, dtype=np.int64)
    previous_positive = np.zeros(n_neurons, dtype=bool)
    for start in range(0, n_frames, chunk_frames):
        positive = activity[:, start : start + chunk_frames] > 0
        onset_counts += positive[:, 0] & ~previous_positive
        if positive.shape[1] > 1:
            onset_counts += np.count_nonzero(
                positive[:, 1:] & ~positive[:, :-1],
                axis=1,
            )
        previous_positive = positive[:, -1]
    return onset_counts.astype(np.float64) * fs / n_frames


def active_deconvolution_rows(
    activity: np.ndarray,
    fs: float,
    min_positive_bin_rate_hz: float = 0.1,
    chunk_frames: int = 2048,
) -> np.ndarray:
    """Return rows suitable for Rastermap and active above a chosen rate.

    The returned Boolean mask requires both a finite, non-constant full trace
    and a positive-bin rate greater than or equal to the threshold.  The
    threshold is inclusive and refers specifically to positive deconvolution
    bins per second, not to a physiologically calibrated firing rate.
    """
    if isinstance(min_positive_bin_rate_hz, (bool, np.bool_)) or not isinstance(
        min_positive_bin_rate_hz, Real
    ):
        raise TypeError("min_positive_bin_rate_hz must be a real number")
    min_positive_bin_rate_hz = float(min_positive_bin_rate_hz)
    if not np.isfinite(min_positive_bin_rate_hz) or min_positive_bin_rate_hz < 0:
        raise ValueError("min_positive_bin_rate_hz must be finite and non-negative")

    rates = positive_deconvolution_bin_rates(
        activity,
        fs,
        chunk_frames=chunk_frames,
    )
    valid = valid_activity_rows(activity, chunk_frames=chunk_frames)
    return valid & (rates >= min_positive_bin_rate_hz)


def active_deconvolution_count_rows(
    activity: np.ndarray,
    min_positive_bins: int = 50,
    chunk_frames: int = 2048,
) -> np.ndarray:
    """Return finite, nonconstant rows with a minimum number of positive bins.

    This is a numerical-support criterion for cross-session visualization, not
    a physiological firing-rate threshold.  Unlike a rate, the same count has
    different per-second equivalents for recordings of different duration; the
    caller should report that equivalent explicitly.
    """
    if isinstance(min_positive_bins, (bool, np.bool_)) or not isinstance(
        min_positive_bins,
        (int, np.integer),
    ):
        raise TypeError("min_positive_bins must be an integer")
    min_positive_bins = int(min_positive_bins)
    if min_positive_bins < 0:
        raise ValueError("min_positive_bins must be non-negative")
    counts = positive_deconvolution_bin_counts(
        activity,
        chunk_frames=chunk_frames,
    )
    valid = valid_activity_rows(activity, chunk_frames=chunk_frames)
    return valid & (counts >= min_positive_bins)


def _common_finite_embeddings(
    reference_embedding: np.ndarray,
    comparison_embedding: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned finite one-dimensional embedding values."""
    reference = np.asarray(reference_embedding)
    comparison = np.asarray(comparison_embedding)
    if reference.ndim != 1 or comparison.ndim != 1:
        raise ValueError("embeddings must be one-dimensional")
    if reference.shape != comparison.shape:
        raise ValueError("embeddings must have the same shape")
    common = np.isfinite(reference) & np.isfinite(comparison)
    if np.count_nonzero(common) < 2:
        raise ValueError("embeddings must share at least two finite neurons")
    reference = reference[common].astype(np.float64, copy=False)
    comparison = comparison[common].astype(np.float64, copy=False)
    if np.unique(reference).size < 2 or np.unique(comparison).size < 2:
        raise ValueError("each embedding must contain at least two distinct values")
    return reference, comparison


def reversal_invariant_rank_correlation(
    reference_embedding: np.ndarray,
    comparison_embedding: np.ndarray,
) -> float:
    """Return absolute Spearman correlation for two aligned embeddings.

    A one-dimensional Rastermap embedding has no intrinsic direction, so an
    exactly reversed order is equivalent and scores 1.  Rows that are non-finite
    in either embedding are omitted pairwise.
    """
    reference, comparison = _common_finite_embeddings(
        reference_embedding,
        comparison_embedding,
    )
    reference_rank = rankdata(reference, method="average")
    comparison_rank = rankdata(comparison, method="average")
    reference_rank -= reference_rank.mean()
    comparison_rank -= comparison_rank.mean()
    denominator = np.linalg.norm(reference_rank) * np.linalg.norm(comparison_rank)
    if denominator == 0:
        raise ValueError("rank correlation is undefined for a constant embedding")
    correlation = float(np.dot(reference_rank, comparison_rank) / denominator)
    return float(np.clip(abs(correlation), 0.0, 1.0))


def _order_neighborhoods(order: np.ndarray, neighborhood_size: int) -> list[set[int]]:
    """Return a fixed-size rank neighborhood for each row in ``order``."""
    n_rows = order.size
    positions = np.empty(n_rows, dtype=np.int64)
    positions[order] = np.arange(n_rows)
    left_size = neighborhood_size // 2
    neighborhoods: list[set[int]] = []
    for row in range(n_rows):
        start = positions[row] - left_size
        start = min(max(start, 0), n_rows - neighborhood_size - 1)
        candidates = order[start : start + neighborhood_size + 1]
        neighbors = set(int(candidate) for candidate in candidates)
        neighbors.remove(row)
        neighborhoods.append(neighbors)
    return neighborhoods


def rank_neighborhood_overlap(
    reference_embedding: np.ndarray,
    comparison_embedding: np.ndarray,
    neighborhood_size: int = 50,
    *,
    tie_permutations: int = 4,
    random_state: int = 0,
) -> float:
    """Return tie-randomized overlap of local rank neighborhoods.

    For each common finite neuron, the metric compares its fixed-size set of
    nearest rows along each one-dimensional order.  A globally reversed
    Rastermap order is equivalent because rank neighborhoods do not depend on
    orientation.  Rastermap's upsampled grid can assign several neurons the
    exact same position; those neurons have no resolved within-tie order.  We
    therefore break ties independently in the two embeddings and average over
    reproducible random permutations instead of using the shared row index,
    which would spuriously increase agreement.

    The result ranges from 0 (no shared neighbors) to 1 (identical resolved
    neighborhoods).  Even two identical tied embeddings can score below one,
    correctly reflecting their unresolved within-tie order.
    """
    reference, comparison = _common_finite_embeddings(
        reference_embedding,
        comparison_embedding,
    )
    n_rows = reference.size
    if isinstance(neighborhood_size, (bool, np.bool_)) or not isinstance(
        neighborhood_size,
        (int, np.integer),
    ):
        raise TypeError("neighborhood_size must be an integer")
    neighborhood_size = int(neighborhood_size)
    if neighborhood_size <= 0 or neighborhood_size >= n_rows:
        raise ValueError(
            "neighborhood_size must be positive and smaller than the number "
            "of common finite neurons"
        )
    if isinstance(tie_permutations, (bool, np.bool_)) or not isinstance(
        tie_permutations,
        (int, np.integer),
    ):
        raise TypeError("tie_permutations must be an integer")
    tie_permutations = int(tie_permutations)
    if tie_permutations <= 0:
        raise ValueError("tie_permutations must be positive")

    reference_rank = rankdata(reference, method="average")
    comparison_rank = rankdata(comparison, method="average")
    if np.corrcoef(reference_rank, comparison_rank)[0, 1] < 0:
        comparison = -comparison

    has_ties = np.unique(reference).size < n_rows or np.unique(comparison).size < n_rows
    repetitions = tie_permutations if has_ties else 1
    rng = np.random.default_rng(random_state)
    repetition_overlap = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        # ``lexsort`` uses the last key as primary, so the random key changes
        # only the ordering within an exactly tied embedding position.
        reference_order = np.lexsort((rng.random(n_rows), reference))
        comparison_order = np.lexsort((rng.random(n_rows), comparison))
        reference_neighborhoods = _order_neighborhoods(
            reference_order,
            neighborhood_size,
        )
        comparison_neighborhoods = _order_neighborhoods(
            comparison_order,
            neighborhood_size,
        )
        repetition_overlap[repetition] = np.mean(
            [
                len(reference_neighbors & comparison_neighbors) / neighborhood_size
                for reference_neighbors, comparison_neighbors in zip(
                    reference_neighborhoods,
                    comparison_neighborhoods,
                    strict=True,
                )
            ]
        )
    return float(repetition_overlap.mean())


def fit_selected_neurons(
    activity: np.ndarray,
    selected_rows: np.ndarray,
    *,
    n_clusters: int,
    n_PCs: int,
    locality: float,
    time_lag_window: int,
    mean_time: bool,
    time_bin: int,
    superneuron_size: int,
    random_state: int,
    verbose: bool = True,
) -> RastermapResult:
    """Fit official Rastermap to prespecified rows of one complete session.

    The continuous activity amplitudes are converted to float32, as recommended
    by Rastermap, but are otherwise passed unchanged. Rastermap performs its own
    per-neuron z-scoring, PCA, clustering, asymmetric similarity calculation,
    cluster sorting, and upsampling. ``time_bin=1`` retains every original frame.
    Nonselected rows remain NaN in the full-length returned embedding.
    """
    activity = np.asarray(activity)
    if activity.ndim != 2:
        raise ValueError("activity must be a neuron-by-frame matrix")
    selected_rows = np.asarray(selected_rows)
    if selected_rows.ndim != 1 or selected_rows.size == 0:
        raise ValueError("selected_rows must be a non-empty one-dimensional array")
    if not np.issubdtype(selected_rows.dtype, np.integer):
        raise TypeError("selected_rows must contain integer ROI indices")
    if np.any(selected_rows < 0) or np.any(selected_rows >= activity.shape[0]):
        raise IndexError("selected_rows contains an ROI outside activity")
    if np.unique(selected_rows).size != selected_rows.size:
        raise ValueError("selected_rows must not contain duplicate ROI indices")
    selected_rows = selected_rows.astype(np.int64, copy=False)

    selected_activity = np.ascontiguousarray(
        activity[selected_rows],
        dtype=np.float32,
    )
    usable = valid_activity_rows(selected_activity)
    input_rows = selected_rows[usable]
    if input_rows.size < 2:
        raise ValueError("Rastermap requires at least two finite, non-constant neurons")

    working = (
        selected_activity
        if np.all(usable)
        else np.ascontiguousarray(selected_activity[usable], dtype=np.float32)
    )

    from rastermap import Rastermap

    model = Rastermap(
        n_clusters=n_clusters,
        n_PCs=n_PCs,
        locality=locality,
        time_lag_window=time_lag_window,
        mean_time=mean_time,
        time_bin=time_bin,
        bin_size=superneuron_size,
        random_state=random_state,
        keep_norm_X=True,
        verbose=verbose,
    ).fit(working, compute_X_embedding=False)

    model_good = np.asarray(model.igood, dtype=bool).ravel()
    if model_good.size != input_rows.size:
        raise RuntimeError("Rastermap returned an unexpected validity mask shape")
    sorted_local = np.asarray(model.isort, dtype=np.int64)
    sorted_local = sorted_local[model_good[sorted_local]]
    sorted_original = input_rows[sorted_local]
    valid_rows = input_rows[model_good]
    X_embedding = ordered_superneurons(
        np.asarray(model.X),
        sorted_local,
        superneuron_size,
    )

    embedding = np.full(activity.shape[0], np.nan, dtype=np.float32)
    embedding[input_rows] = np.asarray(model.embedding, dtype=np.float32).ravel()
    return RastermapResult(
        X_embedding=X_embedding,
        embedding=embedding,
        isort=sorted_original,
        valid_rows=valid_rows,
        runtime_seconds=float(model.runtime),
    )


def fit_all_neurons(
    activity: np.ndarray,
    *,
    n_clusters: int,
    n_PCs: int,
    locality: float,
    time_lag_window: int,
    mean_time: bool,
    time_bin: int,
    superneuron_size: int,
    random_state: int,
    verbose: bool = True,
) -> RastermapResult:
    """Fit every finite, nonconstant row; retained for baseline visualizations."""
    activity = np.asarray(activity)
    if activity.ndim != 2:
        raise ValueError("activity must be a neuron-by-frame matrix")
    return fit_selected_neurons(
        activity,
        np.arange(activity.shape[0], dtype=np.int64),
        n_clusters=n_clusters,
        n_PCs=n_PCs,
        locality=locality,
        time_lag_window=time_lag_window,
        mean_time=mean_time,
        time_bin=time_bin,
        superneuron_size=superneuron_size,
        random_state=random_state,
        verbose=verbose,
    )


def ordered_superneurons(
    normalized_activity: np.ndarray,
    order: np.ndarray,
    superneuron_size: int,
) -> np.ndarray:
    """Average every ordered row exactly once, including the final partial bin.

    Rastermap's published figures show adjacent sorted neurons averaged into
    superneurons and z-scored across time.  Its generic ``bin1d`` helper drops a
    final incomplete bin; this dataset adapter retains that last group so the
    display still contains every neuron used by the fit.
    """
    normalized_activity = np.asarray(normalized_activity)
    order = np.asarray(order)
    if normalized_activity.ndim != 2:
        raise ValueError("normalized_activity must be neuron by time")
    if order.ndim != 1 or order.size == 0:
        raise ValueError("order must be a non-empty one-dimensional index array")
    if np.any(order < 0) or np.any(order >= normalized_activity.shape[0]):
        raise IndexError("order contains a neuron row outside normalized_activity")
    if np.unique(order).size != order.size:
        raise ValueError("order must not contain duplicate neuron rows")
    if isinstance(superneuron_size, (bool, np.bool_)) or not isinstance(
        superneuron_size, (int, np.integer)
    ):
        raise TypeError("superneuron_size must be an integer")
    if superneuron_size <= 0:
        raise ValueError("superneuron_size must be positive")

    n_groups = (order.size + int(superneuron_size) - 1) // int(superneuron_size)
    grouped = np.empty(
        (n_groups, normalized_activity.shape[1]),
        dtype=np.float32,
    )
    for group in range(n_groups):
        rows = order[
            group * int(superneuron_size) : (group + 1) * int(superneuron_size)
        ]
        grouped[group] = np.mean(
            normalized_activity[rows],
            axis=0,
            dtype=np.float32,
        )

    grouped -= grouped.mean(axis=1, keepdims=True, dtype=np.float32)
    scale = grouped.std(axis=1, keepdims=True, dtype=np.float32)
    np.divide(grouped, scale, out=grouped, where=scale > 0)
    grouped[~np.isfinite(grouped)] = 0
    return grouped


def make_cache_metadata(
    *,
    recording_name: str,
    n_neurons: int,
    n_frames: int,
    fs: float,
    parameters: Mapping[str, Any],
    neuron_selection: Mapping[str, Any] | None = None,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Describe all inputs that must match before a cached fit can be reused."""
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "recording_name": str(recording_name),
        "n_neurons": int(n_neurons),
        "n_frames": int(n_frames),
        "fs": float(fs),
        "signal": "spike_deconv",
        "neuron_selection": (
            {"definition": "all_finite_nonconstant_rows"}
            if neuron_selection is None
            else dict(neuron_selection)
        ),
        "rastermap_version": installed_rastermap_version(),
        "parameters": dict(parameters),
    }
    if source_path is not None:
        source = Path(source_path)
        source_stat = source.stat()
        metadata["source_size_bytes"] = int(source_stat.st_size)
        metadata["source_mtime_ns"] = int(source_stat.st_mtime_ns)
    return metadata


def save_cache(
    path: str | Path,
    result: RastermapResult,
    metadata: Mapping[str, Any],
) -> None:
    """Save a compact Rastermap result and its exact provenance as one NPZ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    payload["runtime_seconds"] = float(result.runtime_seconds)
    np.savez_compressed(
        path,
        metadata=np.asarray(json.dumps(payload, sort_keys=True)),
        X_embedding=result.X_embedding,
        embedding=result.embedding,
        isort=result.isort,
        valid_rows=result.valid_rows,
    )


def load_cache(
    path: str | Path,
    expected_metadata: Mapping[str, Any],
) -> RastermapResult | None:
    """Load a cache only when its recording, package, and settings all match."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as cached:
            metadata = json.loads(str(cached["metadata"].item()))
            runtime = float(metadata.pop("runtime_seconds"))
            if metadata != dict(expected_metadata):
                return None
            return RastermapResult(
                X_embedding=np.asarray(cached["X_embedding"], dtype=np.float32),
                embedding=np.asarray(cached["embedding"], dtype=np.float32),
                isort=np.asarray(cached["isort"], dtype=np.int64),
                valid_rows=np.asarray(cached["valid_rows"], dtype=np.int64),
                runtime_seconds=runtime,
            )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


__all__ = [
    "RastermapResult",
    "active_deconvolution_count_rows",
    "active_deconvolution_rows",
    "fit_all_neurons",
    "fit_selected_neurons",
    "installed_rastermap_version",
    "load_cache",
    "make_cache_metadata",
    "ordered_superneurons",
    "positive_deconvolution_bin_counts",
    "positive_deconvolution_bin_rates",
    "positive_deconvolution_onset_rates",
    "rank_neighborhood_overlap",
    "reversal_invariant_rank_correlation",
    "save_cache",
    "valid_activity_rows",
]
