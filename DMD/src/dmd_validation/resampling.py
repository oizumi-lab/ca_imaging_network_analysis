"""Empirical bootstrap, invariant-subspace, and neuron-subset validation."""

from __future__ import annotations

from dataclasses import asdict
from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data import RecordingMetadata, Window, read_signal
from .model import (
    DMDModel,
    eigenvalue_diagnostics,
    fit_diagonal_ar,
    fit_ridge_dmd,
    rolling_forecast_metrics,
)
from .preprocessing import ArmSpec, FrozenRepresentation, randomized_pca
from .stability import (
    InvariantSubspace,
    bootstrap_segments,
    match_subspace_to_reference,
    moving_block_bootstrap_pairs,
    projector_distance,
    projector_similarity,
    select_energy_ranked_subspace,
)
from .validation import load_scores


def deterministic_seed(base_seed: int, *coordinates: int) -> int:
    """Map integer analysis coordinates to one reproducible NumPy seed."""
    sequence = np.random.SeedSequence([int(base_seed), *(int(value) for value in coordinates)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _train_stop(n_samples: int, config: dict[str, Any]) -> int:
    return int(np.floor(n_samples * float(config["windows"]["train_fraction"])))


def _forecast_rows(
    model: DMDModel,
    diagonal: DMDModel,
    scores: np.ndarray,
    train_stop: int,
    fs_effective: float,
    config: dict[str, Any],
) -> list[dict[str, float | int | bool]]:
    return rolling_forecast_metrics(
        model,
        diagonal,
        scores,
        train_stop=train_stop,
        fs_effective=fs_effective,
        horizons_seconds=config["model"]["forecast_horizons_seconds"],
        explosion_multiplier=float(config["model"]["forecast_explosion_multiplier"]),
    )


def _target_role(actual_seconds: float) -> str:
    if abs(actual_seconds - 1.0) < 0.2:
        return "near_one_second"
    if abs(actual_seconds - 2.0) < 0.2:
        return "near_two_seconds"
    return "short_horizon"


def spectral_class(
    model: DMDModel,
    fs_effective: float,
    window_samples: int,
    config: dict[str, Any],
) -> tuple[str, int]:
    """Classify only rotations that meet the frozen modulus and three-cycle rules."""
    diagnostics = eigenvalue_diagnostics(
        model,
        fs_effective=fs_effective,
        window_samples=window_samples,
        minimum_modulus=float(config["model"]["rotation_minimum_modulus"]),
        nearly_real_tolerance=float(config["model"]["rotation_nearly_real_tolerance"]),
    )
    count = int(diagnostics["interpretable_rotation"].sum())
    return ("resolved_rotation" if count else "no_resolved_rotation"), count


def fit_development_reference(
    development_scores: Iterable[np.ndarray],
    rank: int,
    lag: int,
    eta: float,
    target_dimension: int,
    config: dict[str, Any],
) -> tuple[DMDModel, InvariantSubspace]:
    """Fit one pooled, no-cross-window development reference."""
    score_windows = [np.asarray(scores[:rank], dtype=np.float64) for scores in development_scores]
    model = fit_ridge_dmd(score_windows, lag=lag, eta=eta)
    subspace = select_energy_ranked_subspace(
        model,
        np.concatenate(score_windows, axis=1),
        target_dimension=target_dimension,
        nearly_real_tolerance=float(config["model"]["rotation_nearly_real_tolerance"]),
    )
    return model, subspace


def evaluate_tracking_windows(
    meta: RecordingMetadata,
    windows: Iterable[Window],
    spec: ArmSpec,
    representation: FrozenRepresentation,
    selected: dict[str, Any],
    reference: InvariantSubspace,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, InvariantSubspace], dict[str, np.ndarray]]:
    """Fit the untouched-window training portions and match to the fixed reference."""
    rank = int(selected["rank"])
    lag = int(selected["lag"])
    eta = float(selected["eta"])
    fs_effective = float(config["fs_hz"]) / spec.bin_frames
    rows: list[dict[str, Any]] = []
    subspaces: dict[str, InvariantSubspace] = {}
    scores_by_window: dict[str, np.ndarray] = {}
    for window in windows:
        scores = load_scores(meta, window, spec, representation, rank=rank)
        scores_by_window[window.window_id] = scores
        train_stop = _train_stop(scores.shape[1], config)
        model = fit_ridge_dmd([scores[:, :train_stop]], lag=lag, eta=eta)
        base = {
            **asdict(window),
            "arm": spec.arm_id,
            "rank": rank,
            "lag": lag,
            "eta": eta,
            "train_stop": train_stop,
            **model.diagnostics,
        }
        try:
            matched, matches = match_subspace_to_reference(
                reference,
                model,
                scores[:, :train_stop],
                nearly_real_tolerance=float(
                    config["model"]["rotation_nearly_real_tolerance"]
                ),
            )
            subspaces[window.window_id] = matched
            classification, rotation_count = spectral_class(
                model,
                fs_effective,
                train_stop,
                config,
            )
            rows.append(
                {
                    **base,
                    "match_success": True,
                    "reference_similarity": projector_similarity(
                        reference.projector, matched.projector
                    ),
                    "reference_distance": projector_distance(
                        reference.projector, matched.projector
                    ),
                    "mean_matched_spectral_distance": float(
                        np.mean([match.spectral_distance for match in matches])
                    ),
                    "spectral_class": classification,
                    "interpretable_rotation_groups": rotation_count,
                    "failure": "",
                }
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            rows.append(
                {
                    **base,
                    "match_success": False,
                    "reference_similarity": np.nan,
                    "reference_distance": np.nan,
                    "mean_matched_spectral_distance": np.nan,
                    "spectral_class": "unmatched",
                    "interpretable_rotation_groups": 0,
                    "failure": f"{type(error).__name__}: {error}",
                }
            )
    return pd.DataFrame(rows), subspaces, scores_by_window


def bootstrap_predictive_windows(
    windows: Iterable[Window],
    scores_by_window: dict[str, np.ndarray],
    selected: dict[str, Any],
    spec: ArmSpec,
    config: dict[str, Any],
    recording_index: int,
) -> pd.DataFrame:
    """Bootstrap operator estimation and rescore the fixed contiguous test tail."""
    rank = int(selected["rank"])
    lag = int(selected["lag"])
    eta = float(selected["eta"])
    diagonal_eta = float(selected["diagonal_eta"])
    repetitions = int(config["resampling"]["bootstrap_repetitions"])
    block_length = max(int(config["resampling"]["minimum_block_frames"]), lag + 1)
    fs_effective = float(config["fs_hz"]) / spec.bin_frames
    rows: list[dict[str, Any]] = []
    for window_index, window in enumerate(windows):
        scores = scores_by_window[window.window_id][:rank]
        train_stop = _train_stop(scores.shape[1], config)
        training = scores[:, :train_stop]
        for repetition in range(repetitions):
            sample = moving_block_bootstrap_pairs(
                n_samples=train_stop,
                lag=lag,
                block_length=block_length,
                seed=deterministic_seed(
                    int(config["random_seed"]), recording_index, window_index, repetition, 101
                ),
            )
            segments = bootstrap_segments(training, sample)
            model = fit_ridge_dmd(segments, lag=lag, eta=eta)
            diagonal = fit_diagonal_ar(segments, lag=lag, eta=diagonal_eta)
            for metric in _forecast_rows(
                model,
                diagonal,
                scores,
                train_stop,
                fs_effective,
                config,
            ):
                rows.append(
                    {
                        **asdict(window),
                        "arm": spec.arm_id,
                        "rank": rank,
                        "lag": lag,
                        "eta": eta,
                        "diagonal_eta": diagonal_eta,
                        "repetition": repetition,
                        "block_length_samples": block_length,
                        "bootstrap_blocks": len(sample.block_bounds),
                        "cross_block_pairs": 0,
                        "horizon_role": _target_role(
                            float(metric["actual_horizon_seconds"])
                        ),
                        **model.diagnostics,
                        **metric,
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_tracking_windows(
    windows: Iterable[Window],
    scores_by_window: dict[str, np.ndarray],
    selected: dict[str, Any],
    reference: InvariantSubspace,
    original_subspaces: dict[str, InvariantSubspace],
    config: dict[str, Any],
    recording_index: int,
) -> pd.DataFrame:
    """Quantify fixed-reference and within-window invariant-subspace uncertainty."""
    rank = int(selected["rank"])
    lag = int(selected["lag"])
    eta = float(selected["eta"])
    repetitions = int(config["resampling"]["bootstrap_repetitions"])
    block_length = max(int(config["resampling"]["minimum_block_frames"]), lag + 1)
    rows: list[dict[str, Any]] = []
    for window_index, window in enumerate(windows):
        scores = scores_by_window[window.window_id][:rank]
        train_stop = _train_stop(scores.shape[1], config)
        training = scores[:, :train_stop]
        original = original_subspaces.get(window.window_id)
        for repetition in range(repetitions):
            sample = moving_block_bootstrap_pairs(
                n_samples=train_stop,
                lag=lag,
                block_length=block_length,
                seed=deterministic_seed(
                    int(config["random_seed"]), recording_index, window_index, repetition, 202
                ),
            )
            segments = bootstrap_segments(training, sample)
            model = fit_ridge_dmd(segments, lag=lag, eta=eta)
            base = {
                **asdict(window),
                "rank": rank,
                "lag": lag,
                "eta": eta,
                "repetition": repetition,
                "block_length_samples": block_length,
                "bootstrap_blocks": len(sample.block_bounds),
                "cross_block_pairs": 0,
                **model.diagnostics,
            }
            try:
                matched, matches = match_subspace_to_reference(
                    reference,
                    model,
                    np.concatenate(segments, axis=1),
                    nearly_real_tolerance=float(
                        config["model"]["rotation_nearly_real_tolerance"]
                    ),
                )
                rows.append(
                    {
                        **base,
                        "match_success": True,
                        "reference_similarity": projector_similarity(
                            reference.projector, matched.projector
                        ),
                        "reference_distance": projector_distance(
                            reference.projector, matched.projector
                        ),
                        "within_window_similarity": (
                            projector_similarity(original.projector, matched.projector)
                            if original is not None
                            else np.nan
                        ),
                        "within_window_distance": (
                            projector_distance(original.projector, matched.projector)
                            if original is not None
                            else np.nan
                        ),
                        "mean_matched_spectral_distance": float(
                            np.mean([match.spectral_distance for match in matches])
                        ),
                        "failure": "",
                    }
                )
            except (ValueError, np.linalg.LinAlgError) as error:
                rows.append(
                    {
                        **base,
                        "match_success": False,
                        "reference_similarity": np.nan,
                        "reference_distance": np.nan,
                        "within_window_similarity": np.nan,
                        "within_window_distance": np.nan,
                        "mean_matched_spectral_distance": np.nan,
                        "failure": f"{type(error).__name__}: {error}",
                    }
                )
    return pd.DataFrame(rows)


def synchronized_cell_intervals(
    bootstrap_metrics: pd.DataFrame,
    confidence_level: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synchronize repetitions, take three-window medians, then percentile CIs."""
    grouped = (
        bootstrap_metrics.groupby(
            ["recording", "label", "horizon_role", "repetition"],
            as_index=False,
        )
        .agg(
            windows=("window_id", "nunique"),
            median_skill_persistence=("skill_persistence", "median"),
            median_skill_diagonal=("skill_diagonal", "median"),
            finite_fraction=("finite_forecast", "mean"),
            explosive_fraction=("explosive_forecast", "mean"),
        )
    )
    tail = (1.0 - confidence_level) / 2.0
    summary_rows: list[dict[str, Any]] = []
    for keys, group in grouped.groupby(["recording", "label", "horizon_role"], sort=False):
        summary_rows.append(
            {
                "recording": keys[0],
                "label": keys[1],
                "horizon_role": keys[2],
                "repetitions": int(group["repetition"].nunique()),
                "windows_per_repetition_min": int(group["windows"].min()),
                "bootstrap_median_skill_persistence": float(
                    group["median_skill_persistence"].median()
                ),
                "skill_persistence_ci_lower": float(
                    group["median_skill_persistence"].quantile(tail)
                ),
                "skill_persistence_ci_upper": float(
                    group["median_skill_persistence"].quantile(1 - tail)
                ),
                "bootstrap_median_skill_diagonal": float(
                    group["median_skill_diagonal"].median()
                ),
                "skill_diagonal_ci_lower": float(
                    group["median_skill_diagonal"].quantile(tail)
                ),
                "skill_diagonal_ci_upper": float(
                    group["median_skill_diagonal"].quantile(1 - tail)
                ),
                "finite_fraction": float(group["finite_fraction"].mean()),
                "explosive_fraction": float(group["explosive_fraction"].mean()),
            }
        )
    return grouped, pd.DataFrame(summary_rows)


def tracking_resolution(
    original_subspaces: dict[str, InvariantSubspace],
    evaluation_windows: Iterable[Window],
    bootstrap_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute all same-label between-window distances and the frozen ratio."""
    rows: list[dict[str, Any]] = []
    windows = list(evaluation_windows)
    for label in dict.fromkeys(window.label for window in windows):
        candidates = [
            window for window in windows if window.label == label and window.window_id in original_subspaces
        ]
        for first, second in combinations(candidates, 2):
            rows.append(
                {
                    "recording": first.recording,
                    "label": label,
                    "first_window": first.window_id,
                    "second_window": second.window_id,
                    "between_window_distance": projector_distance(
                        original_subspaces[first.window_id].projector,
                        original_subspaces[second.window_id].projector,
                    ),
                }
            )
    between = pd.DataFrame(rows)
    within_values = bootstrap_metrics.loc[
        bootstrap_metrics["match_success"], "within_window_distance"
    ].dropna()
    between_values = between["between_window_distance"].dropna() if not between.empty else pd.Series(dtype=float)
    numerator = float(within_values.median()) if not within_values.empty else np.nan
    denominator = float(between_values.median()) if not between_values.empty else np.nan
    ratio = numerator / denominator if np.isfinite(denominator) and denominator > 0 else np.nan
    summary = {
        "median_within_window_bootstrap_distance": numerator,
        "median_between_window_same_label_distance": denominator,
        "tracking_resolution_ratio": float(ratio),
        "within_distance_count": int(len(within_values)),
        "between_distance_count": int(len(between_values)),
    }
    return between, summary


def deterministic_neuron_subset_classes(
    meta: RecordingMetadata,
    development_windows: Iterable[Window],
    evaluation_windows: Iterable[Window],
    spec: ArmSpec,
    representation: FrozenRepresentation,
    selected: dict[str, Any],
    config: dict[str, Any],
    recording_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Refit equal-size disjoint neuron-subset PCA bases and compare spectral class."""
    development_windows = list(development_windows)
    evaluation_windows = list(evaluation_windows)
    rank = int(selected["rank"])
    lag = int(selected["lag"])
    eta = float(selected["eta"])
    subset_size = min(
        int(config["resampling"]["neuron_subset_size"]),
        representation.eligible_rows.size // 2,
    )
    if subset_size <= rank:
        raise ValueError("Neuron subsets must contain more rows than the tracking rank")
    rng = np.random.default_rng(
        deterministic_seed(int(config["random_seed"]), recording_index, 303)
    )
    order = rng.permutation(representation.eligible_rows.size)
    subset_positions = {
        "subset_A": np.sort(order[:subset_size]),
        "subset_B": np.sort(order[subset_size : 2 * subset_size]),
    }
    raw_development = [
        read_signal(meta, spec.signal, window.start, window.stop)
        for window in development_windows
    ]
    raw_evaluation = {
        window.window_id: read_signal(meta, spec.signal, window.start, window.stop)
        for window in evaluation_windows
    }
    fs_effective = float(config["fs_hz"]) / spec.bin_frames
    fit_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    for subset_index, (subset_name, positions) in enumerate(subset_positions.items()):
        neuron_rows = representation.eligible_rows[positions]
        for row in neuron_rows:
            membership_rows.append(
                {
                    "recording": meta.name,
                    "subset": subset_name,
                    "neuron_row_zero_based": int(row),
                }
            )
        mean = representation.mean[positions]
        scale = representation.scale[positions]
        development_scaled = [
            (raw[neuron_rows] - mean[:, None]) / scale[:, None]
            for raw in raw_development
        ]
        components, _, _, _, orthogonality_error = randomized_pca(
            development_scaled,
            max_rank=rank,
            oversamples=int(config["model"]["pca_oversamples"]),
            power_iterations=int(config["model"]["pca_power_iterations"]),
            seed=deterministic_seed(
                int(config["random_seed"]), recording_index, subset_index, 404
            ),
        )
        for window in evaluation_windows:
            scaled = (raw_evaluation[window.window_id][neuron_rows] - mean[:, None]) / scale[:, None]
            scores = components.T @ scaled
            train_stop = _train_stop(scores.shape[1], config)
            model = fit_ridge_dmd([scores[:, :train_stop]], lag=lag, eta=eta)
            classification, rotation_count = spectral_class(
                model,
                fs_effective,
                train_stop,
                config,
            )
            fit_rows.append(
                {
                    **asdict(window),
                    "subset": subset_name,
                    "subset_size": subset_size,
                    "rank": rank,
                    "lag": lag,
                    "eta": eta,
                    "pca_orthogonality_error": orthogonality_error,
                    "spectral_class": classification,
                    "interpretable_rotation_groups": rotation_count,
                    **model.diagnostics,
                }
            )
    fits = pd.DataFrame(fit_rows)
    cell_rows: list[dict[str, Any]] = []
    for keys, group in fits.groupby(["recording", "label", "subset"], sort=False):
        counts = group["spectral_class"].value_counts()
        majority = sorted(counts.index, key=lambda value: (-counts[value], value))[0]
        cell_rows.append(
            {
                "recording": keys[0],
                "label": keys[1],
                "subset": keys[2],
                "windows": int(group["window_id"].nunique()),
                "majority_spectral_class": majority,
                "majority_fraction": float(counts[majority] / len(group)),
            }
        )
    cells = pd.DataFrame(cell_rows)
    agreement = cells.pivot(
        index=["recording", "label"],
        columns="subset",
        values="majority_spectral_class",
    ).reset_index()
    agreement["subset_class_agreement"] = agreement["subset_A"] == agreement["subset_B"]
    audit = {
        "subset_size": subset_size,
        "subsets_disjoint": bool(
            not np.intersect1d(subset_positions["subset_A"], subset_positions["subset_B"]).size
        ),
        "eligible_neurons": int(representation.eligible_rows.size),
        "classes_agree_all_cells": bool(agreement["subset_class_agreement"].all()),
    }
    return fits, pd.DataFrame(membership_rows), {"audit": audit, "agreement": agreement}


__all__ = [
    "bootstrap_predictive_windows",
    "bootstrap_tracking_windows",
    "deterministic_neuron_subset_classes",
    "deterministic_seed",
    "evaluate_tracking_windows",
    "fit_development_reference",
    "spectral_class",
    "synchronized_cell_intervals",
    "tracking_resolution",
]
