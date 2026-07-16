"""Empirical circular-shift and independent-stationary negative controls."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data import RecordingMetadata, Window, read_signal, state_runs
from .model import fit_diagonal_ar, fit_ridge_dmd, rolling_forecast_metrics
from .preprocessing import ArmSpec, FrozenRepresentation
from .resampling import deterministic_seed, spectral_class
from .stability import (
    InvariantSubspace,
    apply_rowwise_indices,
    match_subspace_to_reference,
    projector_similarity,
    stationary_iid_null,
    within_bout_circular_shift_indices,
)


def containing_state_run(meta: RecordingMetadata, window: Window) -> tuple[int, int]:
    """Return the unique acquisition-split constant-label bout containing a window."""
    matches = [
        (start, stop)
        for start, stop, code, segment in state_runs(meta, window.state_code)
        if segment == window.segment and start <= window.start and window.stop <= stop
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one containing bout for {window.window_id}, found {matches}")
    return matches[0]


def _forecast_rows(
    scores: np.ndarray,
    selected: dict[str, Any],
    fs_effective: float,
    config: dict[str, Any],
) -> tuple[list[dict[str, float | int | bool]], object]:
    rank = int(selected["rank"])
    lag = int(selected["lag"])
    train_stop = int(np.floor(scores.shape[1] * float(config["windows"]["train_fraction"])))
    model = fit_ridge_dmd(
        [scores[:rank, :train_stop]],
        lag=lag,
        eta=float(selected["eta"]),
    )
    diagonal = fit_diagonal_ar(
        [scores[:rank, :train_stop]],
        lag=lag,
        eta=float(selected["diagonal_eta"]),
    )
    rows = rolling_forecast_metrics(
        model,
        diagonal,
        scores[:rank],
        train_stop=train_stop,
        fs_effective=fs_effective,
        horizons_seconds=config["model"]["forecast_horizons_seconds"],
        explosion_multiplier=float(config["model"]["forecast_explosion_multiplier"]),
    )
    return rows, model


def _tracking_metrics(
    scores: np.ndarray,
    selected: dict[str, Any],
    reference: InvariantSubspace,
    fs_effective: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    rank = int(selected["rank"])
    lag = int(selected["lag"])
    train_stop = int(np.floor(scores.shape[1] * float(config["windows"]["train_fraction"])))
    model = fit_ridge_dmd(
        [scores[:rank, :train_stop]],
        lag=lag,
        eta=float(selected["eta"]),
    )
    classification, rotation_count = spectral_class(
        model,
        fs_effective,
        train_stop,
        config,
    )
    try:
        matched, matches = match_subspace_to_reference(
            reference,
            model,
            scores[:rank, :train_stop],
            nearly_real_tolerance=float(config["model"]["rotation_nearly_real_tolerance"]),
        )
        return {
            "tracking_match_success": True,
            "reference_similarity": projector_similarity(reference.projector, matched.projector),
            "mean_matched_spectral_distance": float(
                np.mean([match.spectral_distance for match in matches])
            ),
            "spectral_class": classification,
            "interpretable_rotation_groups": rotation_count,
            "tracking_failure": "",
            "tracking_spectral_radius": model.diagnostics["spectral_radius"],
        }
    except (ValueError, np.linalg.LinAlgError) as error:
        return {
            "tracking_match_success": False,
            "reference_similarity": np.nan,
            "mean_matched_spectral_distance": np.nan,
            "spectral_class": classification,
            "interpretable_rotation_groups": rotation_count,
            "tracking_failure": f"{type(error).__name__}: {error}",
            "tracking_spectral_radius": model.diagnostics["spectral_radius"],
        }


def _transform_surrogate(
    values: np.ndarray,
    representation: FrozenRepresentation,
    maximum_rank: int,
) -> np.ndarray:
    scaled = (values - representation.mean[:, None]) / representation.scale[:, None]
    return representation.components[:, :maximum_rank].T @ scaled


def evaluate_negative_controls(
    meta: RecordingMetadata,
    evaluation_windows: Iterable[Window],
    spec: ArmSpec,
    representation: FrozenRepresentation,
    predictive_selected: dict[str, Any],
    tracking_selected: dict[str, Any],
    reference: InvariantSubspace,
    config: dict[str, Any],
    recording_index: int,
) -> pd.DataFrame:
    """Generate both null families inside each legal constant-label bout."""
    if spec.bin_frames != 1:
        raise ValueError("The initial negative-control implementation requires a native-frame arm")
    repetitions = int(config["resampling"]["circular_shift_repetitions"])
    stationary_repetitions = int(config["resampling"]["stationary_null_repetitions"])
    if repetitions != stationary_repetitions:
        raise ValueError("Synchronized null comparison requires equal repetition counts")
    maximum_rank = max(int(predictive_selected["rank"]), int(tracking_selected["rank"]))
    fs_effective = float(config["fs_hz"]) / spec.bin_frames
    windows = list(evaluation_windows)
    grouped: dict[tuple[int, int], list[tuple[int, Window]]] = {}
    for window_index, window in enumerate(windows):
        grouped.setdefault(containing_state_run(meta, window), []).append((window_index, window))
    rows: list[dict[str, Any]] = []
    minimum_shift = int(config["resampling"]["minimum_block_frames"])
    for (bout_start, bout_stop), bout_windows in grouped.items():
        raw_bout = read_signal(meta, spec.signal, bout_start, bout_stop)[
            representation.eligible_rows
        ]
        bout_length = bout_stop - bout_start
        for window_index, window in bout_windows:
            local_start = window.start - bout_start
            local_stop = window.stop - bout_start
            for repetition in range(repetitions):
                circular = within_bout_circular_shift_indices(
                    n_rows=raw_bout.shape[0],
                    bout_start=0,
                    bout_stop=bout_length,
                    target_start=local_start,
                    target_stop=local_stop,
                    seed=deterministic_seed(
                        int(config["random_seed"]), recording_index, window_index, repetition, 505
                    ),
                    minimum_shift=minimum_shift,
                )
                circular_values = apply_rowwise_indices(raw_bout, circular.source_indices)
                stationary = stationary_iid_null(
                    raw_bout,
                    seed=deterministic_seed(
                        int(config["random_seed"]), recording_index, window_index, repetition, 606
                    ),
                    n_samples=window.n_frames,
                )
                samples = {
                    "circular_shift": (
                        circular_values,
                        int(np.sum(np.any(np.diff(circular.source_indices, axis=1) < 0, axis=1))),
                    ),
                    "stationary_iid": (stationary.values, 0),
                }
                for null_kind, (values, wrapped_rows) in samples.items():
                    scores = _transform_surrogate(values, representation, maximum_rank)
                    forecast_rows, forecast_model = _forecast_rows(
                        scores,
                        predictive_selected,
                        fs_effective,
                        config,
                    )
                    tracking = _tracking_metrics(
                        scores,
                        tracking_selected,
                        reference,
                        fs_effective,
                        config,
                    )
                    for metric in forecast_rows:
                        actual = float(metric["actual_horizon_seconds"])
                        if abs(actual - 1.0) < 0.2:
                            role = "near_one_second"
                        elif abs(actual - 2.0) < 0.2:
                            role = "near_two_seconds"
                        else:
                            role = "short_horizon"
                        rows.append(
                            {
                                **asdict(window),
                                "arm": spec.arm_id,
                                "null_kind": null_kind,
                                "repetition": repetition,
                                "bout_start": bout_start,
                                "bout_stop": bout_stop,
                                "bout_length": bout_length,
                                "minimum_circular_shift": minimum_shift,
                                "source_indices_inside_bout": True,
                                "periodic_wrap_rows": wrapped_rows,
                                "horizon_role": role,
                                "predictive_rank": int(predictive_selected["rank"]),
                                "predictive_lag": int(predictive_selected["lag"]),
                                "predictive_eta": float(predictive_selected["eta"]),
                                "predictive_diagonal_eta": float(
                                    predictive_selected["diagonal_eta"]
                                ),
                                "forecast_spectral_radius": forecast_model.diagnostics[
                                    "spectral_radius"
                                ],
                                **tracking,
                                **metric,
                            }
                        )
        del raw_bout
    return pd.DataFrame(rows)


def summarize_negative_controls(
    metrics: pd.DataFrame,
    observed_heldout: pd.DataFrame,
    percentile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create synchronized cell null distributions and observed-vs-null decisions."""
    synchronized = (
        metrics.groupby(
            ["null_kind", "recording", "label", "horizon_role", "repetition"],
            as_index=False,
        )
        .agg(
            windows=("window_id", "nunique"),
            median_skill_persistence=("skill_persistence", "median"),
            median_skill_diagonal=("skill_diagonal", "median"),
            median_reference_similarity=("reference_similarity", "median"),
            tracking_match_fraction=("tracking_match_success", "mean"),
            resolved_rotation_fraction=(
                "spectral_class",
                lambda values: float(np.mean(np.asarray(values) == "resolved_rotation")),
            ),
        )
    )
    observed = observed_heldout[
        (observed_heldout["arm"] == "P")
        & (observed_heldout["sensitivity"] == "primary")
        & (observed_heldout["horizon_role"] == "gate")
    ].copy()
    observed["horizon_role"] = np.where(
        (observed["actual_horizon_seconds"] - 1.0).abs() < 0.2,
        "near_one_second",
        np.where(
            (observed["actual_horizon_seconds"] - 2.0).abs() < 0.2,
            "near_two_seconds",
            "short_horizon",
        ),
    )
    observed_summary = (
        observed.groupby(["recording", "label", "horizon_role"], as_index=False)
        .agg(
            observed_windows=("window_id", "nunique"),
            observed_median_skill_persistence=("skill_persistence", "median"),
            observed_median_skill_diagonal=("skill_diagonal", "median"),
        )
    )
    rows: list[dict[str, Any]] = []
    for keys, group in synchronized.groupby(
        ["null_kind", "recording", "label", "horizon_role"], sort=False
    ):
        observed_row = observed_summary[
            (observed_summary["recording"] == keys[1])
            & (observed_summary["label"] == keys[2])
            & (observed_summary["horizon_role"] == keys[3])
        ]
        observed_diagonal = (
            float(observed_row.iloc[0]["observed_median_skill_diagonal"])
            if not observed_row.empty
            else np.nan
        )
        null_threshold = float(group["median_skill_diagonal"].quantile(percentile / 100.0))
        rows.append(
            {
                "null_kind": keys[0],
                "recording": keys[1],
                "label": keys[2],
                "horizon_role": keys[3],
                "repetitions": int(group["repetition"].nunique()),
                "windows_per_repetition_min": int(group["windows"].min()),
                "null_median_skill_persistence": float(
                    group["median_skill_persistence"].median()
                ),
                "null_median_skill_diagonal": float(group["median_skill_diagonal"].median()),
                "null_skill_diagonal_percentile": null_threshold,
                "observed_median_skill_diagonal": observed_diagonal,
                "observed_exceeds_null_percentile": bool(observed_diagonal > null_threshold),
                "observed_positive_diagonal_gain": bool(observed_diagonal > 0),
                "coordinated_mode_qualifier_cell": bool(
                    observed_diagonal > 0 and observed_diagonal > null_threshold
                ),
                "null_median_reference_similarity": float(
                    group["median_reference_similarity"].median()
                ),
                "null_tracking_match_fraction": float(group["tracking_match_fraction"].mean()),
                "null_resolved_rotation_fraction": float(
                    group["resolved_rotation_fraction"].mean()
                ),
            }
        )
    return synchronized, pd.DataFrame(rows)


__all__ = [
    "containing_state_run",
    "evaluate_negative_controls",
    "summarize_negative_controls",
]
