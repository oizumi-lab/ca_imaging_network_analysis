"""Blocked development selection and untouched empirical forecast evaluation."""

from __future__ import annotations

from dataclasses import asdict
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
from .preprocessing import ArmSpec, FrozenRepresentation


def arm_ranks_lags(arm_id: str, config: dict[str, Any]) -> tuple[list[int], list[int]]:
    model = config["model"]
    if arm_id == "B4":
        return list(model["b4_ranks"]), list(model["b4_lags_bins"])
    lags = list(model["native_lags_frames"])
    if arm_id in {"Sz", "Sc"}:
        lags.append(int(model["smoothed_extra_lag_frames"]))
    return list(model["native_ranks"]), lags


def load_scores(
    meta: RecordingMetadata,
    window: Window,
    spec: ArmSpec,
    representation: FrozenRepresentation,
    rank: int | None = None,
    trim_samples: int = 0,
) -> np.ndarray:
    raw = read_signal(meta, spec.signal, window.start, window.stop)
    scores = representation.transform(raw, rank=rank)
    if trim_samples:
        if 2 * trim_samples >= scores.shape[1]:
            raise ValueError("Trim removes the entire score window")
        scores = scores[:, trim_samples:-trim_samples]
    return scores


def _with_one_step_diagnostic(
    model: DMDModel,
    diagonal: DMDModel,
    scores: np.ndarray,
    train_stop: int,
    fs_effective: float,
    config: dict[str, Any],
) -> list[dict[str, float | int | bool | str]]:
    configured = list(config["model"]["forecast_horizons_seconds"])
    diagnostic = model.lag / fs_effective
    requested = configured + [diagnostic]
    rows = rolling_forecast_metrics(
        model,
        diagonal,
        scores,
        train_stop=train_stop,
        fs_effective=fs_effective,
        horizons_seconds=requested,
        explosion_multiplier=float(config["model"]["forecast_explosion_multiplier"]),
    )
    unique: dict[int, dict[str, float | int | bool | str]] = {}
    gate_steps = {int(round(value * fs_effective)) for value in configured}
    for row in rows:
        step = int(row["actual_horizon_steps"])
        row["horizon_role"] = "gate" if step in gate_steps else "one_step_diagnostic"
        unique.setdefault(step, row)
    return list(unique.values())


def development_grid(
    meta: RecordingMetadata,
    windows: Iterable[Window],
    spec: ArmSpec,
    representation: FrozenRepresentation,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Fit local first-80%, score final-20% models throughout the frozen grid."""
    windows = list(windows)
    ranks, lags = arm_ranks_lags(spec.arm_id, config)
    maximum_rank = max(ranks)
    score_windows = {
        window.window_id: load_scores(meta, window, spec, representation, rank=maximum_rank)
        for window in windows
    }
    rows: list[dict[str, Any]] = []
    minimum_pairs = int(config["model"]["minimum_pairs_per_dimension"])
    fs_effective = float(config["fs_hz"]) / spec.bin_frames
    for rank in ranks:
        for lag in lags:
            for eta in config["model"]["ridge_relative"]:
                for window in windows:
                    scores = score_windows[window.window_id][:rank]
                    train_stop = int(np.floor(scores.shape[1] * float(config["windows"]["train_fraction"])))
                    pair_count = train_stop - lag
                    if pair_count < minimum_pairs * rank:
                        continue
                    model = fit_ridge_dmd([scores[:, :train_stop]], lag=lag, eta=float(eta))
                    diagonal = fit_diagonal_ar([scores[:, :train_stop]], lag=lag, eta=float(eta))
                    metrics = _with_one_step_diagnostic(
                        model,
                        diagonal,
                        scores,
                        train_stop,
                        fs_effective,
                        config,
                    )
                    for metric in metrics:
                        sse_persistence = float(metric["sse_persistence"])
                        diagonal_skill = (
                            1.0 - float(metric["sse_diagonal"]) / sse_persistence
                            if sse_persistence > np.finfo(float).eps
                            else np.nan
                        )
                        rows.append(
                            {
                                **asdict(window),
                                "arm": spec.arm_id,
                                "rank": rank,
                                "lag": lag,
                                "eta": float(eta),
                                "diagonal_skill_persistence": diagonal_skill,
                                **model.diagnostics,
                                **metric,
                            }
                        )
    return pd.DataFrame(rows)


def _target_steps(config: dict[str, Any], fs_effective: float) -> tuple[int, int]:
    horizons = list(config["model"]["forecast_horizons_seconds"])
    return int(round(horizons[1] * fs_effective)), int(round(horizons[2] * fs_effective))


def summarize_grid(grid: pd.DataFrame, fs_effective: float, config: dict[str, Any]) -> pd.DataFrame:
    primary_step, secondary_step = _target_steps(config, fs_effective)
    keys = ["recording", "arm", "rank", "lag", "eta"]
    primary = (
        grid[grid["actual_horizon_steps"] == primary_step]
        .groupby(keys, as_index=False)
        .agg(
            development_windows=("window_id", "nunique"),
            median_skill_one_second=("skill_persistence", "median"),
            median_equal_pc_skill_one_second=("equal_pc_skill_persistence", "median"),
            median_diagonal_skill_one_second=("diagonal_skill_persistence", "median"),
            finite_fraction=("finite_forecast", "mean"),
            explosive_fraction=("explosive_forecast", "mean"),
        )
    )
    secondary = (
        grid[grid["actual_horizon_steps"] == secondary_step]
        .groupby(keys, as_index=False)
        .agg(
            median_skill_two_seconds=("skill_persistence", "median"),
            median_equal_pc_skill_two_seconds=("equal_pc_skill_persistence", "median"),
        )
    )
    return primary.merge(secondary, on=keys, how="left")


def select_configuration(
    summary: pd.DataFrame,
    precedent_lag: int,
    minimum_rank: int | None = None,
) -> dict[str, Any] | None:
    eligible = summary.copy()
    if minimum_rank is not None:
        eligible = eligible[eligible["rank"] >= minimum_rank]
    eligible = eligible[
        np.isfinite(eligible["median_skill_one_second"])
        & (eligible["finite_fraction"] >= 0.95)
        & (eligible["explosive_fraction"] <= 0.05)
    ]
    if eligible.empty:
        return None
    eligible = eligible.assign(lag_distance=np.abs(eligible["lag"] - precedent_lag))
    ordered = eligible.sort_values(
        [
            "median_skill_one_second",
            "median_skill_two_seconds",
            "rank",
            "lag_distance",
            "eta",
        ],
        ascending=[False, False, True, True, False],
        kind="mergesort",
    )
    selected = ordered.iloc[0].to_dict()
    matching = summary[
        (summary["rank"] == selected["rank"])
        & (summary["lag"] == selected["lag"])
    ].sort_values(
        ["median_diagonal_skill_one_second", "eta"],
        ascending=[False, False],
        kind="mergesort",
    )
    selected["diagonal_eta"] = float(matching.iloc[0]["eta"])
    return selected


def evaluate_configuration(
    meta: RecordingMetadata,
    windows: Iterable[Window],
    spec: ArmSpec,
    representation: FrozenRepresentation,
    selected: dict[str, Any],
    config: dict[str, Any],
    trim_samples: int = 0,
    sensitivity_name: str = "primary",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    spectra: list[pd.DataFrame] = []
    rank = int(selected["rank"])
    lag = int(selected["lag"])
    eta = float(selected["eta"])
    diagonal_eta = float(selected["diagonal_eta"])
    fs_effective = float(config["fs_hz"]) / spec.bin_frames
    for window in windows:
        scores = load_scores(
            meta,
            window,
            spec,
            representation,
            rank=rank,
            trim_samples=trim_samples,
        )
        original_samples = scores.shape[1] + 2 * trim_samples
        original_train_stop = int(
            np.floor(original_samples * float(config["windows"]["train_fraction"]))
        )
        train_stop = original_train_stop - trim_samples
        if not 0 < train_stop < scores.shape[1]:
            raise ValueError("Edge trimming leaves an invalid fixed train/test split")
        model = fit_ridge_dmd([scores[:, :train_stop]], lag=lag, eta=eta)
        diagonal = fit_diagonal_ar([scores[:, :train_stop]], lag=lag, eta=diagonal_eta)
        for metric in _with_one_step_diagnostic(
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
                    "sensitivity": sensitivity_name,
                    "trim_samples_each_edge": trim_samples,
                    "original_train_stop": original_train_stop,
                    "effective_train_stop": train_stop,
                    "original_split_preserved": True,
                    **model.diagnostics,
                    **metric,
                }
            )
        spectrum = eigenvalue_diagnostics(
            model,
            fs_effective=fs_effective,
            window_samples=train_stop,
            minimum_modulus=float(config["model"]["rotation_minimum_modulus"]),
            nearly_real_tolerance=float(config["model"]["rotation_nearly_real_tolerance"]),
        )
        for key, value in {
            "window_id": window.window_id,
            "recording": window.recording,
            "label": window.label,
            "arm": spec.arm_id,
            "rank": rank,
            "lag": lag,
            "eta": eta,
            "sensitivity": sensitivity_name,
            "trim_samples_each_edge": trim_samples,
            "original_train_stop": original_train_stop,
            "effective_train_stop": train_stop,
            "original_split_preserved": True,
        }.items():
            spectrum[key] = value
        spectra.append(spectrum)
    return pd.DataFrame(rows), pd.concat(spectra, ignore_index=True)


def fit_long_block(
    meta: RecordingMetadata,
    window: Window,
    spec: ArmSpec,
    representation: FrozenRepresentation,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Nested 64/16/20 split: select in the first 80%, score the final 20%."""
    maximum_rank = max(config["model"]["long_block_ranks"])
    scores_all = load_scores(meta, window, spec, representation, rank=maximum_rank)
    outer_train = int(np.floor(scores_all.shape[1] * 0.8))
    inner_train = int(np.floor(scores_all.shape[1] * 0.64))
    fs_effective = float(config["fs_hz"]) / spec.bin_frames
    _, lags = arm_ranks_lags(spec.arm_id, config)
    candidates: list[dict[str, Any]] = []
    for rank in config["model"]["long_block_ranks"]:
        for lag in lags:
            if inner_train - lag < int(config["model"]["minimum_pairs_per_dimension"]) * rank:
                continue
            for eta in config["model"]["ridge_relative"]:
                model = fit_ridge_dmd([scores_all[:rank, :inner_train]], lag=lag, eta=float(eta))
                diagonal = fit_diagonal_ar([scores_all[:rank, :inner_train]], lag=lag, eta=float(eta))
                validation_scores = scores_all[:rank, :outer_train]
                metrics = _with_one_step_diagnostic(
                    model,
                    diagonal,
                    validation_scores,
                    inner_train,
                    fs_effective,
                    config,
                )
                target_step, _ = _target_steps(config, fs_effective)
                target = [row for row in metrics if row["actual_horizon_steps"] == target_step]
                if target:
                    candidates.append(
                        {
                            "rank": rank,
                            "lag": lag,
                            "eta": float(eta),
                            "validation_skill": float(target[0]["skill_persistence"]),
                        }
                    )
    if not candidates:
        raise ValueError(f"No eligible long-block configuration for {window.window_id}:{spec.arm_id}")
    candidate_table = pd.DataFrame(candidates).sort_values(
        ["validation_skill", "rank", "lag", "eta"],
        ascending=[False, True, True, False],
        kind="mergesort",
    )
    selected = candidate_table.iloc[0].to_dict()
    rank, lag, eta = int(selected["rank"]), int(selected["lag"]), float(selected["eta"])
    validation_scores = scores_all[:rank, :outer_train]
    selection_model = fit_ridge_dmd(
        [scores_all[:rank, :inner_train]],
        lag=lag,
        eta=eta,
    )
    diagonal_candidates: list[dict[str, float]] = []
    target_step, _ = _target_steps(config, fs_effective)
    for diagonal_eta in config["model"]["ridge_relative"]:
        diagonal_candidate = fit_diagonal_ar(
            [scores_all[:rank, :inner_train]],
            lag=lag,
            eta=float(diagonal_eta),
        )
        candidate_metrics = _with_one_step_diagnostic(
            selection_model,
            diagonal_candidate,
            validation_scores,
            inner_train,
            fs_effective,
            config,
        )
        target = [row for row in candidate_metrics if row["actual_horizon_steps"] == target_step]
        if target and float(target[0]["sse_persistence"]) > np.finfo(float).eps:
            diagonal_skill = 1.0 - float(target[0]["sse_diagonal"]) / float(
                target[0]["sse_persistence"]
            )
            diagonal_candidates.append(
                {"diagonal_eta": float(diagonal_eta), "diagonal_validation_skill": diagonal_skill}
            )
    if not diagonal_candidates:
        raise ValueError(f"No eligible long-block diagonal baseline for {window.window_id}:{spec.arm_id}")
    diagonal_choice = sorted(
        diagonal_candidates,
        key=lambda item: (-item["diagonal_validation_skill"], -item["diagonal_eta"]),
    )[0]
    diagonal_eta = float(diagonal_choice["diagonal_eta"])
    selected.update(diagonal_choice)
    model = fit_ridge_dmd([scores_all[:rank, :outer_train]], lag=lag, eta=eta)
    diagonal = fit_diagonal_ar([scores_all[:rank, :outer_train]], lag=lag, eta=diagonal_eta)
    metrics = _with_one_step_diagnostic(
        model,
        diagonal,
        scores_all[:rank],
        outer_train,
        fs_effective,
        config,
    )
    rows = pd.DataFrame(
        [
            {
                **asdict(window),
                "arm": spec.arm_id,
                "rank": rank,
                "lag": lag,
                "eta": eta,
                "diagonal_eta": diagonal_eta,
                "inner_validation_skill": float(selected["validation_skill"]),
                "diagonal_inner_validation_skill": float(
                    selected["diagonal_validation_skill"]
                ),
                **model.diagnostics,
                **metric,
            }
            for metric in metrics
        ]
    )
    return rows, selected


def representative_prediction(
    meta: RecordingMetadata,
    window: Window,
    spec: ArmSpec,
    representation: FrozenRepresentation,
    selected: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, object]:
    """Return observed and rolling-origin DMD arrays at the gate horizon near 1 s."""
    rank = int(selected["rank"])
    lag = int(selected["lag"])
    scores = load_scores(meta, window, spec, representation, rank=rank)
    train_stop = int(np.floor(scores.shape[1] * float(config["windows"]["train_fraction"])))
    model = fit_ridge_dmd([scores[:, :train_stop]], lag=lag, eta=float(selected["eta"]))
    fs_effective = float(config["fs_hz"]) / spec.bin_frames
    horizon = int(round(float(config["model"]["forecast_horizons_seconds"][1]) * fs_effective))
    if horizon < lag or horizon % lag:
        raise ValueError("Selected configuration cannot represent the one-second gate horizon")
    origins = np.arange(train_stop, scores.shape[1] - horizon)
    prediction = np.linalg.matrix_power(model.operator, horizon // lag) @ scores[:, origins]
    truth = scores[:, origins + horizon]
    return {
        "recording": window.recording,
        "label": window.label,
        "arm": spec.arm_id,
        "truth": truth[:3],
        "prediction": prediction[:3],
        "time": (origins + horizon - train_stop) / fs_effective,
        "horizon_seconds": horizon / fs_effective,
    }
