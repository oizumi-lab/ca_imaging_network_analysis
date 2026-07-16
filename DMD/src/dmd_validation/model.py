"""Numerically stable ridge DMD, diagonal baselines, forecasts, and spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.linalg


@dataclass
class DMDModel:
    operator: np.ndarray
    lag: int
    alpha: float
    eta: float | None
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    diagnostics: dict[str, float | int | bool]
    model_kind: str = "full"


def snapshot_pairs(segments: Iterable[np.ndarray], lag: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Form pairs separately inside each legal segment; never join boundaries."""
    if lag <= 0:
        raise ValueError("lag must be positive")
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    counts: list[int] = []
    q: int | None = None
    for segment in segments:
        segment = np.asarray(segment, dtype=np.float64)
        if segment.ndim != 2:
            raise ValueError("Each segment must be coordinate by time")
        if q is None:
            q = segment.shape[0]
        elif segment.shape[0] != q:
            raise ValueError("All segments must use the same coordinate dimension")
        count = max(0, segment.shape[1] - lag)
        if count:
            x_parts.append(segment[:, :-lag])
            y_parts.append(segment[:, lag:])
            counts.append(count)
    if not x_parts:
        raise ValueError("No legal snapshot pair remains at this lag")
    return np.concatenate(x_parts, axis=1), np.concatenate(y_parts, axis=1), counts


def _operator_diagnostics(
    operator: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    singular_values: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool]]:
    prediction = operator @ x
    residual_denominator = np.linalg.norm(y, ord="fro")
    residual = np.linalg.norm(y - prediction, ord="fro")
    normalized_residual = float(residual / residual_denominator) if residual_denominator else np.nan
    eigenvalues, eigenvectors = scipy.linalg.eig(operator, check_finite=False)
    q, pair_count = x.shape
    tolerance = max(q, pair_count) * np.finfo(float).eps * singular_values[0]
    numerical_rank = int(np.count_nonzero(singular_values > tolerance))
    minimum_square = float(singular_values[-1] ** 2) if singular_values.size == q else 0.0
    denominator = minimum_square + alpha
    condition = (
        float((singular_values[0] ** 2 + alpha) / denominator)
        if denominator > 0
        else np.inf
    )
    squared = singular_values**2
    effective_df = float(
        np.sum(np.divide(squared, squared + alpha, out=np.zeros_like(squared), where=(squared + alpha) > 0))
    )
    diagnostics: dict[str, float | int | bool] = {
        "q": q,
        "pair_count": pair_count,
        "numerical_rank": numerical_rank,
        "effective_degrees_of_freedom": effective_df,
        "normal_condition_number": condition,
        "normalized_residual": normalized_residual,
        "spectral_radius": float(np.max(np.abs(eigenvalues))),
        "eigenvector_condition_number": float(np.linalg.cond(eigenvectors)),
        "finite_operator": bool(np.all(np.isfinite(operator))),
        "finite_eigenvalues": bool(np.all(np.isfinite(eigenvalues))),
    }
    return eigenvalues, eigenvectors, diagnostics


def fit_ridge_dmd(
    segments: Iterable[np.ndarray],
    lag: int,
    eta: float | None = None,
    raw_alpha: float | None = None,
) -> DMDModel:
    """Fit ridge DMD with a thin-SVD solve and explicit regularization scale."""
    if (eta is None) == (raw_alpha is None):
        raise ValueError("Specify exactly one of eta or raw_alpha")
    x, y, pair_counts = snapshot_pairs(segments, lag)
    u, singular_values, vh = scipy.linalg.svd(
        x, full_matrices=False, check_finite=False, lapack_driver="gesdd"
    )
    scale = float(np.sum(singular_values**2) / x.shape[0])
    alpha = float(raw_alpha if raw_alpha is not None else eta * scale)
    if alpha < 0:
        raise ValueError("Ridge penalty cannot be negative")
    denominator = singular_values**2 + alpha
    shrink = np.divide(
        singular_values,
        denominator,
        out=np.zeros_like(singular_values),
        where=denominator > np.finfo(float).tiny,
    )
    operator = ((y @ vh.T) * shrink[None, :]) @ u.T
    eigenvalues, eigenvectors, diagnostics = _operator_diagnostics(
        operator, x, y, singular_values, alpha
    )
    diagnostics["segment_count"] = len(pair_counts)
    diagnostics["pair_count_by_segment_min"] = min(pair_counts)
    diagnostics["pair_count_by_segment_max"] = max(pair_counts)
    diagnostics["predictor_gram_mean_eigenvalue"] = scale
    return DMDModel(
        operator=operator,
        lag=lag,
        alpha=alpha,
        eta=eta,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        diagnostics=diagnostics,
        model_kind="full",
    )


def fit_diagonal_ar(segments: Iterable[np.ndarray], lag: int, eta: float) -> DMDModel:
    """Fit a no-intercept diagonal ridge AR baseline in the same PC coordinates."""
    x, y, pair_counts = snapshot_pairs(segments, lag)
    singular_values = scipy.linalg.svdvals(x, check_finite=False)
    scale = float(np.sum(singular_values**2) / x.shape[0])
    alpha = float(eta * scale)
    denominator = np.einsum("ij,ij->i", x, x, optimize=True) + alpha
    numerator = np.einsum("ij,ij->i", y, x, optimize=True)
    diagonal = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    operator = np.diag(diagonal)
    eigenvalues, eigenvectors, diagnostics = _operator_diagnostics(
        operator, x, y, singular_values, alpha
    )
    diagnostics["segment_count"] = len(pair_counts)
    diagnostics["predictor_gram_mean_eigenvalue"] = scale
    return DMDModel(
        operator=operator,
        lag=lag,
        alpha=alpha,
        eta=eta,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        diagnostics=diagnostics,
        model_kind="diagonal",
    )


def _safe_skill(model_sse: float, baseline_sse: float) -> float:
    tolerance = np.finfo(float).eps * max(1.0, abs(model_sse), abs(baseline_sse))
    if not np.isfinite(model_sse) or not np.isfinite(baseline_sse) or baseline_sse <= tolerance:
        return np.nan
    return float(1.0 - model_sse / baseline_sse)


def _sse(truth: np.ndarray, prediction: np.ndarray, weights: np.ndarray | None = None) -> float:
    residual = truth - prediction
    if weights is not None:
        residual = residual * np.sqrt(weights[:, None])
    return float(np.einsum("ij,ij->", residual, residual, optimize=True))


def rolling_forecast_metrics(
    model: DMDModel,
    diagonal_model: DMDModel,
    scores: np.ndarray,
    train_stop: int,
    fs_effective: float,
    horizons_seconds: Iterable[float],
    explosion_multiplier: float,
) -> list[dict[str, float | int | bool]]:
    """Score local rolling-origin forecasts on a final contiguous test block."""
    scores = np.asarray(scores, dtype=np.float64)
    if model.lag != diagonal_model.lag:
        raise ValueError("Full and diagonal models must use the same lag")
    train = scores[:, :train_stop]
    train_mean = np.mean(train, axis=1)
    train_variance = np.var(train, axis=1)
    equal_pc_weights = 1.0 / np.maximum(train_variance, np.finfo(float).eps)
    maximum_train_norm = max(float(np.max(np.linalg.norm(train, axis=0))), np.finfo(float).eps)
    rows: list[dict[str, float | int | bool]] = []
    for target_seconds in horizons_seconds:
        desired = int(round(float(target_seconds) * fs_effective))
        if desired < model.lag or desired % model.lag:
            continue
        power = desired // model.lag
        origins = np.arange(train_stop, scores.shape[1] - desired, dtype=int)
        if origins.size == 0:
            continue
        truth = scores[:, origins + desired]
        initial = scores[:, origins]
        full_prediction = np.linalg.matrix_power(model.operator, power) @ initial
        diagonal_prediction = np.linalg.matrix_power(diagonal_model.operator, power) @ initial
        persistence = initial
        mean_prediction = np.repeat(train_mean[:, None], origins.size, axis=1)
        predictions_finite = bool(np.all(np.isfinite(full_prediction)))
        prediction_norm = (
            float(np.max(np.linalg.norm(full_prediction, axis=0))) if predictions_finite else np.inf
        )
        explosive = (not predictions_finite) or prediction_norm > explosion_multiplier * maximum_train_norm
        sse_full = _sse(truth, full_prediction)
        sse_mean = _sse(truth, mean_prediction)
        sse_persistence = _sse(truth, persistence)
        sse_diagonal = _sse(truth, diagonal_prediction)
        equal_full = _sse(truth, full_prediction, equal_pc_weights)
        equal_mean = _sse(truth, mean_prediction, equal_pc_weights)
        equal_persistence = _sse(truth, persistence, equal_pc_weights)
        equal_diagonal = _sse(truth, diagonal_prediction, equal_pc_weights)
        truth_flat = truth.ravel()
        prediction_flat = full_prediction.ravel()
        correlation = (
            float(np.corrcoef(truth_flat, prediction_flat)[0, 1])
            if np.std(truth_flat) > 0 and np.std(prediction_flat) > 0
            else np.nan
        )
        truth_scale = float(np.std(truth_flat))
        nrmse = (
            float(np.sqrt(sse_full / truth_flat.size) / truth_scale) if truth_scale > 0 else np.nan
        )
        rows.append(
            {
                "target_horizon_seconds": float(target_seconds),
                "actual_horizon_steps": desired,
                "actual_horizon_seconds": float(desired / fs_effective),
                "operator_power": power,
                "origin_count": int(origins.size),
                "sse_dmd": sse_full,
                "sse_mean": sse_mean,
                "sse_persistence": sse_persistence,
                "sse_diagonal": sse_diagonal,
                "skill_mean": _safe_skill(sse_full, sse_mean),
                "skill_persistence": _safe_skill(sse_full, sse_persistence),
                "skill_diagonal": _safe_skill(sse_full, sse_diagonal),
                "equal_pc_skill_mean": _safe_skill(equal_full, equal_mean),
                "equal_pc_skill_persistence": _safe_skill(equal_full, equal_persistence),
                "equal_pc_skill_diagonal": _safe_skill(equal_full, equal_diagonal),
                "correlation": correlation,
                "nrmse": nrmse,
                "finite_forecast": predictions_finite,
                "explosive_forecast": bool(explosive),
                "maximum_prediction_norm_ratio": prediction_norm / maximum_train_norm,
            }
        )
    return rows


def eigenvalue_diagnostics(
    model: DMDModel,
    fs_effective: float,
    window_samples: int,
    minimum_modulus: float = 0.25,
    nearly_real_tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Return discrete spectra and cautiously interpreted continuous summaries."""
    delta_seconds = model.lag / fs_effective
    rows: list[dict[str, float | int | bool]] = []
    for index, value in enumerate(model.eigenvalues):
        modulus = float(abs(value))
        angle = float(np.angle(value))
        is_complex = bool(abs(value.imag) > nearly_real_tolerance)
        positive_pair_member = bool(value.imag > nearly_real_tolerance)
        stable = bool(0 < modulus < 1)
        frequency = abs(angle) / (2 * np.pi * delta_seconds) if is_complex else 0.0
        decay = -delta_seconds / np.log1p(modulus - 1) if stable else np.nan
        rotations = (
            abs(angle) * np.log(10) / (-2 * np.pi * np.log1p(modulus - 1))
            if stable and is_complex
            else np.nan
        )
        cycles = frequency * (window_samples / fs_effective)
        rows.append(
            {
                "eigen_index": index,
                "real": float(value.real),
                "imag": float(value.imag),
                "modulus": modulus,
                "angle_radians": angle,
                "stable_discrete": stable,
                "complex_mode": is_complex,
                "positive_conjugate_member": positive_pair_member,
                "frequency_hz": float(frequency),
                "decay_seconds": float(decay),
                "rotations_per_decade": float(rotations),
                "cycles_in_window": float(cycles),
                "three_cycle_eligible": bool(cycles >= 3),
                "v1_code_filter": bool(modulus > minimum_modulus),
                "caption_realpart_filter": bool(value.real > minimum_modulus),
                "interpretable_rotation": bool(
                    positive_pair_member and stable and modulus > minimum_modulus and cycles >= 3
                ),
            }
        )
    return pd.DataFrame(rows)


def spectral_class(
    model: DMDModel,
    fs_effective: float,
    minimum_frequency_hz: float,
) -> str:
    """Classify a spectrum as rotational when a stable conjugate mode is resolvable."""
    diagnostics = eigenvalue_diagnostics(model, fs_effective, window_samples=10**9)
    rotational = diagnostics[
        diagnostics["positive_conjugate_member"]
        & diagnostics["stable_discrete"]
        & (diagnostics["frequency_hz"] >= minimum_frequency_hz)
    ]
    return "rotational" if not rotational.empty else "real_relaxation"
