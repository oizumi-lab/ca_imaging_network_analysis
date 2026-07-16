"""Frozen known-system positive controls for sparse and calcium observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.ndimage
import scipy.optimize
import scipy.signal

from .model import fit_diagonal_ar, fit_ridge_dmd, rolling_forecast_metrics, spectral_class
from .preprocessing import apply_scaler, fit_scaler, randomized_pca


@dataclass
class SimulatedSystem:
    system_class: str
    operator: np.ndarray
    latent: np.ndarray
    mixing: np.ndarray
    observations: dict[str, np.ndarray]


def _one_frame_operator(system_class: str, simulation: dict[str, Any], fs_hz: float) -> np.ndarray:
    dt = 1.0 / fs_hz
    time_constants = np.asarray(simulation["real_mode_time_constants_seconds"], dtype=float)
    eigenvalues = np.exp(-dt / time_constants)
    operator = np.diag(eigenvalues)
    if system_class == "rotational":
        radius = np.exp(-dt / float(simulation["rotation_decay_seconds"]))
        angle = 2 * np.pi * float(simulation["rotation_frequency_hz"]) * dt
        operator[:2, :2] = radius * np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
    elif system_class != "real_relaxation":
        raise ValueError(f"Unknown system class: {system_class}")
    return operator


def _poisson_scale_for_zero_fraction(rate_shape: np.ndarray, target_zero: float) -> float:
    target_zero = float(np.clip(target_zero, 1e-5, 1 - 1e-5))

    def difference(log_scale: float) -> float:
        scale = np.exp(log_scale)
        return float(np.mean(np.exp(-scale * rate_shape)) - target_zero)

    lower, upper = -20.0, 20.0
    if difference(upper) > 0:
        return float(np.exp(upper))
    root = scipy.optimize.brentq(difference, lower, upper)
    return float(np.exp(root))


def simulate_system(
    system_class: str,
    seed: int,
    simulation: dict[str, Any],
    fs_hz: float,
    target_event_zero_fraction: float,
) -> SimulatedSystem:
    """Generate latent dynamics and four causally ordered observation levels."""
    rng = np.random.default_rng(seed)
    dimension = int(simulation["latent_dimension"])
    n_frames = int(simulation["n_frames"])
    burn_in = int(simulation["burn_in_frames"])
    n_neurons = int(simulation["n_neurons"])
    operator = _one_frame_operator(system_class, simulation, fs_hz)
    latent = np.zeros((dimension, burn_in + n_frames), dtype=np.float64)
    process_sd = float(simulation["latent_process_noise_sd"])
    for time in range(1, latent.shape[1]):
        latent[:, time] = operator @ latent[:, time - 1] + process_sd * rng.standard_normal(dimension)
    latent = latent[:, burn_in:]
    mixing = rng.standard_normal((n_neurons, dimension)) / np.sqrt(dimension)
    continuous_clean = mixing @ latent
    continuous_scale = np.std(continuous_clean, axis=1, keepdims=True)
    continuous_scale[continuous_scale == 0] = 1.0
    continuous = continuous_clean + float(simulation["continuous_observation_noise_sd"]) * (
        continuous_scale * rng.standard_normal(continuous_clean.shape)
    )

    standardized_drive = (continuous_clean - np.mean(continuous_clean, axis=1, keepdims=True)) / continuous_scale
    rate_shape = np.maximum(standardized_drive + 0.5, 0.0)
    poisson_scale = _poisson_scale_for_zero_fraction(rate_shape, target_event_zero_fraction)
    events = rng.poisson(poisson_scale * rate_shape).astype(np.float64)
    smoothed = scipy.ndimage.gaussian_filter1d(
        events,
        sigma=float(simulation["smoothing_sigma_frames"]),
        axis=1,
        mode="nearest",
        truncate=3.0,
    )
    decay = np.exp(-1.0 / (fs_hz * float(simulation["calcium_decay_seconds"])))
    calcium = scipy.signal.lfilter([1.0], [1.0, -decay], events, axis=1)
    calcium_sd = np.std(calcium, axis=1, keepdims=True)
    calcium_sd[calcium_sd == 0] = 1.0
    calcium += float(simulation["calcium_noise_fraction"]) * calcium_sd * rng.standard_normal(calcium.shape)
    return SimulatedSystem(
        system_class=system_class,
        operator=operator,
        latent=latent,
        mixing=mixing,
        observations={
            "continuous": continuous,
            "event": events,
            "smoothed_event": smoothed,
            "calcium": calcium,
        },
    )


def _orthonormal_basis(matrix: np.ndarray, dimension: int | None = None) -> np.ndarray:
    q, r = scipy.linalg.qr(matrix, mode="economic", check_finite=False)
    if dimension is not None:
        q = q[:, :dimension]
    return q


def _real_mode_basis(eigenvalues: np.ndarray, eigenvectors: np.ndarray, indices: list[int]) -> np.ndarray:
    columns: list[np.ndarray] = []
    used: set[int] = set()
    for index in indices:
        if index in used:
            continue
        value = eigenvalues[index]
        vector = eigenvectors[:, index]
        if abs(value.imag) > 1e-8:
            partner = int(np.argmin(np.abs(eigenvalues - np.conj(value))))
            used.update({index, partner})
            columns.extend([vector.real, vector.imag])
        else:
            used.add(index)
            columns.append(vector.real)
    return _orthonormal_basis(np.column_stack(columns))


def score_simulation(
    simulated: SimulatedSystem,
    observation_name: str,
    seed: int,
    simulation: dict[str, Any],
    model_config: dict[str, Any],
    fs_hz: float,
) -> dict[str, float | int | str | bool]:
    """Fit the frozen q=6, lag-2 DMD and compare with the known generator."""
    observation = simulated.observations[observation_name]
    train_stop = int(np.floor(observation.shape[1] * 0.8))
    mean, scale, rows, _ = fit_scaler([observation[:, :train_stop]], "rms")
    scaled = apply_scaler([observation], mean, scale, rows)[0]
    rank = int(simulation["rank"])
    components, _, _, _, _ = randomized_pca(
        [scaled[:, :train_stop]],
        max_rank=rank,
        oversamples=8,
        power_iterations=3,
        seed=seed + 101,
    )
    scores = components.T @ scaled
    lag = int(simulation["lag_frames"])
    eta = float(simulation["ridge_relative"])
    model = fit_ridge_dmd([scores[:, :train_stop]], lag=lag, eta=eta)
    diagonal = fit_diagonal_ar([scores[:, :train_stop]], lag=lag, eta=eta)
    forecast_rows = rolling_forecast_metrics(
        model,
        diagonal,
        scores,
        train_stop=train_stop,
        fs_effective=fs_hz,
        horizons_seconds=model_config["forecast_horizons_seconds"],
        explosion_multiplier=float(model_config["forecast_explosion_multiplier"]),
    )
    near_one = min(forecast_rows, key=lambda row: abs(float(row["actual_horizon_seconds"]) - 1.0))

    true_values, true_vectors = scipy.linalg.eig(np.linalg.matrix_power(simulated.operator, lag))
    costs = np.abs(true_values[:, None] - model.eigenvalues[None, :])
    truth_indices, recovered_indices = scipy.optimize.linear_sum_assignment(costs)
    matched_error = costs[truth_indices, recovered_indices] / np.maximum(np.abs(true_values[truth_indices]), 1e-12)

    target_truth_indices = [0, 1]
    target_recovered_indices = [
        int(recovered_indices[np.flatnonzero(truth_indices == index)[0]]) for index in target_truth_indices
    ]
    recovered_basis = _real_mode_basis(model.eigenvalues, model.eigenvectors, target_recovered_indices)
    standardized_mixing = simulated.mixing[rows] / scale[:, None]
    latent_embedding = components.T @ standardized_mixing[:, target_truth_indices]
    true_basis = _orthonormal_basis(latent_embedding, dimension=2)
    overlap = float(np.linalg.norm(true_basis.T @ recovered_basis, ord="fro") ** 2 / 2)

    predicted_class = spectral_class(
        model,
        fs_effective=fs_hz,
        minimum_frequency_hz=float(simulation["classification_frequency_hz"]),
    )
    return {
        "system_class": simulated.system_class,
        "observation": observation_name,
        "seed": seed,
        "target_zero_fraction": float(np.mean(simulated.observations["event"] == 0)),
        "eligible_neurons": int(rows.size),
        "predicted_class": predicted_class,
        "classification_correct": predicted_class == simulated.system_class,
        "latent_embedding_overlap": overlap,
        "median_eigenvalue_relative_error": float(np.median(matched_error)),
        "maximum_eigenvalue_relative_error": float(np.max(matched_error)),
        "skill_persistence_near_one_second": float(near_one["skill_persistence"]),
        "skill_diagonal_near_one_second": float(near_one["skill_diagonal"]),
        "spectral_radius": float(model.diagnostics["spectral_radius"]),
        "finite_fit": bool(model.diagnostics["finite_operator"] and model.diagnostics["finite_eigenvalues"]),
    }


def run_known_systems(
    config: dict[str, Any],
    target_event_zero_fraction: float,
) -> pd.DataFrame:
    simulation = config["simulation"]
    rows: list[dict[str, float | int | str | bool]] = []
    base_seed = int(config["random_seed"])
    for class_index, system_class in enumerate(("real_relaxation", "rotational")):
        for seed_index in range(int(simulation["seeds_per_system"])):
            seed = base_seed + class_index * 10000 + seed_index
            simulated = simulate_system(
                system_class,
                seed,
                simulation,
                fs_hz=float(config["fs_hz"]),
                target_event_zero_fraction=target_event_zero_fraction,
            )
            for observation_name in simulated.observations:
                rows.append(
                    score_simulation(
                        simulated,
                        observation_name,
                        seed,
                        simulation,
                        config["model"],
                        fs_hz=float(config["fs_hz"]),
                    )
                )
    return pd.DataFrame(rows)
