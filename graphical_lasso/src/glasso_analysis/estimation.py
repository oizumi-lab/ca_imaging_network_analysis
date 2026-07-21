"""Exact full-neuron graphical-lasso estimation and validation."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import time
from dataclasses import asdict, dataclass
from functools import cache
from types import ModuleType

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class FitDiagnostics:
    alpha: float
    n_nodes: int
    n_components: int
    n_nontrivial_components: int
    max_component_size: int
    n_edges: int
    native_density: float
    max_iterations: int
    total_iterations: int
    max_abs_duality_gap: float
    max_abs_duality_gap_per_node: float
    max_inverse_error: float
    wall_seconds: float
    quic_cpu_seconds: float
    objective_sum: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def standardize_activity(activity: np.ndarray) -> np.ndarray:
    """Return `(frames, neurons)` float64 data with unit population variance."""
    x = np.asarray(activity, dtype=np.float64).T.copy()
    if x.ndim != 2:
        raise ValueError("activity must be a two-dimensional neurons-by-frames array")
    if not np.isfinite(x).all():
        raise ValueError("activity contains non-finite values")
    x -= x.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(x * x, axis=0, keepdims=True))
    if np.any(scale <= 0):
        raise ValueError("activity contains a constant neuron")
    x /= scale
    return x


def empirical_correlation(standardized: np.ndarray) -> np.ndarray:
    """Population-normalized empirical correlation of standardized samples."""
    x = np.asarray(standardized, dtype=np.float64)
    corr = np.ascontiguousarray((x.T @ x) / x.shape[0])
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    return corr


def common_valid_rows(
    activity_by_state: dict[str, np.ndarray],
) -> np.ndarray:
    """Rows finite and nonconstant in every paired state matrix."""
    if not activity_by_state:
        raise ValueError("at least one state matrix is required")
    first = next(iter(activity_by_state.values()))
    valid = np.ones(first.shape[0], dtype=bool)
    for activity in activity_by_state.values():
        if activity.shape[0] != valid.size:
            raise ValueError("paired state matrices must contain the same rows")
        valid &= np.isfinite(activity).all(axis=1)
        valid &= np.ptp(activity, axis=1) > 0
    return valid


def precision_to_partial(precision: np.ndarray) -> np.ndarray:
    """Convert a positive precision matrix to signed partial correlations."""
    theta = np.asarray(precision, dtype=np.float64)
    diag = np.diag(theta)
    if np.any(diag <= 0) or not np.isfinite(diag).all():
        raise ValueError("precision diagonal must be finite and positive")
    denom = np.sqrt(np.outer(diag, diag))
    partial = -theta / denom
    np.fill_diagonal(partial, 1.0)
    return (partial + partial.T) / 2.0


def covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    """Rescale a positive covariance estimate to marginal correlations."""
    cov = np.asarray(covariance, dtype=np.float64)
    diag = np.diag(cov)
    if np.any(diag <= 0) or not np.isfinite(diag).all():
        raise ValueError("covariance diagonal must be finite and positive")
    corr = cov / np.sqrt(np.outer(diag, diag))
    np.fill_diagonal(corr, 1.0)
    return (corr + corr.T) / 2.0


@cache
def _load_quic_extension() -> ModuleType:
    """Load only skggm's compiled extension, bypassing its Python-2 wrapper."""
    try:
        files = importlib.metadata.files("skggm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "skggm 0.2.5 is required; see graphical_lasso/README.md"
        ) from exc
    candidates = [
        path.locate()
        for path in files
        if "inverse_covariance/pyquic/pyquic" in str(path) and str(path).endswith(".so")
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one compiled pyquic extension, found {candidates}")
    spec = importlib.util.spec_from_file_location("pyquic", candidates[0])
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create an import specification for pyquic")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quic_component(
    sample_covariance: np.ndarray,
    alpha: float,
    theta0: np.ndarray,
    sigma0: np.ndarray,
    *,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Solve one connected graphical-lasso block with QUIC."""
    quic = _load_quic_extension()
    sample_covariance = np.ascontiguousarray(sample_covariance, dtype=np.float64)
    n_nodes = sample_covariance.shape[0]
    penalty = np.full((n_nodes, n_nodes), float(alpha), dtype=np.float64)
    np.fill_diagonal(penalty, 0.0)
    theta = np.ascontiguousarray(theta0, dtype=np.float64).copy()
    sigma = np.ascontiguousarray(sigma0, dtype=np.float64).copy()
    path = np.empty(1, dtype=np.float64)
    objective = np.zeros(1, dtype=np.float64)
    cpu_time = np.zeros(1, dtype=np.float64)
    iterations = np.zeros(1, dtype=np.uint32)
    duality_gap = np.zeros(1, dtype=np.float64)
    quic.quic(
        b"default",
        n_nodes,
        sample_covariance,
        penalty,
        1,
        path,
        float(tol),
        0,
        int(max_iter),
        theta,
        sigma,
        objective,
        cpu_time,
        iterations,
        duality_gap,
    )
    diagnostics = {
        "n_nodes": int(n_nodes),
        "objective": float(objective[0]),
        "cpu_seconds": float(cpu_time[0]),
        "iterations": int(iterations[0]),
        "duality_gap": float(duality_gap[0]),
    }
    return theta, sigma, diagnostics


def fit_screened_graphical_lasso(
    sample_correlation: np.ndarray,
    alpha: float,
    *,
    init_precision: np.ndarray | None = None,
    init_covariance: np.ndarray | None = None,
    tol: float = 1e-6,
    max_iter: int = 500,
    support_tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, FitDiagnostics]:
    """Fit the exact graphical lasso after exact component screening.

    Mazumder & Hastie's component theorem makes this decomposition exactly
    equivalent to fitting the full objective at the same `alpha`.
    """
    sample = np.ascontiguousarray(sample_correlation, dtype=np.float64)
    if sample.ndim != 2 or sample.shape[0] != sample.shape[1]:
        raise ValueError("sample_correlation must be square")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    n_nodes = sample.shape[0]
    if (init_precision is None) != (init_covariance is None):
        raise ValueError("initial precision and covariance must be supplied together")

    screen = csr_matrix(np.abs(sample) > alpha)
    screen.setdiag(False)
    screen.eliminate_zeros()
    n_components, labels = connected_components(screen, directed=False)
    sizes = np.bincount(labels, minlength=n_components)

    precision = np.zeros_like(sample)
    covariance = np.zeros_like(sample)
    component_diagnostics: list[dict[str, float | int]] = []
    max_inverse_error = 0.0
    start = time.perf_counter()

    for label in range(n_components):
        indices = np.flatnonzero(labels == label)
        block = np.ascontiguousarray(sample[np.ix_(indices, indices)])
        if indices.size == 1:
            variance = float(block[0, 0])
            covariance[indices[0], indices[0]] = variance
            precision[indices[0], indices[0]] = 1.0 / variance
            continue

        if init_precision is None:
            sigma0 = np.diag(np.diag(block))
            theta0 = np.diag(1.0 / np.diag(block))
        else:
            theta0 = np.ascontiguousarray(init_precision[np.ix_(indices, indices)])
            sigma0 = np.ascontiguousarray(init_covariance[np.ix_(indices, indices)])

        theta, sigma, diag = _quic_component(
            block,
            alpha,
            theta0,
            sigma0,
            tol=tol,
            max_iter=max_iter,
        )
        if not np.isfinite(theta).all() or not np.isfinite(sigma).all():
            raise RuntimeError(f"QUIC returned non-finite values for component {label}")
        inverse_error = float(
            np.max(np.abs(theta @ sigma - np.eye(indices.size)))
        )
        max_inverse_error = max(max_inverse_error, inverse_error)
        precision[np.ix_(indices, indices)] = theta
        covariance[np.ix_(indices, indices)] = sigma
        component_diagnostics.append(diag)

    max_gap = max(
        (abs(float(item["duality_gap"])) for item in component_diagnostics),
        default=0.0,
    )
    max_component_iterations = max(
        (int(item["iterations"]) for item in component_diagnostics),
        default=0,
    )
    if max_component_iterations >= max_iter:
        raise RuntimeError("at least one QUIC component reached max_iter")
    gap_per_node = max(
        (
            abs(float(item["duality_gap"])) / int(item["n_nodes"])
            for item in component_diagnostics
        ),
        default=0.0,
    )
    if max_gap > 1e-3 or gap_per_node > 1e-8:
        raise RuntimeError(
            "QUIC duality-gap gate failed: "
            f"absolute={max_gap:.3e}, per_node={gap_per_node:.3e}"
        )
    if max_inverse_error > 1e-8:
        raise RuntimeError(f"precision/covariance inverse gate failed: {max_inverse_error:.3e}")

    upper = np.triu(np.abs(precision) > support_tol, k=1)
    n_edges = int(np.count_nonzero(upper))
    n_possible = n_nodes * (n_nodes - 1) // 2
    diagnostics = FitDiagnostics(
        alpha=float(alpha),
        n_nodes=n_nodes,
        n_components=int(n_components),
        n_nontrivial_components=int(np.count_nonzero(sizes > 1)),
        max_component_size=int(sizes.max()),
        n_edges=n_edges,
        native_density=n_edges / n_possible,
        max_iterations=max_component_iterations,
        total_iterations=sum(int(item["iterations"]) for item in component_diagnostics),
        max_abs_duality_gap=max_gap,
        max_abs_duality_gap_per_node=gap_per_node,
        max_inverse_error=max_inverse_error,
        wall_seconds=time.perf_counter() - start,
        quic_cpu_seconds=sum(float(item["cpu_seconds"]) for item in component_diagnostics),
        objective_sum=sum(float(item["objective"]) for item in component_diagnostics),
    )
    return precision, covariance, diagnostics
