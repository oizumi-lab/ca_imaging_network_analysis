"""Invariant-subspace stability, block bootstrap, and temporal null controls.

The functions in this module deliberately keep source indices and eigengroup
matches explicit.  That makes the two most important safeguards auditable:
projector scores may not use the entire ambient PC space, and lagged bootstrap
pairs may not cross an artificial block join.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import scipy.linalg
import scipy.optimize

from .model import DMDModel


@dataclass(frozen=True)
class EigenGroup:
    """One real mode or one two-dimensional real conjugate-pair subspace."""

    indices: tuple[int, ...]
    eigenvalues: tuple[complex, ...]
    basis: np.ndarray
    energy: float

    @property
    def dimension(self) -> int:
        return int(self.basis.shape[1])

    @property
    def energy_per_dimension(self) -> float:
        return float(self.energy / self.dimension)


@dataclass(frozen=True)
class InvariantSubspace:
    """A proper, energy-selected real invariant subspace."""

    groups: tuple[EigenGroup, ...]
    basis: np.ndarray
    projector: np.ndarray
    captured_energy: float

    @property
    def dimension(self) -> int:
        return int(self.basis.shape[1])

    @property
    def ambient_dimension(self) -> int:
        return int(self.basis.shape[0])


@dataclass(frozen=True)
class EigenvalueMatch:
    """Minimum-cost assignment from reference to candidate eigenvalues."""

    reference_indices: np.ndarray
    candidate_indices: np.ndarray
    relative_distances: np.ndarray


@dataclass(frozen=True)
class EigenGroupMatch:
    """One dimension-compatible fixed-reference eigengroup assignment."""

    reference_group: int
    candidate_group: int
    spectral_distance: float
    subspace_similarity: float
    total_cost: float


@dataclass(frozen=True)
class BlockBootstrapPairs:
    """Auditable moving-block pair indices with no cross-block transitions."""

    predictor_indices: np.ndarray
    response_indices: np.ndarray
    block_ids: np.ndarray
    block_bounds: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class CircularShiftSample:
    """Independent neuronwise phase shifts restricted to one source bout."""

    shifts: np.ndarray
    source_indices: np.ndarray
    bout_start: int
    bout_stop: int


@dataclass(frozen=True)
class StationaryNullSample:
    """Independent rowwise empirical-marginal draws for a stationary null."""

    values: np.ndarray
    source_indices: np.ndarray


def _canonical_real_basis(matrix: np.ndarray, expected_dimension: int | None = None) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("A nonempty two-dimensional basis matrix is required")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Basis matrix contains non-finite values")
    u, singular_values, _ = scipy.linalg.svd(
        matrix, full_matrices=False, check_finite=False, lapack_driver="gesdd"
    )
    tolerance = max(matrix.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank == 0:
        raise ValueError("Basis matrix has numerical rank zero")
    if expected_dimension is not None and rank != expected_dimension:
        raise ValueError(
            f"Invariant vectors have rank {rank}, expected {expected_dimension}; "
            "the requested eigenspace is defective or unresolved"
        )
    basis = u[:, :rank]
    for column in range(rank):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    return basis


def _proper_subspace_basis(basis: np.ndarray) -> np.ndarray:
    orthonormal = _canonical_real_basis(basis)
    ambient, dimension = orthonormal.shape
    if dimension >= ambient:
        raise ValueError(
            "Invariant-subspace dimension must be smaller than the ambient PC dimension; "
            "a full-space projector is identically the identity and cannot measure stability"
        )
    return orthonormal


def orthogonal_projector(basis: np.ndarray) -> np.ndarray:
    """Return a projector after rejecting the uninformative full-space case."""
    orthonormal = _proper_subspace_basis(basis)
    return orthonormal @ orthonormal.T


def _validated_projector(projector: np.ndarray) -> tuple[np.ndarray, int]:
    projector = np.asarray(projector, dtype=np.float64)
    if projector.ndim != 2 or projector.shape[0] != projector.shape[1]:
        raise ValueError("Projectors must be square matrices")
    if not np.all(np.isfinite(projector)):
        raise ValueError("Projector contains non-finite values")
    tolerance = 1e-8 * max(1, projector.shape[0])
    if not np.allclose(projector, projector.T, atol=tolerance, rtol=tolerance):
        raise ValueError("Projector must be symmetric")
    if not np.allclose(projector @ projector, projector, atol=tolerance, rtol=tolerance):
        raise ValueError("Projector must be idempotent")
    rank = int(np.linalg.matrix_rank(projector, tol=tolerance))
    if rank <= 0:
        raise ValueError("Projector rank must be positive")
    if rank >= projector.shape[0]:
        raise ValueError("A full-space projector cannot measure subspace stability")
    return projector, rank


def projector_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Compute mean squared principal-angle cosine for equal-rank projectors."""
    first, first_rank = _validated_projector(first)
    second, second_rank = _validated_projector(second)
    if first.shape != second.shape:
        raise ValueError("Projectors must use the same ambient coordinates")
    if first_rank != second_rank:
        raise ValueError("Projector similarity requires equal subspace dimensions")
    similarity = float(np.trace(first @ second) / first_rank)
    return float(np.clip(similarity, 0.0, 1.0))


def projector_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return normalized chordal distance, ``sqrt(1 - similarity)``."""
    return float(np.sqrt(max(0.0, 1.0 - projector_similarity(first, second))))


def _group_basis(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    indices: tuple[int, ...],
) -> np.ndarray:
    if len(indices) == 1:
        return _canonical_real_basis(eigenvectors[:, [indices[0]]].real, expected_dimension=1)
    positive = max(indices, key=lambda index: eigenvalues[index].imag)
    vector = eigenvectors[:, positive]
    return _canonical_real_basis(
        np.column_stack([vector.real, vector.imag]), expected_dimension=2
    )


def energy_ranked_eigengroups(
    model: DMDModel,
    training_scores: np.ndarray,
    nearly_real_tolerance: float = 1e-8,
    conjugate_tolerance: float = 1e-6,
) -> list[EigenGroup]:
    """Build real invariant eigengroups and rank them by modal train energy.

    Complex conjugates are one two-dimensional real group.  Energy is the
    mean squared modal coefficient, corrected for eigenvector normalization.
    This implements the frozen protocol's modal-coefficient ranking; the
    orthonormal real span is retained separately for projector geometry.
    """
    values = np.asarray(model.eigenvalues, dtype=np.complex128)
    vectors = np.asarray(model.eigenvectors, dtype=np.complex128)
    scores = np.asarray(training_scores, dtype=np.float64)
    q = model.operator.shape[0]
    if model.operator.shape != (q, q) or values.shape != (q,) or vectors.shape != (q, q):
        raise ValueError("Model operator, eigenvalues, and eigenvectors have inconsistent shapes")
    if scores.ndim != 2 or scores.shape[0] != q or scores.shape[1] == 0:
        raise ValueError("Training scores must be q by nonzero time")
    if not (
        np.all(np.isfinite(values))
        and np.all(np.isfinite(vectors))
        and np.all(np.isfinite(scores))
    ):
        raise ValueError("Eigendecomposition and training scores must be finite")

    scale = np.maximum(1.0, np.abs(values))
    nearly_real = np.abs(values.imag) <= nearly_real_tolerance * scale
    positive = np.flatnonzero((~nearly_real) & (values.imag > 0))
    negative = np.flatnonzero((~nearly_real) & (values.imag < 0))
    if positive.size != negative.size:
        raise ValueError("Non-real eigenvalues do not form complete conjugate pairs")

    paired: list[tuple[int, int]] = []
    if positive.size:
        costs = np.abs(values[positive, None] - np.conj(values[negative[None, :]]))
        positive_rows, negative_columns = scipy.optimize.linear_sum_assignment(costs)
        for positive_row, negative_column in zip(positive_rows, negative_columns, strict=True):
            first = int(positive[positive_row])
            second = int(negative[negative_column])
            error = float(costs[positive_row, negative_column])
            tolerance = conjugate_tolerance * max(1.0, abs(values[first]))
            if error > tolerance:
                raise ValueError(
                    f"Eigenvalues {values[first]!r} and {values[second]!r} are not a "
                    "numerically resolved conjugate pair"
                )
            paired.append((first, second))

    coefficients = scipy.linalg.pinv(vectors, check_finite=False) @ scores
    vector_norm_squared = np.einsum("ij,ij->j", vectors.conj(), vectors).real
    modal_energy = (
        np.mean(np.abs(coefficients) ** 2, axis=1) * vector_norm_squared
    ).real
    index_groups = [(int(index),) for index in np.flatnonzero(nearly_real)] + paired
    groups: list[EigenGroup] = []
    for indices in index_groups:
        basis = _group_basis(values, vectors, indices)
        energy = float(np.sum(modal_energy[list(indices)]))
        ordered_indices = tuple(
            sorted(indices, key=lambda index: (-values[index].imag, values[index].real, index))
        )
        groups.append(
            EigenGroup(
                indices=ordered_indices,
                eigenvalues=tuple(complex(values[index]) for index in ordered_indices),
                basis=basis,
                energy=energy,
            )
        )
    groups.sort(
        key=lambda group: (
            -group.energy,
            -group.dimension,
            min(group.indices),
        )
    )
    return groups


def select_energy_ranked_subspace(
    model: DMDModel,
    training_scores: np.ndarray,
    target_dimension: int,
    nearly_real_tolerance: float = 1e-8,
    conjugate_tolerance: float = 1e-6,
) -> InvariantSubspace:
    """Select the maximum-energy exact-dimensional proper group combination."""
    q = int(model.operator.shape[0])
    if target_dimension <= 0:
        raise ValueError("target_dimension must be positive")
    if target_dimension >= q:
        raise ValueError(
            "target_dimension must be smaller than q; k=q would make every projector identity"
        )
    groups = energy_ranked_eigengroups(
        model,
        training_scores,
        nearly_real_tolerance=nearly_real_tolerance,
        conjugate_tolerance=conjugate_tolerance,
    )

    # Dynamic programming avoids splitting a conjugate pair while finding the
    # maximum-energy exact-dimensional combination.  Iteration order supplies a
    # deterministic preference for earlier energy-ranked groups at exact ties.
    choices: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for group_index, group in enumerate(groups):
        updated = dict(choices)
        for dimension, (energy, selected) in choices.items():
            new_dimension = dimension + group.dimension
            if new_dimension > target_dimension:
                continue
            candidate = (energy + group.energy, selected + (group_index,))
            current = updated.get(new_dimension)
            if current is None or candidate[0] > current[0] + np.finfo(float).eps:
                updated[new_dimension] = candidate
        choices = updated
    if target_dimension not in choices:
        raise ValueError(
            f"Cannot form a {target_dimension}-dimensional real invariant subspace "
            "without splitting a conjugate pair"
        )
    energy, selected_indices = choices[target_dimension]
    selected_groups = tuple(groups[index] for index in selected_indices)
    basis = _canonical_real_basis(
        np.column_stack([group.basis for group in selected_groups]),
        expected_dimension=target_dimension,
    )
    projector = orthogonal_projector(basis)
    return InvariantSubspace(
        groups=selected_groups,
        basis=basis,
        projector=projector,
        captured_energy=float(energy),
    )


def match_eigenvalues(
    reference: Iterable[complex],
    candidate: Iterable[complex],
) -> EigenvalueMatch:
    """Match a reference spectrum to a candidate spectrum by relative distance."""
    reference = np.asarray(list(reference), dtype=np.complex128)
    candidate = np.asarray(list(candidate), dtype=np.complex128)
    if reference.ndim != 1 or candidate.ndim != 1 or reference.size == 0:
        raise ValueError("Reference and candidate spectra must be nonempty one-dimensional arrays")
    if reference.size > candidate.size:
        raise ValueError("Candidate spectrum has fewer eigenvalues than the reference")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("Spectra must be finite")
    costs = np.abs(reference[:, None] - candidate[None, :]) / np.maximum(
        1.0, np.abs(reference[:, None])
    )
    reference_indices, candidate_indices = scipy.optimize.linear_sum_assignment(costs)
    order = np.argsort(reference_indices, kind="stable")
    reference_indices = reference_indices[order]
    candidate_indices = candidate_indices[order]
    return EigenvalueMatch(
        reference_indices=reference_indices.astype(int),
        candidate_indices=candidate_indices.astype(int),
        relative_distances=costs[reference_indices, candidate_indices].astype(float),
    )


def _basis_similarity(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape[0] != second.shape[0] or first.shape[1] != second.shape[1]:
        raise ValueError("Eigenspaces must have equal ambient and subspace dimensions")
    return float(np.clip(np.linalg.norm(first.T @ second, ord="fro") ** 2 / first.shape[1], 0, 1))


def match_eigengroups(
    reference_groups: Iterable[EigenGroup],
    candidate_groups: Iterable[EigenGroup],
    eigenvalue_weight: float = 1.0,
    subspace_weight: float = 0.25,
) -> list[EigenGroupMatch]:
    """Match fixed reference groups without re-ranking candidate modes.

    Only equal-dimensional real groups are compatible.  Thus a resolved
    complex pair is never silently matched to one real eigenvector.
    """
    reference_groups = list(reference_groups)
    candidate_groups = list(candidate_groups)
    if not reference_groups or len(reference_groups) > len(candidate_groups):
        raise ValueError("Candidate eigengroups must be at least as numerous as references")
    if eigenvalue_weight < 0 or subspace_weight < 0 or not (eigenvalue_weight + subspace_weight):
        raise ValueError("Matching weights must be nonnegative and not both zero")
    incompatible = 1e12
    costs = np.full((len(reference_groups), len(candidate_groups)), incompatible, dtype=float)
    spectral = np.full_like(costs, np.nan)
    similarities = np.full_like(costs, np.nan)
    for row, reference in enumerate(reference_groups):
        for column, candidate in enumerate(candidate_groups):
            if reference.dimension != candidate.dimension:
                continue
            value_match = match_eigenvalues(reference.eigenvalues, candidate.eigenvalues)
            spectral_distance = float(np.mean(value_match.relative_distances))
            similarity = _basis_similarity(reference.basis, candidate.basis)
            spectral[row, column] = spectral_distance
            similarities[row, column] = similarity
            costs[row, column] = (
                eigenvalue_weight * spectral_distance + subspace_weight * (1.0 - similarity)
            )
    reference_indices, candidate_indices = scipy.optimize.linear_sum_assignment(costs)
    if np.any(costs[reference_indices, candidate_indices] >= incompatible):
        raise ValueError("No dimension-compatible one-to-one eigengroup assignment exists")
    order = np.argsort(reference_indices, kind="stable")
    matches: list[EigenGroupMatch] = []
    for position in order:
        row = int(reference_indices[position])
        column = int(candidate_indices[position])
        matches.append(
            EigenGroupMatch(
                reference_group=row,
                candidate_group=column,
                spectral_distance=float(spectral[row, column]),
                subspace_similarity=float(similarities[row, column]),
                total_cost=float(costs[row, column]),
            )
        )
    return matches


def match_subspace_to_reference(
    reference: InvariantSubspace,
    candidate_model: DMDModel,
    candidate_training_scores: np.ndarray,
    eigenvalue_weight: float = 1.0,
    subspace_weight: float = 0.25,
    nearly_real_tolerance: float = 1e-8,
    conjugate_tolerance: float = 1e-6,
) -> tuple[InvariantSubspace, list[EigenGroupMatch]]:
    """Match a candidate fit to development-selected fixed reference groups."""
    candidate_groups = energy_ranked_eigengroups(
        candidate_model,
        candidate_training_scores,
        nearly_real_tolerance=nearly_real_tolerance,
        conjugate_tolerance=conjugate_tolerance,
    )
    matches = match_eigengroups(
        reference.groups,
        candidate_groups,
        eigenvalue_weight=eigenvalue_weight,
        subspace_weight=subspace_weight,
    )
    matched_groups = tuple(candidate_groups[match.candidate_group] for match in matches)
    basis = _canonical_real_basis(
        np.column_stack([group.basis for group in matched_groups]),
        expected_dimension=reference.dimension,
    )
    projector = orthogonal_projector(basis)
    return (
        InvariantSubspace(
            groups=matched_groups,
            basis=basis,
            projector=projector,
            captured_energy=float(sum(group.energy for group in matched_groups)),
        ),
        matches,
    )


def moving_block_bootstrap_pairs(
    n_samples: int,
    lag: int,
    block_length: int,
    seed: int,
    target_pairs: int | None = None,
) -> BlockBootstrapPairs:
    """Draw moving blocks and return only within-block lagged source pairs."""
    if n_samples <= 0 or lag <= 0 or block_length <= 0:
        raise ValueError("n_samples, lag, and block_length must be positive")
    if block_length > n_samples:
        raise ValueError("block_length cannot exceed n_samples")
    if block_length <= lag:
        raise ValueError("block_length must exceed lag to contain a legal pair")
    if target_pairs is None:
        target_pairs = n_samples - lag
    if target_pairs <= 0:
        raise ValueError("target_pairs must be positive")
    rng = np.random.default_rng(seed)
    predictor_parts: list[np.ndarray] = []
    response_parts: list[np.ndarray] = []
    block_id_parts: list[np.ndarray] = []
    bounds: list[tuple[int, int]] = []
    remaining = int(target_pairs)
    block_id = 0
    while remaining:
        pair_count = min(block_length - lag, remaining)
        sample_count = pair_count + lag
        start = int(rng.integers(0, n_samples - sample_count + 1))
        stop = start + sample_count
        predictors = np.arange(start, start + pair_count, dtype=int)
        predictor_parts.append(predictors)
        response_parts.append(predictors + lag)
        block_id_parts.append(np.full(pair_count, block_id, dtype=int))
        bounds.append((start, stop))
        remaining -= pair_count
        block_id += 1
    return BlockBootstrapPairs(
        predictor_indices=np.concatenate(predictor_parts),
        response_indices=np.concatenate(response_parts),
        block_ids=np.concatenate(block_id_parts),
        block_bounds=tuple(bounds),
    )


def bootstrap_segments(values: np.ndarray, sample: BlockBootstrapPairs) -> list[np.ndarray]:
    """Materialize sampled contiguous blocks for :func:`model.fit_ridge_dmd`."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values must be coordinate by time")
    segments: list[np.ndarray] = []
    for start, stop in sample.block_bounds:
        if start < 0 or stop > values.shape[1] or start >= stop:
            raise ValueError("Bootstrap block lies outside the source series")
        segments.append(values[:, start:stop])
    return segments


def within_bout_circular_shift_indices(
    n_rows: int,
    bout_start: int,
    bout_stop: int,
    target_start: int,
    target_stop: int,
    seed: int,
    minimum_shift: int = 1,
) -> CircularShiftSample:
    """Generate independent circular source indices without leaving one bout."""
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if not (0 <= bout_start < bout_stop):
        raise ValueError("Bout bounds must be ordered nonnegative indices")
    if not (bout_start <= target_start < target_stop <= bout_stop):
        raise ValueError("Target interval must lie completely inside the source bout")
    length = bout_stop - bout_start
    if minimum_shift < 0:
        raise ValueError("minimum_shift cannot be negative")
    offsets = np.arange(length, dtype=int)
    circular_distance = np.minimum(offsets, length - offsets)
    allowed = offsets[circular_distance >= minimum_shift]
    if not allowed.size:
        raise ValueError("No circular shift satisfies minimum_shift in this bout")
    rng = np.random.default_rng(seed)
    shifts = rng.choice(allowed, size=n_rows, replace=True)
    target_local = np.arange(target_start, target_stop, dtype=int) - bout_start
    source = bout_start + (target_local[None, :] + shifts[:, None]) % length
    return CircularShiftSample(
        shifts=shifts.astype(int),
        source_indices=source.astype(int),
        bout_start=bout_start,
        bout_stop=bout_stop,
    )


def apply_rowwise_indices(values: np.ndarray, source_indices: np.ndarray) -> np.ndarray:
    """Gather one independently indexed time sequence for each matrix row."""
    values = np.asarray(values)
    source_indices = np.asarray(source_indices, dtype=int)
    if values.ndim != 2 or source_indices.ndim != 2:
        raise ValueError("values and source_indices must both be two-dimensional")
    if values.shape[0] != source_indices.shape[0]:
        raise ValueError("Each value row requires one source-index row")
    if source_indices.size and (
        np.min(source_indices) < 0 or np.max(source_indices) >= values.shape[1]
    ):
        raise ValueError("source_indices lie outside values")
    return np.take_along_axis(values, source_indices, axis=1)


def stationary_iid_null(
    values: np.ndarray,
    seed: int,
    n_samples: int | None = None,
) -> StationaryNullSample:
    """Draw independent time samples from each row's empirical marginal.

    Observation-specific smoothing or calcium filtering should be applied after
    this event-level null so the real and null pipelines use exactly the same
    fixed filter.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("values must be row by nonzero time")
    if not np.all(np.isfinite(values)):
        raise ValueError("Stationary-null source values must be finite")
    if n_samples is None:
        n_samples = values.shape[1]
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.shape[1], size=(values.shape[0], int(n_samples)))
    return StationaryNullSample(
        values=apply_rowwise_indices(values, indices),
        source_indices=indices.astype(int),
    )


__all__ = [
    "BlockBootstrapPairs",
    "CircularShiftSample",
    "EigenGroup",
    "EigenGroupMatch",
    "EigenvalueMatch",
    "InvariantSubspace",
    "StationaryNullSample",
    "apply_rowwise_indices",
    "bootstrap_segments",
    "energy_ranked_eigengroups",
    "match_eigengroups",
    "match_eigenvalues",
    "match_subspace_to_reference",
    "moving_block_bootstrap_pairs",
    "orthogonal_projector",
    "projector_distance",
    "projector_similarity",
    "select_energy_ranked_subspace",
    "stationary_iid_null",
    "within_bout_circular_shift_indices",
]
