"""Tests for proper invariant subspaces, bootstrap blocks, and null controls."""

from __future__ import annotations

import unittest

import numpy as np
import scipy.linalg

from dmd_validation.model import DMDModel, snapshot_pairs
from dmd_validation.stability import (
    apply_rowwise_indices,
    bootstrap_segments,
    energy_ranked_eigengroups,
    match_eigenvalues,
    match_subspace_to_reference,
    moving_block_bootstrap_pairs,
    orthogonal_projector,
    projector_distance,
    projector_similarity,
    select_energy_ranked_subspace,
    stationary_iid_null,
    within_bout_circular_shift_indices,
)


def _model(operator: np.ndarray) -> DMDModel:
    eigenvalues, eigenvectors = scipy.linalg.eig(operator)
    return DMDModel(
        operator=np.asarray(operator, dtype=float),
        lag=1,
        alpha=0.0,
        eta=0.0,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        diagnostics={},
    )


def _rotation(radius: float, angle: float) -> np.ndarray:
    return radius * np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )


class StabilityTests(unittest.TestCase):
    def test_energy_ranking_groups_conjugates_into_real_subspace(self) -> None:
        operator = scipy.linalg.block_diag(_rotation(0.9, 0.4), 0.8, 0.6)
        scores = np.vstack(
            [
                np.linspace(-20.0, 20.0, 100),
                np.sin(np.linspace(0.0, 12.0, 100)) * 15.0,
                np.ones(100),
                np.linspace(-0.1, 0.1, 100),
            ]
        )
        groups = energy_ranked_eigengroups(_model(operator), scores)
        self.assertEqual(groups[0].dimension, 2)
        self.assertEqual(len(groups[0].indices), 2)
        np.testing.assert_allclose(
            groups[0].basis @ groups[0].basis.T,
            np.diag([1.0, 1.0, 0.0, 0.0]),
            atol=1e-12,
        )

    def test_full_space_projectors_are_rejected(self) -> None:
        operator = np.diag([0.9, 0.8, 0.7, 0.6])
        scores = np.eye(4)
        with self.assertRaisesRegex(ValueError, "smaller than q"):
            select_energy_ranked_subspace(_model(operator), scores, target_dimension=4)
        with self.assertRaisesRegex(ValueError, "full-space"):
            orthogonal_projector(np.eye(4))

    def test_projector_similarity_is_basis_invariant(self) -> None:
        first_basis = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
        )
        within_basis_rotation = _rotation(1.0, 0.71)
        same_basis = first_basis @ within_basis_rotation
        orthogonal_basis = np.asarray(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        )
        first = orthogonal_projector(first_basis)
        same = orthogonal_projector(same_basis)
        orthogonal = orthogonal_projector(orthogonal_basis)
        self.assertAlmostEqual(projector_similarity(first, same), 1.0)
        self.assertAlmostEqual(projector_distance(first, same), 0.0)
        self.assertAlmostEqual(projector_similarity(first, orthogonal), 0.0)
        self.assertAlmostEqual(projector_distance(first, orthogonal), 1.0)

    def test_eigenvalue_matching_handles_reordered_candidates(self) -> None:
        reference = np.asarray([0.9 + 0.2j, 0.9 - 0.2j, 0.7])
        candidate = np.asarray([0.699, 0.901 - 0.199j, 0.901 + 0.199j, 0.2])
        match = match_eigenvalues(reference, candidate)
        np.testing.assert_array_equal(match.candidate_indices, [2, 1, 0])
        self.assertLess(float(np.max(match.relative_distances)), 0.002)

    def test_candidate_modes_match_fixed_reference_groups(self) -> None:
        reference_operator = scipy.linalg.block_diag(_rotation(0.90, 0.40), 0.8, 0.6)
        candidate_operator = scipy.linalg.block_diag(_rotation(0.91, 0.39), 0.79, 0.59)
        scores = np.vstack(
            [
                10 * np.sin(np.linspace(0, 10, 120)),
                10 * np.cos(np.linspace(0, 10, 120)),
                np.linspace(-1, 1, 120),
                np.linspace(-0.2, 0.2, 120),
            ]
        )
        reference = select_energy_ranked_subspace(
            _model(reference_operator), scores, target_dimension=2
        )
        candidate, matches = match_subspace_to_reference(
            reference,
            _model(candidate_operator),
            scores,
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(candidate.dimension, 2)
        self.assertAlmostEqual(
            projector_similarity(reference.projector, candidate.projector), 1.0, places=12
        )

    def test_moving_block_pairs_never_cross_joins(self) -> None:
        sample = moving_block_bootstrap_pairs(
            n_samples=20,
            lag=3,
            block_length=7,
            seed=41,
            target_pairs=17,
        )
        repeated = moving_block_bootstrap_pairs(20, 3, 7, 41, target_pairs=17)
        np.testing.assert_array_equal(sample.predictor_indices, repeated.predictor_indices)
        np.testing.assert_array_equal(sample.response_indices, repeated.response_indices)
        np.testing.assert_array_equal(sample.response_indices - sample.predictor_indices, 3)
        self.assertEqual(sample.predictor_indices.size, 17)
        for block_id, (start, stop) in enumerate(sample.block_bounds):
            selected = sample.block_ids == block_id
            self.assertTrue(np.all(sample.predictor_indices[selected] >= start))
            self.assertTrue(np.all(sample.response_indices[selected] < stop))

        values = np.vstack([np.arange(20), 100 + np.arange(20)])
        segments = bootstrap_segments(values, sample)
        x, y, counts = snapshot_pairs(segments, lag=3)
        self.assertEqual(sum(counts), 17)
        np.testing.assert_array_equal(y - x, np.full_like(x, 3))

    def test_circular_surrogate_stays_inside_bout_and_is_deterministic(self) -> None:
        sample = within_bout_circular_shift_indices(
            n_rows=5,
            bout_start=10,
            bout_stop=30,
            target_start=12,
            target_stop=18,
            seed=13,
            minimum_shift=3,
        )
        repeated = within_bout_circular_shift_indices(5, 10, 30, 12, 18, 13, 3)
        np.testing.assert_array_equal(sample.shifts, repeated.shifts)
        np.testing.assert_array_equal(sample.source_indices, repeated.source_indices)
        circular_distance = np.minimum(sample.shifts, 20 - sample.shifts)
        self.assertTrue(np.all(circular_distance >= 3))
        self.assertTrue(np.all(sample.source_indices >= 10))
        self.assertTrue(np.all(sample.source_indices < 30))
        local_steps = np.diff(sample.source_indices - 10, axis=1) % 20
        np.testing.assert_array_equal(local_steps, np.ones_like(local_steps))

        source = np.repeat(np.arange(40, dtype=float)[None, :], 5, axis=0)
        np.testing.assert_array_equal(
            apply_rowwise_indices(source, sample.source_indices), sample.source_indices
        )

    def test_stationary_null_draws_only_from_each_rows_empirical_marginal(self) -> None:
        values = np.asarray(
            [
                [0.0, 0.0, 1.0, 2.0],
                [10.0, 20.0, 30.0, 40.0],
                [-3.0, -2.0, -1.0, 0.0],
            ]
        )
        sample = stationary_iid_null(values, seed=22, n_samples=40)
        repeated = stationary_iid_null(values, seed=22, n_samples=40)
        np.testing.assert_array_equal(sample.values, repeated.values)
        np.testing.assert_array_equal(sample.source_indices, repeated.source_indices)
        self.assertEqual(sample.values.shape, (3, 40))
        for row in range(values.shape[0]):
            self.assertTrue(set(sample.values[row]).issubset(set(values[row])))


if __name__ == "__main__":
    unittest.main()
