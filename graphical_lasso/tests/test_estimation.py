"""Numerical regression tests for the specialized graphical-lasso code."""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.covariance import graphical_lasso

from glasso_analysis.comparison import exact_fixed_density
from glasso_analysis.estimation import (
    empirical_correlation,
    fit_screened_graphical_lasso,
    precision_to_partial,
    standardize_activity,
)


class EstimationTests(unittest.TestCase):
    def test_standardization_reconstructs_pearson(self) -> None:
        rng = np.random.default_rng(4)
        activity = rng.normal(size=(8, 50))
        observed = empirical_correlation(standardize_activity(activity))
        expected = np.corrcoef(activity)
        np.testing.assert_allclose(observed, expected, atol=1e-12)

    def test_quic_matches_sklearn_on_small_problem(self) -> None:
        rng = np.random.default_rng(9)
        samples = rng.normal(size=(300, 12))
        sample = np.cov(samples, rowvar=False, bias=True)
        alpha = 0.08
        covariance_ref, precision_ref = graphical_lasso(
            sample, alpha=alpha, tol=1e-9, enet_tol=1e-9, max_iter=500
        )
        precision, covariance, diagnostics = fit_screened_graphical_lasso(
            sample, alpha, tol=1e-9, max_iter=500
        )
        np.testing.assert_allclose(precision, precision_ref, atol=2e-7)
        np.testing.assert_allclose(covariance, covariance_ref, atol=2e-7)
        self.assertLess(diagnostics.max_inverse_error, 1e-10)

    def test_partial_correlation_sign(self) -> None:
        precision = np.array([[2.0, -0.5], [-0.5, 1.0]])
        partial = precision_to_partial(precision)
        self.assertAlmostEqual(partial[0, 1], 0.5 / np.sqrt(2.0))
        np.testing.assert_array_equal(np.diag(partial), np.ones(2))

    def test_exact_density_never_pads_sparse_support(self) -> None:
        matrix = np.eye(5)
        matrix[0, 1] = matrix[1, 0] = 0.7
        with self.assertRaises(ValueError):
            exact_fixed_density(matrix, 0.3, require_nonzero=True)
        selected = exact_fixed_density(matrix, 0.1, require_nonzero=True)
        self.assertEqual(selected.adjacency.nnz // 2, 1)


if __name__ == "__main__":
    unittest.main()

