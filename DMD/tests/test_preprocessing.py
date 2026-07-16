"""Tests for arm construction, train-only scaling, and frozen PCA."""

from __future__ import annotations

import unittest

import numpy as np

from dmd_validation.preprocessing import (
    apply_scaler,
    construct_arm_window,
    fit_scaler,
    randomized_pca,
)


class PreprocessingTests(unittest.TestCase):
    def test_b4_is_constructed_inside_each_window(self) -> None:
        first = np.arange(8, dtype=float)[None, :]
        second = (100 + np.arange(8, dtype=float))[None, :]
        binned_first = construct_arm_window(first, 4, fs_hz=4.0)
        binned_second = construct_arm_window(second, 4, fs_hz=4.0)
        np.testing.assert_array_equal(binned_first, [[6.0, 22.0]])
        np.testing.assert_array_equal(binned_second, [[406.0, 422.0]])

    def test_rms_scaling_matches_reference_rule_and_drops_constant(self) -> None:
        windows = [
            np.asarray([[1.0, 3.0], [5.0, 5.0], [0.0, 2.0]]),
            np.asarray([[5.0, 7.0], [5.0, 5.0], [4.0, 6.0]]),
        ]
        mean, scale, rows, audit = fit_scaler(windows, "rms")
        np.testing.assert_array_equal(rows, [0, 2])
        np.testing.assert_allclose(mean, [4.0, 3.0])
        expected_rms = np.sqrt(5.0) + 1e-3
        np.testing.assert_allclose(scale, [expected_rms, expected_rms])
        self.assertEqual(audit["constant_neurons"], 1)
        scaled = apply_scaler(windows, mean, scale, rows)
        pooled = np.concatenate(scaled, axis=1)
        np.testing.assert_allclose(np.mean(pooled, axis=1), 0.0, atol=1e-12)

    def test_randomized_pca_is_orthonormal_and_deterministic(self) -> None:
        rng = np.random.default_rng(4)
        windows = [rng.standard_normal((20, 30)), rng.standard_normal((20, 25))]
        result1 = randomized_pca(windows, 6, 4, 2, seed=8)
        result2 = randomized_pca(windows, 6, 4, 2, seed=8)
        components = result1[0]
        np.testing.assert_allclose(components.T @ components, np.eye(6), atol=1e-10)
        np.testing.assert_allclose(result1[0], result2[0])
        self.assertLess(result1[-1], 1e-10)


if __name__ == "__main__":
    unittest.main()
