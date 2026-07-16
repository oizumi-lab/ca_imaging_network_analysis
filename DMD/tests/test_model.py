"""Tests for stable ridge DMD, legal pairs, forecasts, and spectra."""

from __future__ import annotations

import unittest

import numpy as np

from dmd_validation.model import (
    eigenvalue_diagnostics,
    fit_diagonal_ar,
    fit_ridge_dmd,
    rolling_forecast_metrics,
    snapshot_pairs,
)


class ModelTests(unittest.TestCase):
    def test_snapshot_pairs_do_not_cross_segment_joins(self) -> None:
        first = np.asarray([[0.0, 1.0, 2.0]])
        second = np.asarray([[100.0, 101.0, 102.0]])
        x, y, counts = snapshot_pairs([first, second], lag=1)
        np.testing.assert_array_equal(x, [[0.0, 1.0, 100.0, 101.0]])
        np.testing.assert_array_equal(y, [[1.0, 2.0, 101.0, 102.0]])
        self.assertEqual(counts, [2, 2])

    def test_exact_linear_recovery(self) -> None:
        rng = np.random.default_rng(2)
        operator = np.asarray([[0.9, -0.2], [0.1, 0.8]])
        # Independent short segments excite the full state space while each
        # legal pair still obeys the same operator exactly.
        series = []
        for _ in range(100):
            initial = rng.standard_normal(2)
            series.append(np.column_stack([initial, operator @ initial]))
        model = fit_ridge_dmd(series, lag=1, eta=0.0)
        np.testing.assert_allclose(model.operator, operator, atol=1e-10)

    def test_relative_ridge_is_invariant_to_global_scaling(self) -> None:
        rng = np.random.default_rng(3)
        series = rng.standard_normal((5, 100))
        first = fit_ridge_dmd([series], lag=2, eta=0.01)
        second = fit_ridge_dmd([17.0 * series], lag=2, eta=0.01)
        np.testing.assert_allclose(first.operator, second.operator, atol=1e-11)

    def test_rotation_formula_matches_known_value(self) -> None:
        radius, angle = 0.9, 0.3
        operator = radius * np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        rng = np.random.default_rng(5)
        initial = rng.standard_normal(2)
        series = [initial]
        for _ in range(300):
            series.append(operator @ series[-1])
        model = fit_ridge_dmd([np.column_stack(series)], lag=1, eta=0.0)
        table = eigenvalue_diagnostics(model, fs_effective=10.0, window_samples=301)
        row = table.loc[table["positive_conjugate_member"]].iloc[0]
        expected = angle * np.log(10) / (-2 * np.pi * np.log(radius))
        self.assertAlmostEqual(row["rotations_per_decade"], expected, places=8)

    def test_multi_step_forecast_is_exact_for_linear_series(self) -> None:
        operator = np.diag([0.95, 0.8])
        series = np.empty((2, 80))
        series[:, 0] = [2.0, -1.0]
        for index in range(1, series.shape[1]):
            series[:, index] = operator @ series[:, index - 1]
        model = fit_ridge_dmd([series[:, :60]], lag=1, eta=0.0)
        diagonal = fit_diagonal_ar([series[:, :60]], lag=1, eta=0.0)
        rows = rolling_forecast_metrics(
            model,
            diagonal,
            series,
            train_stop=60,
            fs_effective=10.0,
            horizons_seconds=[0.2, 0.8],
            explosion_multiplier=100.0,
        )
        self.assertEqual(len(rows), 2)
        self.assertLess(max(row["sse_dmd"] for row in rows), 1e-20)


if __name__ == "__main__":
    unittest.main()
