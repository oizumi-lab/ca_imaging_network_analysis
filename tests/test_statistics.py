"""Unit tests for shared statistical summaries."""

from __future__ import annotations

import unittest

import numpy as np

from src.funcnet.statistics import mean_confidence_interval


class MeanConfidenceIntervalTests(unittest.TestCase):
    def test_nan_aware_interval_uses_per_column_student_t_quantiles(self) -> None:
        values = np.array(
            [
                [1.0, 10.0, np.nan, np.nan],
                [2.0, np.nan, np.nan, np.nan],
                [3.0, 14.0, 9.0, np.nan],
            ]
        )

        mean, lower, upper = mean_confidence_interval(values)

        np.testing.assert_allclose(mean, [2.0, 12.0, 9.0, np.nan], equal_nan=True)
        # The first two margins use t(df=2) and t(df=1), respectively.  The
        # deliberately wide second interval catches accidental normal CIs.
        np.testing.assert_allclose(
            lower,
            [-0.4841377117503303, -13.41240947234939, np.nan, np.nan],
            equal_nan=True,
        )
        np.testing.assert_allclose(
            upper,
            [4.48413771175033, 37.41240947234939, np.nan, np.nan],
            equal_nan=True,
        )

    def test_axis_none_summarizes_every_valid_observation(self) -> None:
        mean, lower, upper = mean_confidence_interval(
            np.array([[1.0, np.nan], [3.0, 5.0]]),
            axis=None,
        )

        self.assertEqual(float(mean), 3.0)
        self.assertLess(float(lower), float(mean))
        self.assertGreater(float(upper), float(mean))

    def test_confidence_level_must_be_open_interval(self) -> None:
        for confidence in (0.0, 1.0, -0.1, 1.1):
            with self.subTest(confidence=confidence):
                with self.assertRaises(ValueError):
                    mean_confidence_interval([1, 2, 3], confidence=confidence)


if __name__ == "__main__":
    unittest.main()
