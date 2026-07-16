"""Small deterministic integration test for the activity-to-modularity helper."""

from __future__ import annotations

import unittest

import numpy as np

from src.funcnet.network import modularity_from_activity


class ModularityFromActivityTests(unittest.TestCase):
    def test_tiny_activity_matrix_runs_complete_pipeline(self) -> None:
        # Nodes 0 and 1 are perfectly positively correlated, while node 2 is
        # perfectly anticorrelated.  Signed ranking at density 1/3 therefore
        # retains exactly the 0--1 edge and leaves node 2 isolated.
        activity = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0, 0.0],
            ]
        )

        result = modularity_from_activity(
            activity,
            density=1 / 3,
            n_runs=2,
            negative=False,
            seed=5,
            warm_start=True,
        )

        self.assertEqual(result["correlation_threshold"], 1.0)
        self.assertEqual(result["Q_all"].shape, (2,))
        self.assertEqual(result["ci_all"].shape, (3, 2))
        self.assertEqual(result["n_modules_max"], 2)
        self.assertEqual(result["ci_max"][0], result["ci_max"][1])
        self.assertNotEqual(result["ci_max"][0], result["ci_max"][2])
        np.testing.assert_allclose(result["Q_all"], 0.0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
