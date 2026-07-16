"""Known-system simulation reproducibility and observation-shape tests."""

from __future__ import annotations

import unittest

import numpy as np

from dmd_validation.simulation import simulate_system


class SimulationTests(unittest.TestCase):
    def test_simulation_is_reproducible_and_matches_shapes(self) -> None:
        config = {
            "latent_dimension": 6,
            "n_frames": 100,
            "burn_in_frames": 20,
            "n_neurons": 30,
            "real_mode_time_constants_seconds": [0.4, 0.8, 1.5, 3.0, 6.0, 12.0],
            "rotation_decay_seconds": 3.0,
            "rotation_frequency_hz": 0.5,
            "latent_process_noise_sd": 0.15,
            "continuous_observation_noise_sd": 0.05,
            "smoothing_sigma_frames": 2.5,
            "calcium_decay_seconds": 0.25,
            "calcium_noise_fraction": 0.05,
        }
        first = simulate_system("rotational", 9, config, 7.65, 0.98)
        second = simulate_system("rotational", 9, config, 7.65, 0.98)
        self.assertEqual(first.latent.shape, (6, 100))
        for name, values in first.observations.items():
            self.assertEqual(values.shape, (30, 100), name)
            np.testing.assert_allclose(values, second.observations[name])


if __name__ == "__main__":
    unittest.main()
