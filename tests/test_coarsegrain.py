"""Tests for spatial coarse-graining measures extracted from tutorials."""

from __future__ import annotations

import unittest

import numpy as np

from src.funcnet.coarsegrain import module_localization_index


class ModuleLocalizationIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        # Two well-separated nearest-neighbour pairs avoid ambiguous ties.
        self.coords = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [10.0, 0.0],
                [10.0, 1.0],
            ]
        )

    def test_local_pairs_score_above_label_frequency_chance(self) -> None:
        # All four nearest-neighbour relationships agree.  With two equally
        # frequent modules, chance agreement is 1/2, giving an index of 2.
        score = module_localization_index(self.coords, np.array([0, 0, 1, 1]))

        self.assertEqual(score, 2.0)

    def test_interleaved_labels_have_zero_nearest_neighbour_agreement(self) -> None:
        score = module_localization_index(self.coords, np.array([0, 1, 0, 1]))

        self.assertEqual(score, 0.0)

    def test_shape_and_minimum_size_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            module_localization_index(np.zeros((4, 3)), np.arange(4))
        with self.assertRaises(ValueError):
            module_localization_index(self.coords, np.arange(3))
        with self.assertRaises(ValueError):
            module_localization_index(np.array([[0.0, 0.0]]), np.array([0]))


if __name__ == "__main__":
    unittest.main()
