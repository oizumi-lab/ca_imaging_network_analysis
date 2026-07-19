"""Tests for the dataset adapter around the official Rastermap package."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.funcnet import rastermap_tools as rmt


class RastermapRowValidationTests(unittest.TestCase):
    def test_only_nonfinite_and_constant_rows_are_removed(self) -> None:
        activity = np.array(
            [
                [0, 1, 0, 2, 0, 1],
                [3, 3, 3, 3, 3, 3],
                [0, 1, np.nan, 2, 1, 0],
                [0, 0, 1, 0, 2, 0],
                [0, np.inf, 1, 0, 2, 0],
            ],
            dtype=float,
        )

        np.testing.assert_array_equal(
            rmt.valid_activity_rows(activity, chunk_frames=2),
            [True, False, False, True, False],
        )

    def test_activity_shape_and_chunk_size_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            rmt.valid_activity_rows(np.ones(5))
        with self.assertRaises(ValueError):
            rmt.valid_activity_rows(np.ones((2, 3)), chunk_frames=0)


class RastermapActivityRateTests(unittest.TestCase):
    def test_positive_bin_and_onset_rates_cross_chunk_boundaries(self) -> None:
        activity = np.array(
            [
                [0, 1, 1, 0, 2, 0, 3],
                [0, 0, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [-1, 0, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        )

        np.testing.assert_allclose(
            rmt.positive_deconvolution_bin_rates(activity, fs=2.0, chunk_frames=3),
            np.array([8, 6, 14, 0]) / 7,
        )
        np.testing.assert_array_equal(
            rmt.positive_deconvolution_bin_counts(activity, chunk_frames=3),
            [4, 3, 7, 0],
        )
        np.testing.assert_allclose(
            rmt.positive_deconvolution_onset_rates(
                activity,
                fs=2.0,
                chunk_frames=3,
            ),
            np.array([6, 2, 2, 0]) / 7,
        )

    def test_active_rows_apply_inclusive_rate_and_trace_validity(self) -> None:
        activity = np.array(
            [
                [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # exactly 0.2/s
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 0.1/s
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # active but constant
                [0, 1, 0, np.nan, 1, 0, 0, 0, 0, 0],
                [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0],
            ],
            dtype=float,
        )

        np.testing.assert_array_equal(
            rmt.active_deconvolution_rows(
                activity,
                fs=1.0,
                min_positive_bin_rate_hz=0.2,
                chunk_frames=3,
            ),
            [True, False, False, False, False],
        )
        np.testing.assert_array_equal(
            rmt.active_deconvolution_count_rows(
                activity,
                min_positive_bins=2,
                chunk_frames=3,
            ),
            [True, False, False, False, False],
        )

    def test_activity_rate_arguments_are_validated(self) -> None:
        activity = np.ones((2, 3))
        with self.assertRaises(ValueError):
            rmt.positive_deconvolution_bin_rates(activity, fs=0)
        with self.assertRaises(TypeError):
            rmt.positive_deconvolution_onset_rates(activity, fs=True)
        with self.assertRaises(ValueError):
            rmt.active_deconvolution_rows(
                activity,
                fs=1,
                min_positive_bin_rate_hz=-0.1,
            )
        with self.assertRaises(TypeError):
            rmt.active_deconvolution_count_rows(
                activity,
                min_positive_bins=True,
            )


class RastermapOrderMetricTests(unittest.TestCase):
    def test_rank_correlation_treats_reversal_as_equivalent(self) -> None:
        reference = np.array([0, 1, 2, np.nan, 3, 4], dtype=float)
        reversed_embedding = np.array([4, 3, 2, 99, 1, 0], dtype=float)

        self.assertAlmostEqual(
            rmt.reversal_invariant_rank_correlation(
                reference,
                reversed_embedding,
            ),
            1.0,
        )

    def test_rank_neighborhoods_are_reversal_invariant(self) -> None:
        reference = np.arange(20, dtype=float)
        self.assertAlmostEqual(
            rmt.rank_neighborhood_overlap(
                reference,
                -reference,
                neighborhood_size=5,
            ),
            1.0,
        )

    def test_ties_do_not_get_a_shared_row_index_advantage(self) -> None:
        tied = np.repeat(np.arange(5, dtype=float), 8)

        overlap = rmt.rank_neighborhood_overlap(
            tied,
            tied,
            neighborhood_size=4,
            tie_permutations=32,
            random_state=3,
        )

        self.assertGreater(overlap, 0.3)
        self.assertLess(overlap, 0.95)

    def test_scrambled_order_has_low_neighborhood_overlap(self) -> None:
        reference = np.arange(40, dtype=float)
        scrambled = np.random.default_rng(5).permutation(reference)

        overlap = rmt.rank_neighborhood_overlap(
            reference,
            scrambled,
            neighborhood_size=4,
        )

        self.assertLess(overlap, 0.4)

    def test_order_metric_arguments_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            rmt.reversal_invariant_rank_correlation(
                np.arange(4),
                np.arange(5),
            )
        with self.assertRaises(ValueError):
            rmt.rank_neighborhood_overlap(
                np.arange(4),
                np.arange(4),
                neighborhood_size=4,
            )
        with self.assertRaises(ValueError):
            rmt.rank_neighborhood_overlap(
                np.arange(4),
                np.arange(4),
                neighborhood_size=2,
                tie_permutations=0,
            )


class RastermapFitTests(unittest.TestCase):
    def test_fit_maps_every_usable_row_back_to_original_roi_ids(self) -> None:
        rng = np.random.default_rng(12)
        activity = rng.gamma(shape=0.8, scale=1.0, size=(24, 120)).astype(np.float32)
        activity[3] = 0
        activity[17, 9] = np.nan

        result = rmt.fit_all_neurons(
            activity,
            n_clusters=8,
            n_PCs=8,
            locality=0.0,
            time_lag_window=2,
            mean_time=True,
            time_bin=1,
            superneuron_size=4,
            random_state=0,
            verbose=False,
        )

        expected_rows = np.setdiff1d(np.arange(24), [3, 17])
        np.testing.assert_array_equal(result.valid_rows, expected_rows)
        np.testing.assert_array_equal(np.sort(result.isort), expected_rows)
        self.assertEqual(np.unique(result.isort).size, expected_rows.size)
        self.assertEqual(result.embedding.shape, (24,))
        self.assertTrue(np.isnan(result.embedding[[3, 17]]).all())
        self.assertEqual(
            result.X_embedding.shape,
            ((expected_rows.size + 3) // 4, 120),
        )

    def test_selected_fit_keeps_nonselected_rois_out_of_the_embedding(self) -> None:
        rng = np.random.default_rng(21)
        activity = rng.gamma(shape=0.8, scale=1.0, size=(30, 140)).astype(np.float32)
        activity[7] = 0
        selected_rows = np.array([2, 5, 7, 11, 18, 23, 29])

        result = rmt.fit_selected_neurons(
            activity,
            selected_rows,
            n_clusters=3,
            n_PCs=4,
            locality=0.0,
            time_lag_window=2,
            mean_time=True,
            time_bin=1,
            superneuron_size=3,
            random_state=0,
            verbose=False,
        )

        expected_rows = np.setdiff1d(selected_rows, [7])
        np.testing.assert_array_equal(np.sort(result.valid_rows), expected_rows)
        np.testing.assert_array_equal(np.sort(result.isort), expected_rows)
        self.assertTrue(
            np.isnan(result.embedding[np.setdiff1d(np.arange(30), expected_rows)]).all()
        )
        self.assertTrue(np.isfinite(result.embedding[expected_rows]).all())

    def test_superneurons_include_the_final_partial_group(self) -> None:
        activity = np.array(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 2, 0, 2],
                [2, 0, 2, 0],
                [0, 4, 0, 4],
            ],
            dtype=np.float32,
        )
        grouped = rmt.ordered_superneurons(
            activity,
            order=np.arange(5),
            superneuron_size=2,
        )

        self.assertEqual(grouped.shape, (3, 4))
        # The final group is the fifth neuron alone, standardized across time.
        np.testing.assert_allclose(grouped[-1], [-1, 1, -1, 1])


class RastermapCacheTests(unittest.TestCase):
    def test_cache_is_used_only_for_exactly_matching_metadata(self) -> None:
        metadata = rmt.make_cache_metadata(
            recording_name="synthetic",
            n_neurons=4,
            n_frames=12,
            fs=3.0,
            parameters={"n_clusters": 2, "time_bin": 1},
            neuron_selection={
                "definition": "positive_deconvolution_bin_rate",
                "minimum_rate_hz": 0.1,
            },
        )
        result = rmt.RastermapResult(
            X_embedding=np.arange(12, dtype=np.float32).reshape(1, 12),
            embedding=np.arange(4, dtype=np.float32)[:, None].ravel(),
            isort=np.array([2, 0, 3, 1]),
            valid_rows=np.arange(4),
            runtime_seconds=1.25,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fit.npz"
            rmt.save_cache(path, result, metadata)
            loaded = rmt.load_cache(path, metadata)
            self.assertIsNotNone(loaded)
            np.testing.assert_array_equal(loaded.isort, result.isort)
            np.testing.assert_array_equal(loaded.X_embedding, result.X_embedding)
            self.assertEqual(loaded.runtime_seconds, 1.25)

            changed = dict(metadata)
            changed["n_frames"] = 13
            self.assertIsNone(rmt.load_cache(path, changed))

            changed_selection = dict(metadata)
            changed_selection["neuron_selection"] = {
                "definition": "positive_deconvolution_bin_rate",
                "minimum_rate_hz": 0.05,
            }
            self.assertIsNone(rmt.load_cache(path, changed_selection))


if __name__ == "__main__":
    unittest.main()
