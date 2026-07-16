"""Tests for data-only preparation helpers used by visualization tutorials."""

from __future__ import annotations

import unittest

import numpy as np

from src.funcnet import visualization as viz


class ViewSelectionTests(unittest.TestCase):
    def test_time_limits_are_clipped_to_recording(self) -> None:
        self.assertEqual(viz.resolve_time_limits(12.5, None), (0.0, 12.5))
        self.assertEqual(viz.resolve_time_limits(12.5, (2, 20)), (2.0, 12.5))

        with self.assertRaises(ValueError):
            viz.resolve_time_limits(12.5, (12.5, 13))

    def test_trace_selection_is_seeded_and_sorted(self) -> None:
        selected = viz.select_trace_neurons(n_neurons=10, n_select=4, seed=3)

        # A literal sample distinguishes Generator sampling from legacy
        # RandomState sampling used by the network-analysis row selector.
        np.testing.assert_array_equal(selected, [0, 1, 2, 5])
        np.testing.assert_array_equal(
            viz.select_trace_neurons(n_neurons=3, n_select=20, seed=3),
            [0, 1, 2],
        )

    def test_visible_frame_range_and_labels_use_recorded_time(self) -> None:
        view = {
            "time_limits_min": (0.1, 0.2),
            "duration_min": 1.0,
            "fs": 2.0,
            "n_frames": 100,
        }

        self.assertEqual(viz.visible_frame_range(view), (12, 24))
        self.assertEqual(viz.time_window_label(view), "recorded minutes 0.1–0.2")

        view["time_limits_min"] = (0.0, 1.0)
        self.assertEqual(viz.time_window_label(view), "full recorded sequence")

    def test_nice_scale_bar_uses_one_two_five_series(self) -> None:
        self.assertEqual(viz.nice_scale_bar(6.0), 5.0)
        self.assertEqual(viz.nice_scale_bar(0.08), 0.05)
        self.assertEqual(viz.nice_scale_bar(0.0), 1.0)


class BinnedDisplayPreparationTests(unittest.TestCase):
    def test_spike_bins_restart_at_acquisition_and_state_boundaries(self) -> None:
        spikes = np.zeros((3, 10), dtype=float)
        spikes[0, [0, 1, 4]] = 1
        spikes[1, [1, 2, 6, 9]] = 1
        spikes[2, [3, 8]] = 1
        state = np.r_[np.zeros(7), np.ones(3)]

        raster, active_counts, bin_frames, order, centers = viz.binned_spike_raster(
            spikes,
            fs=1.0,
            bin_seconds=3.0,
            boundary_ind=np.array([3]),
            state=state,
        )

        # boundary_ind=3 means the first acquisition includes frame 3.  The
        # state transition at frame 7 also begins a fresh display bin.
        self.assertEqual(bin_frames, 3)
        np.testing.assert_allclose(centers, [1.0, 3.0, 5.0, 8.0])
        np.testing.assert_array_equal(order, [1, 0, 2])
        np.testing.assert_array_equal(
            raster,
            [
                [1, 0, 1, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
            ],
        )
        np.testing.assert_array_equal(
            active_counts,
            [1, 2, 1, 1, 1, 0, 1, 0, 1, 1],
        )

    def test_dff_panels_share_a_session_median_baseline(self) -> None:
        dff = np.array(
            [
                [0, 2, 4, 6, 8, 10],
                [10, 8, 6, 4, 2, 0],
            ],
            dtype=float,
        )
        state = np.array([0, 0, 0, 0, 1, 1], dtype=float)

        panels, color_limit, bin_frames = viz.binned_dff_heatmaps(
            dff,
            neuron_order=np.array([1, 0]),
            fs=1.0,
            bin_seconds=2.0,
            boundary_ind=np.array([2]),
            state=state,
        )

        self.assertEqual(bin_frames, 2)
        self.assertEqual(len(panels), 3)
        self.assertAlmostEqual(panels[0]["stop_min"], 3 / 60)
        self.assertAlmostEqual(panels[1]["start_min"], 3 / 60)
        self.assertAlmostEqual(panels[2]["start_min"], 4 / 60)
        np.testing.assert_allclose(panels[0]["values"], [[4, 1], [-4, -1]])
        np.testing.assert_allclose(panels[1]["values"], [[-1], [1]])
        np.testing.assert_allclose(panels[2]["values"], [[-4], [4]])
        self.assertEqual(color_limit, 4.0)


if __name__ == "__main__":
    unittest.main()
