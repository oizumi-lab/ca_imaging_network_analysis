"""Tests for the focused plotting helpers used by the hands-on."""

from __future__ import annotations

import unittest

import numpy as np
from matplotlib.figure import Figure

from src.funcnet import visualization as viz


class BinnedSpikeRasterTests(unittest.TestCase):
    def test_bins_restart_at_acquisition_and_state_boundaries(self) -> None:
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

    def test_invalid_state_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            viz.binned_spike_raster(
                np.zeros((2, 4)),
                fs=1.0,
                bin_seconds=1.0,
                boundary_ind=np.array([], dtype=int),
                state=np.zeros(3),
            )


class StateStripTests(unittest.TestCase):
    def test_strip_uses_recorded_time_and_visible_states(self) -> None:
        fig = Figure()
        ax = fig.subplots()
        view = {
            "n_frames": 12,
            "fs": 1.0,
            "duration_min": 0.2,
            "time_limits_min": (0.0, 0.2),
            "state": np.r_[np.zeros(6), np.ones(6)],
            "codes": {0.0: "awake", 1.0: "nrem"},
            "boundary_minutes": [5 / 60],
        }

        viz.plot_state_strip(ax, view)

        self.assertEqual(tuple(ax.get_xlim()), (0.0, 0.2))
        self.assertEqual(len(ax.images), 1)
        self.assertEqual([text.get_text() for text in ax.get_legend().texts], ["Awake", "NREM sleep"])


class CorticalAreaColorTests(unittest.TestCase):
    def test_layer_suffixes_collapse_into_shared_anatomical_palette(self) -> None:
        labels = viz.cortical_region_labels(
            np.array(["MOs2/3", "SSp-bfd2/3", "RSPd2/3", "VISp2/3", "root"])
        )

        np.testing.assert_array_equal(
            labels,
            ["MOs", "SSp-bfd", "RSPd", "VISp", "Unassigned"],
        )
        self.assertEqual(viz.CORTICAL_REGION_COLORS["MOs"], "#005a32")
        self.assertEqual(viz.CORTICAL_REGION_COLORS["SSp-bfd"], "#fb6a4a")
        self.assertEqual(viz.CORTICAL_REGION_COLORS["RSPd"], "#2b8cbe")
        self.assertEqual(viz.CORTICAL_REGION_COLORS["VISp"], "#6a51a3")

    def test_unabridged_legend_uses_anatomical_names(self) -> None:
        handles = viz.cortical_region_legend_handles(
            ["MOp", "VISa"],
            unabridged=True,
        )

        self.assertEqual(
            [handle.get_label() for handle in handles],
            ["MOp — Primary motor area", "VISa — Anterior visual area"],
        )


class SpatialModulePlotTests(unittest.TestCase):
    def test_plot_reports_module_and_node_counts(self) -> None:
        fig = Figure()
        ax = fig.subplots()
        collection = viz.plot_spatial_modules(
            ax,
            np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
            np.array([1, 1, 2]),
            title="NREM",
        )

        self.assertEqual(collection.get_offsets().shape, (3, 2))
        self.assertEqual(ax.get_title(), "NREM\n2 modules, 3 nodes")

    def test_coordinate_shape_is_validated(self) -> None:
        fig = Figure()
        ax = fig.subplots()
        with self.assertRaises(ValueError):
            viz.plot_spatial_modules(ax, np.zeros((3, 3)), np.array([0, 1, 2]))


if __name__ == "__main__":
    unittest.main()
