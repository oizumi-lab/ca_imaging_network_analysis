"""Tests for data-only preparation helpers used by visualization tutorials."""

from __future__ import annotations

import unittest

import numpy as np
from matplotlib.figure import Figure

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

    def test_stacked_signal_uses_supplied_roi_rows_and_units(self) -> None:
        fig = Figure()
        ax = fig.subplots()
        signal = np.array([[0.0, 1.0, 0.0, 2.0], [0.0, 0.5, 0.0, 1.0]])
        original = signal.copy()
        view = {
            "n_frames": 4,
            "fs": 1.0,
            "duration_min": 4 / 60,
            "time_limits_min": (0.0, 4 / 60),
            "state": np.zeros(4),
            "codes": {0.0: "awake"},
            "trace_ids": np.array([4, 9]),
            "boundary_minutes": [],
            "acquisition_segments": [(0, 4)],
        }

        viz.plot_stacked_signal(
            ax,
            view,
            signal,
            signal_label="smoothed deconvolved activity",
            scale_unit="deconvolution units",
        )

        np.testing.assert_array_equal(signal, original)
        self.assertEqual([tick.get_text() for tick in ax.get_yticklabels()], ["4", "9"])
        self.assertEqual(
            ax.get_ylabel(),
            "neuron row (0-based)\n(smoothed deconvolved activity, offset)",
        )
        self.assertTrue(
            any(text.get_text().endswith(" deconvolution units") for text in ax.texts)
        )

    def test_stacked_signal_requires_row_and_frame_alignment(self) -> None:
        fig = Figure()
        ax = fig.subplots()
        view = {
            "n_frames": 4,
            "fs": 1.0,
            "duration_min": 4 / 60,
            "time_limits_min": (0.0, 4 / 60),
            "state": np.zeros(4),
            "codes": {0.0: "awake"},
            "trace_ids": np.array([4, 9]),
            "boundary_minutes": [],
            "acquisition_segments": [(0, 4)],
        }

        with self.assertRaises(ValueError):
            viz.plot_stacked_signal(
                ax,
                view,
                np.zeros((1, 4)),
                signal_label="signal",
                scale_unit="units",
            )
        with self.assertRaises(ValueError):
            viz.plot_stacked_signal(
                ax,
                view,
                np.zeros((2, 3)),
                signal_label="signal",
                scale_unit="units",
            )

    def test_rastermap_display_order_contains_every_neuron_once(self) -> None:
        order, n_fitted = viz.rastermap_display_order(
            n_neurons=7,
            rastermap_isort=np.array([5, 1, 6, 0, 3]),
        )

        # Fitted neurons retain the exact Rastermap order. Neurons that could
        # not be fitted are still visible, appended in original ROI-row order.
        np.testing.assert_array_equal(order, [5, 1, 6, 0, 3, 2, 4])
        np.testing.assert_array_equal(np.sort(order), np.arange(7))
        self.assertEqual(np.unique(order).size, 7)
        self.assertEqual(n_fitted, 5)

    def test_rastermap_display_order_validates_fitted_roi_ids(self) -> None:
        with self.assertRaises(ValueError):
            viz.rastermap_display_order(4, np.array([2, 2, 0]))
        with self.assertRaises(IndexError):
            viz.rastermap_display_order(4, np.array([0, 4]))
        with self.assertRaises(ValueError):
            viz.rastermap_display_order(4, np.array([[0, 1]]))


class CorticalRegionDisplayTests(unittest.TestCase):
    def test_region_palette_groups_related_anatomical_areas(self) -> None:
        expected_palette = {
            "MOs": "#005a32",
            "MOp": "#41ab5d",
            "SSp-ll": "#8c2d04",
            "SSp-ul": "#e6550d",
            "SSp-un": "#cb181d",
            "SSp-bfd": "#fb6a4a",
            "SSp-tr": "#f768a1",
            "SSp-m": "#dd3497",
            "SSp-n": "#ae017e",
            "RSPagl": "#08306b",
            "RSPd": "#2b8cbe",
            "VISa": "#3f007d",
            "VISam": "#54278f",
            "VISp": "#6a51a3",
            "VISpm": "#807dba",
            "VISrl": "#9e9ac8",
            "Unassigned": "#737373",
            "Other": "#666666",
            "Unknown": "#bdbdbd",
        }

        self.assertEqual(viz.CORTICAL_REGION_COLORS, expected_palette)
        self.assertEqual(viz.BRAIN_REGION_COLORS, expected_palette)
        self.assertEqual(
            [viz.BRAIN_REGION_FAMILIES[name] for name in ("MOs", "MOp")],
            ["Motor", "Motor"],
        )
        self.assertTrue(
            all(
                viz.BRAIN_REGION_FAMILIES[name] == "Visual"
                for name in ("VISa", "VISam", "VISp", "VISpm", "VISrl")
            )
        )

    def test_atlas_mapping_separates_other_from_unknown(self) -> None:
        atlas = np.array(
            ["MOs2/3", "RSPd", "VISp2/3", "root", "", None, "unknown"],
            dtype=object,
        )

        np.testing.assert_array_equal(
            viz.cortical_region_labels(atlas),
            [
                "MOs",
                "RSPd",
                "VISp",
                "Unassigned",
                "Unknown",
                "Unknown",
                "Unknown",
            ],
        )
        self.assertNotEqual(
            viz.CORTICAL_REGION_COLORS["Other"],
            viz.CORTICAL_REGION_COLORS["Unknown"],
        )

    def test_region_legend_uses_shared_order_and_only_present_categories(self) -> None:
        handles = viz.cortical_region_legend_handles(
            np.array(["Unknown", "SSp-tr", "Other", "SSp-tr"], dtype=object)
        )

        self.assertEqual(
            [handle.get_label() for handle in handles],
            ["SSp-tr", "Other", "Unknown"],
        )
        self.assertEqual(
            [handle.get_markerfacecolor() for handle in handles],
            [
                viz.CORTICAL_REGION_COLORS["SSp-tr"],
                viz.CORTICAL_REGION_COLORS["Other"],
                viz.CORTICAL_REGION_COLORS["Unknown"],
            ],
        )

    def test_region_legend_can_show_unabridged_area_names(self) -> None:
        self.assertEqual(
            set(viz.CORTICAL_REGION_NAMES),
            set(viz.CORTICAL_REGION_COLORS),
        )
        handles = viz.cortical_region_legend_handles(
            ["MOs", "SSp-bfd", "Other"],
            unabridged=True,
        )

        self.assertEqual(
            [handle.get_label() for handle in handles],
            [
                "MOs — Secondary motor area",
                "SSp-bfd — Primary somatosensory area, barrel field",
                "Other — Other valid atlas area",
            ],
        )

    def test_brain_region_mapping_keeps_exact_atlas_acronyms(self) -> None:
        atlas = np.array(
            ["VISp2/3", "SSp-m2/3", "root", "LP2/3", "", None],
            dtype=object,
        )

        np.testing.assert_array_equal(
            viz.brain_region_labels(atlas),
            ["VISp", "SSp-m", "Unassigned", "Other", "Unknown", "Unknown"],
        )
        self.assertNotEqual(
            viz.BRAIN_REGION_COLORS["VISp"],
            viz.BRAIN_REGION_COLORS["SSp-m"],
        )

    def test_display_region_mapping_is_idempotent(self) -> None:
        atlas = np.array(
            ["VISp2/3", "root", "LP2/3", "", None],
            dtype=object,
        )
        displayed = viz.brain_region_labels(atlas)

        np.testing.assert_array_equal(
            viz.brain_region_labels(displayed),
            displayed,
        )

    def test_brain_region_order_preserves_activity_rank_within_regions(self) -> None:
        atlas = np.array(
            ["VISp2/3", "MOs2/3", "VISa2/3", "MOs2/3", "SSp-m2/3"],
            dtype=object,
        )
        activity_order = np.array([3, 0, 4, 2, 1])

        order = viz.brain_region_order(atlas, activity_order)

        # MOs comes first in the shared anatomical order, but its two neurons
        # retain their relative positions from the activity-ranked permutation.
        np.testing.assert_array_equal(order, [3, 1, 4, 2, 0])
        np.testing.assert_array_equal(np.sort(order), np.arange(atlas.size))

    def test_brain_region_order_requires_a_complete_roi_permutation(self) -> None:
        atlas = np.array(["MOs2/3", "MOp2/3", "RSPd2/3"], dtype=object)

        with self.assertRaises(ValueError):
            viz.brain_region_order(atlas, np.array([0, 0, 2]))
        with self.assertRaises(ValueError):
            viz.brain_region_order(atlas, np.array([0, 1]))
        with self.assertRaises(TypeError):
            viz.brain_region_order(atlas, np.array([0.0, 1.0, 2.0]))

    def test_region_strip_has_one_cell_per_roi_in_supplied_order(self) -> None:
        atlas = np.array(["MOs2/3", "", "VISp2/3", "RSPd2/3"], dtype=object)
        roi_order = np.array([3, 0, 2, 1])
        ax = Figure().subplots()

        image, handles = viz.plot_cortical_region_strip(
            ax,
            atlas,
            roi_order,
            n_fitted=3,
        )

        color_index = {
            region: index for index, region in enumerate(viz.CORTICAL_REGION_COLORS)
        }
        np.testing.assert_array_equal(
            np.asarray(image.get_array()).ravel(),
            [
                color_index["RSPd"],
                color_index["MOs"],
                color_index["VISp"],
                color_index["Unknown"],
            ],
        )
        self.assertEqual(np.asarray(image.get_array()).shape, (4, 1))
        self.assertEqual(
            [handle.get_label() for handle in handles],
            ["MOs", "RSPd", "VISp", "Unknown"],
        )
        self.assertEqual(len(ax.lines), 1)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [2.5, 2.5])
        self.assertEqual(tuple(ax.get_ylim()), (4.0, 0.0))

    def test_region_strip_accepts_an_explicit_unique_roi_subset(self) -> None:
        atlas = np.array(["MOs2/3", "MOp2/3", "RSPd2/3"], dtype=object)
        ax = Figure().subplots()

        image, _ = viz.plot_cortical_region_strip(ax, atlas, np.array([2, 0]))

        self.assertEqual(np.asarray(image.get_array()).shape, (2, 1))
        self.assertEqual(tuple(ax.get_ylim()), (2.0, 0.0))

    def test_region_strip_rejects_invalid_roi_subsets(self) -> None:
        atlas = np.array(["MOs2/3", "MOp2/3", "RSPd2/3"], dtype=object)
        ax = Figure().subplots()

        with self.assertRaises(ValueError):
            viz.plot_cortical_region_strip(ax, atlas, np.array([0, 0, 2]))
        with self.assertRaises(ValueError):
            viz.plot_cortical_region_strip(
                ax,
                atlas,
                np.array([0, 1, 2]),
                n_fitted=4,
            )


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

    def test_region_spike_plot_maps_original_rois_to_grouped_rows(self) -> None:
        # The stored raster is activity-ranked as ROI [2, 0, 3, 1]. Region
        # grouping produces ROI [3, 1, 2, 0], retaining rank within MOs.
        view = {
            "n_neurons": 4,
            "raster": np.array(
                [
                    [1, 0],  # ROI 2
                    [0, 1],  # ROI 0
                    [1, 1],  # ROI 3
                    [0, 0],  # ROI 1
                ],
                dtype=np.uint8,
            ),
            "neuron_order": np.array([2, 0, 3, 1]),
            "brain_regions": np.array(["VISp", "MOs", "VISa", "MOs"]),
            "brain_region_order": np.array([3, 1, 2, 0]),
            "bin_centers_min": np.array([0.005, 0.015]),
            "time_limits_min": (0.0, 2 / 60),
            "state": np.zeros(2),
            "codes": {0.0: "awake"},
            "fs": 1.0,
            "bin_frames": 1,
            "boundary_minutes": [],
        }

        ax = Figure().subplots()
        viz.plot_brain_region_spike_raster(ax, view)

        offsets = np.asarray(ax.collections[0].get_offsets())
        np.testing.assert_allclose(
            offsets,
            [[0.005, 0], [0.015, 0], [0.005, 2], [0.015, 3]],
        )
        self.assertEqual(ax.get_ylabel(), "all neurons\n(grouped by atlas region)")
        self.assertEqual(tuple(ax.get_ylim()), (4.0, 0.0))
        self.assertEqual(len(ax.lines), 2)
        np.testing.assert_allclose(
            [line.get_ydata()[0] for line in ax.lines],
            [1.5, 2.5],
        )

    def test_rastermap_spike_plot_uses_original_roi_ids_to_reorder_rows(self) -> None:
        # The stored compact raster is activity-ranked as ROI [2, 0, 3, 1].
        # Rastermap instead orders fitted ROIs [1, 3, 0], then appends ROI 2.
        view = {
            "n_neurons": 4,
            "raster": np.array(
                [
                    [1, 0],  # ROI 2
                    [0, 1],  # ROI 0
                    [1, 1],  # ROI 3
                    [0, 0],  # ROI 1
                ],
                dtype=np.uint8,
            ),
            "neuron_order": np.array([2, 0, 3, 1]),
            "rastermap_isort": np.array([1, 3, 0]),
            "rastermap_valid_rows": np.array([0, 1, 3]),
            "bin_centers_min": np.array([0.005, 0.015]),
            "time_limits_min": (0.0, 2 / 60),
            "state": np.zeros(2),
            "codes": {0.0: "awake"},
            "fs": 1.0,
            "bin_frames": 1,
            "boundary_minutes": [],
        }

        ax = Figure().subplots()
        viz.plot_rastermap_spike_raster(ax, view)

        # Event rows after reordering are ROI 3 -> row 1, ROI 0 -> row 2,
        # and appended ROI 2 -> row 3. ROI 1 has no events but retains row 0.
        offsets = np.asarray(ax.collections[0].get_offsets())
        np.testing.assert_allclose(
            offsets,
            [[0.005, 1], [0.015, 1], [0.015, 2], [0.005, 3]],
        )
        self.assertEqual(ax.get_ylabel(), "all neurons\n(Rastermap order)")
        self.assertEqual(tuple(ax.get_ylim()), (4.0, 0.0))

    def test_active_rastermap_spike_plot_does_not_append_inactive_rows(self) -> None:
        view = {
            "n_neurons": 4,
            "raster": np.array(
                [[1, 0], [0, 1], [1, 1], [0, 0]],
                dtype=np.uint8,
            ),
            "neuron_order": np.array([2, 0, 3, 1]),
            "rastermap_isort": np.array([1, 3, 0]),
            "rastermap_valid_rows": np.array([0, 1, 3]),
            "rastermap_display_selected_only": True,
            "bin_centers_min": np.array([0.005, 0.015]),
            "time_limits_min": (0.0, 2 / 60),
            "state": np.zeros(2),
            "codes": {0.0: "awake"},
            "fs": 1.0,
            "bin_frames": 1,
            "boundary_minutes": [],
        }

        ax = Figure().subplots()
        viz.plot_rastermap_spike_raster(ax, view)

        offsets = np.asarray(ax.collections[0].get_offsets())
        np.testing.assert_allclose(offsets, [[0.005, 1], [0.015, 1], [0.015, 2]])
        self.assertEqual(
            ax.get_ylabel(),
            "active selected neurons\n(Rastermap order)",
        )
        self.assertEqual(tuple(ax.get_ylim()), (3.0, 0.0))


if __name__ == "__main__":
    unittest.main()
