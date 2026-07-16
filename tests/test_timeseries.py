"""Unit tests for reusable frame and time-series transformations."""

from __future__ import annotations

import unittest

import numpy as np

from src.funcnet import timeseries as ts


class FrameSelectionTests(unittest.TestCase):
    """Frame helpers must retain selected-frame order and temporal gaps."""

    def test_frame_windows_follow_discontinuous_selection_order(self) -> None:
        frames = np.array([0, 1, 10, 11, 12, 20, 21])

        windows = ts.frame_windows(frames, width=3)

        self.assertEqual(len(windows), 2)
        np.testing.assert_array_equal(windows[0], [0, 1, 10])
        np.testing.assert_array_equal(windows[1], [11, 12, 20])
        # The final selected frame is deliberately omitted as an incomplete tail.
        self.assertFalse(any(21 in window for window in windows))

    def test_frame_windows_respect_explicit_cap(self) -> None:
        frames = np.arange(12)

        windows = ts.frame_windows(frames, width=3, max_windows=2)

        self.assertEqual(len(windows), 2)
        np.testing.assert_array_equal(windows[-1], [3, 4, 5])
        self.assertEqual(ts.frame_windows(frames, width=3, max_windows=0), [])

    def test_contiguous_runs_return_input_column_positions(self) -> None:
        frames = np.array([10, 11, 20, 21, 22, 40])

        runs = ts.contiguous_runs(frames)

        self.assertEqual(len(runs), 3)
        np.testing.assert_array_equal(runs[0], [0, 1])
        np.testing.assert_array_equal(runs[1], [2, 3, 4])
        np.testing.assert_array_equal(runs[2], [5])


class CircularShuffleTests(unittest.TestCase):
    """Circular nulls may roll within a bout but never across a frame gap."""

    def test_randomstate_shuffle_is_reproducible_and_bout_local(self) -> None:
        activity = np.array(
            [
                [0, 1, 2, 10, 11, 20],
                [100, 101, 102, 110, 111, 120],
            ]
        )
        frames = np.array([0, 1, 2, 10, 11, 20])

        shuffled = ts.circular_shuffle(
            activity,
            frames,
            rng=np.random.RandomState(7),
        )

        # This literal result locks in legacy RandomState draws as well as the
        # neuron-major, bout-minor draw order used by the tutorials.
        expected = np.array(
            [
                [1, 2, 0, 11, 10, 20],
                [102, 100, 101, 111, 110, 120],
            ]
        )
        np.testing.assert_array_equal(shuffled, expected)
        np.testing.assert_array_equal(activity[:, -1], shuffled[:, -1])

        # Each contiguous bout retains exactly its original values.
        for run in ts.contiguous_runs(frames):
            for neuron in range(activity.shape[0]):
                np.testing.assert_array_equal(
                    np.sort(shuffled[neuron, run]),
                    np.sort(activity[neuron, run]),
                )

    def test_shuffle_validates_frame_alignment(self) -> None:
        with self.assertRaises(ValueError):
            ts.circular_shuffle(np.zeros((2, 3)), np.arange(2))


class SegmentationAndSmoothingTests(unittest.TestCase):
    """Acquisition breaks are inclusive indices in the source recording."""

    def test_acquisition_boundary_stops_at_boundary_plus_one(self) -> None:
        segments = ts.acquisition_segments(
            n_frames=10,
            boundary_ind=np.array([2, 7]),
            extra_splits=[5],
        )

        self.assertEqual(segments, [(0, 3), (3, 5), (5, 8), (8, 10)])

    def test_state_runs_are_half_open(self) -> None:
        state = np.array([0.0, 0.0, 1.0, 1.0, 0.5])

        self.assertEqual(
            list(ts.state_runs(state)),
            [(0, 2, 0.0), (2, 4, 1.0), (4, 5, 0.5)],
        )

    def test_moving_average_clamps_width_for_short_segments(self) -> None:
        smoothed = ts.moving_average(np.array([2.0, 4.0]), width=5)

        # Clamping avoids np.convolve returning the requested kernel length.
        self.assertEqual(smoothed.shape, (2,))
        np.testing.assert_allclose(smoothed, [2.0, 3.0])

    def test_segmented_average_does_not_blend_across_break(self) -> None:
        values = np.array([1.0, 1.0, 9.0, 9.0])

        smoothed = ts.segmented_moving_average(
            values,
            width=5,
            boundary_ind=np.array([1]),
        )

        np.testing.assert_allclose(smoothed, values)


if __name__ == "__main__":
    unittest.main()
