"""Synthetic-recording tests for reusable data-selection helpers."""

from __future__ import annotations

import unittest

import numpy as np

from src.funcnet import dataio


def synthetic_recording(
    data_info: str = "sleep",
    nonzero_roi: np.ndarray | None = None,
) -> dataio.Recording:
    """Construct a row-aligned Recording without loading a dataset file."""
    n_neurons, n_frames = 7, 5
    if nonzero_roi is None:
        nonzero_roi = np.array([True, False, True, True, False, True, True])
    second_label = "nrem" if data_info == "sleep" else "anesthesia"
    return dataio.Recording(
        name="synthetic",
        data_info=data_info,
        dFF=np.zeros((n_neurons, n_frames)),
        spike_deconv=np.zeros((n_neurons, n_frames)),
        spike_smoothed=np.zeros((n_neurons, n_frames)),
        state=np.zeros(n_frames),
        centroid=np.column_stack([np.arange(n_neurons), np.zeros(n_neurons)]),
        used_frame={"awake": np.array([0, 1]), second_label: np.array([2, 3, 4])},
        boundary_ind=np.array([1]),
        nonzero_ROI=nonzero_roi,
    )


class StateCodeTests(unittest.TestCase):
    def test_state_codes_dispatch_by_recording_type(self) -> None:
        self.assertEqual(
            dataio.state_codes(synthetic_recording("sleep")),
            {0.0: "awake", 0.5: "quiet_awake", 1.0: "nrem", 2.0: "rem"},
        )
        self.assertEqual(
            dataio.state_codes(synthetic_recording("ane")),
            {0.0: "awake", 1.0: "anesthesia"},
        )

    def test_unknown_recording_type_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            dataio.state_codes(synthetic_recording("other"))


class NeuronRowSelectionTests(unittest.TestCase):
    def test_active_rows_follow_nonzero_roi_mask(self) -> None:
        rec = synthetic_recording()

        np.testing.assert_array_equal(
            dataio.select_neuron_rows(rec),
            [0, 2, 3, 5, 6],
        )
        np.testing.assert_array_equal(
            dataio.select_neuron_rows(rec, active_only=False),
            np.arange(7),
        )

    def test_subsample_preserves_legacy_randomstate_selection(self) -> None:
        rec = synthetic_recording()

        selected = dataio.select_neuron_rows(rec, max_neurons=3, seed=0)

        # RandomState(0), not default_rng(0), yields these legacy tutorial rows.
        np.testing.assert_array_equal(selected, [0, 2, 3])

    def test_missing_activity_mask_falls_back_to_all_rows(self) -> None:
        rec = synthetic_recording()
        rec.nonzero_ROI = None

        np.testing.assert_array_equal(dataio.select_neuron_rows(rec), np.arange(7))

    def test_invalid_maximum_is_rejected(self) -> None:
        rec = synthetic_recording()
        for maximum in (0, -1):
            with self.subTest(maximum=maximum):
                with self.assertRaises(ValueError):
                    dataio.select_neuron_rows(rec, max_neurons=maximum)
        with self.assertRaises(TypeError):
            dataio.select_neuron_rows(rec, max_neurons=True)


if __name__ == "__main__":
    unittest.main()
