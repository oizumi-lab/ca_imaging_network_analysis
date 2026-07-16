"""Boundary-safe HDF5 and deterministic window-selection tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from dmd_validation.data import load_metadata, maximin_windows, read_signal


def _write_mat(path: Path) -> None:
    n_frames, n_neurons = 100, 4
    with h5py.File(path, "w") as mat:
        mat.create_dataset("data_info", data=np.asarray([[ord(c)] for c in "sleep"], dtype=np.uint16))
        state = np.r_[np.zeros(50), np.ones(50)][:, None]
        mat.create_dataset("state", data=state)
        mat.create_dataset("nonzero_ROI", data=np.ones((1, n_neurons)))
        frame = mat.create_group("frame")
        frame.create_dataset("boundary_ind", data=np.asarray([[50.0], [100.0]]))
        values = np.arange(n_frames * n_neurons, dtype=float).reshape(n_frames, n_neurons)
        for signal in ("dFF", "spike_deconv", "spike_smoothed"):
            mat.create_dataset(signal, data=values)


class DataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "tiny.mat"
        _write_mat(self.path)
        self.meta = load_metadata(self.path, "sleep")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_selective_hdf5_read_returns_neuron_by_time(self) -> None:
        values = read_signal(self.meta, "dFF", 7, 13)
        self.assertEqual(values.shape, (4, 6))
        expected = np.arange(400, dtype=float).reshape(100, 4)[7:13].T
        np.testing.assert_array_equal(values, expected)

    def test_global_maximin_and_alternating_split(self) -> None:
        windows = maximin_windows(
            self.meta,
            label="awake",
            state_code=0.0,
            n_frames=5,
            count=3,
            development_positions=[0, 2],
        )
        self.assertEqual([window.start for window in windows], [0, 22, 44])
        self.assertEqual([window.split for window in windows], ["development", "evaluation", "development"])

    def test_window_never_crosses_label_or_acquisition_boundary(self) -> None:
        windows = maximin_windows(
            self.meta,
            label="nrem",
            state_code=1.0,
            n_frames=10,
            count=2,
            development_positions=[0],
        )
        for window in windows:
            self.assertGreaterEqual(window.start, 50)
            self.assertLessEqual(window.stop, 100)
            self.assertTrue(np.all(self.meta.state[window.start:window.stop] == 1.0))


if __name__ == "__main__":
    unittest.main()
