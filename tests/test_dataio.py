"""Synthetic-recording tests for reusable data-selection helpers."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import h5py
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


def packed_string_payload(labels: list[str]) -> np.ndarray:
    """Build the compact MCOS stream used by the atlas decoder."""
    lengths = np.array([len(label.encode("utf-16le")) // 2 for label in labels])
    encoded = "".join(labels).encode("utf-16le")
    encoded += b"\0" * (-len(encoded) % 8)
    words = np.frombuffer(encoded, dtype="<u8")
    return np.concatenate(
        [
            np.array([1, 2, len(labels), 1], dtype=np.uint64),
            lengths.astype(np.uint64),
            words,
        ]
    )


class AtlasStringDecodingTests(unittest.TestCase):
    def test_variable_length_labels_cross_packed_word_boundaries(self) -> None:
        labels = ["MOs2/3", "SSp-bfd2/3", "RSPd2/3", "root"]

        decoded = dataio._decode_mcos_string_payload(
            packed_string_payload(labels),
            expected_count=len(labels),
        )

        self.assertEqual(decoded, labels)

    def test_malformed_header_and_truncated_payload_are_rejected(self) -> None:
        payload = packed_string_payload(["MOp2/3", "SSp-ul2/3"])
        malformed = payload.copy()
        malformed[0] = 99

        with self.assertRaises(ValueError):
            dataio._decode_mcos_string_payload(malformed, expected_count=2)
        with self.assertRaises(ValueError):
            dataio._decode_mcos_string_payload(payload[:-1], expected_count=2)

    def test_lightweight_atlas_loader_avoids_activity_matrices(self) -> None:
        labels = ["MOs2/3", "VISa2/3", "root"]
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "atlas_only.mat"
            with h5py.File(path, "w") as mat:
                mat.create_dataset("nonzero_ROI", data=np.ones(len(labels)))
                rois = mat.create_group("ROIs")
                atlas = rois.create_dataset("atlas", data=np.zeros(1, dtype=np.uint8))
                atlas.attrs["MATLAB_class"] = np.bytes_("string")
                subsystem = mat.create_group("#subsystem#")
                payload = mat.create_dataset(
                    "atlas_payload",
                    data=packed_string_payload(labels),
                )
                references = subsystem.create_dataset(
                    "MCOS",
                    shape=(1,),
                    dtype=h5py.ref_dtype,
                )
                references[0] = payload.ref

            self.assertEqual(dataio.load_atlas_labels(path), labels)


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
