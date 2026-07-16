"""Regression tests for split preservation and empirical summary logic."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from dmd_validation.data import RecordingMetadata, Window
from dmd_validation.negative_controls import containing_state_run, summarize_negative_controls
from dmd_validation.preprocessing import ArmSpec
from dmd_validation.resampling import synchronized_cell_intervals
from dmd_validation.validation import evaluate_configuration, fit_long_block


def _config() -> dict[str, object]:
    return {
        "fs_hz": 10.0,
        "windows": {"train_fraction": 0.8},
        "model": {
            "forecast_horizons_seconds": [0.2, 1.0, 2.0],
            "forecast_explosion_multiplier": 100.0,
            "rotation_minimum_modulus": 0.25,
            "rotation_nearly_real_tolerance": 1e-8,
            "long_block_ranks": [2, 4],
            "native_ranks": [2, 4],
            "native_lags_frames": [1, 2],
            "smoothed_extra_lag_frames": 15,
            "b4_lags_bins": [1],
            "minimum_pairs_per_dimension": 10,
            "ridge_relative": [0.001, 0.1, 1.0],
        },
        "inference": {"confidence_level": 0.95},
    }


def _window(n_frames: int, kind: str = "deployment") -> Window:
    return Window(
        window_id="recording__awake__test",
        recording="recording",
        label="awake",
        state_code=0.0,
        start=0,
        stop=n_frames,
        n_frames=n_frames,
        segment=0,
        kind=kind,
        split="evaluation" if kind == "deployment" else "diagnostic",
        temporal_position=0,
    )


class ValidationTests(unittest.TestCase):
    def test_trimmed_sensitivity_preserves_original_train_test_boundary(self) -> None:
        rng = np.random.default_rng(4)
        trimmed_scores = rng.standard_normal((2, 286))
        selected = {"rank": 2, "lag": 1, "eta": 0.1, "diagonal_eta": 0.1}
        spec = ArmSpec("Sz", "spike_smoothed", 1, "rms", "test")
        with patch("dmd_validation.validation.load_scores", return_value=trimmed_scores):
            metrics, spectra = evaluate_configuration(
                None,
                [_window(300)],
                spec,
                None,
                selected,
                _config(),
                trim_samples=7,
                sensitivity_name="trimmed_seven_frame_edges",
            )
        self.assertTrue((metrics["original_train_stop"] == 240).all())
        self.assertTrue((metrics["effective_train_stop"] == 233).all())
        self.assertTrue((spectra["effective_train_stop"] == 233).all())

    def test_long_block_records_independently_selected_diagonal_ridge(self) -> None:
        rng = np.random.default_rng(8)
        scores = rng.standard_normal((4, 1500))
        for time in range(1, scores.shape[1]):
            scores[:, time] += 0.7 * scores[:, time - 1]
        spec = ArmSpec("P", "spike_deconv", 1, "rms", "test")
        with patch("dmd_validation.validation.load_scores", return_value=scores):
            metrics, selected = fit_long_block(
                None,
                _window(1500, kind="long"),
                spec,
                None,
                _config(),
            )
        self.assertIn("diagonal_eta", selected)
        self.assertIn("diagonal_validation_skill", selected)
        self.assertIn("diagonal_eta", metrics)
        self.assertTrue(np.isfinite(metrics["skill_diagonal"]).all())

    def test_synchronized_bootstrap_uses_three_window_median_per_repetition(self) -> None:
        rows = []
        for repetition in range(5):
            for window in range(3):
                rows.append(
                    {
                        "recording": "r",
                        "label": "awake",
                        "horizon_role": "near_one_second",
                        "repetition": repetition,
                        "window_id": f"w{window}",
                        "skill_persistence": repetition + window,
                        "skill_diagonal": repetition - window,
                        "finite_forecast": True,
                        "explosive_forecast": False,
                    }
                )
        synchronized, summary = synchronized_cell_intervals(pd.DataFrame(rows), 0.95)
        np.testing.assert_allclose(
            synchronized["median_skill_persistence"],
            np.arange(5) + 1,
        )
        self.assertEqual(int(summary.iloc[0]["windows_per_repetition_min"]), 3)

    def test_containing_state_run_is_split_at_acquisition_boundary(self) -> None:
        meta = RecordingMetadata(
            name="r",
            path=Path("unused"),
            paradigm="sleep",
            n_neurons=2,
            n_frames=12,
            state=np.zeros(12),
            segment_stops=np.asarray([6, 12]),
            file_size=0,
            file_mtime_ns=0,
        )
        window = Window("w", "r", "awake", 0.0, 7, 11, 4, 1, "deployment", "evaluation", 0)
        self.assertEqual(containing_state_run(meta, window), (6, 12))

    def test_null_summary_compares_observed_to_circular_percentile(self) -> None:
        null_rows = []
        for kind in ("circular_shift", "stationary_iid"):
            for repetition in range(10):
                for window in range(3):
                    null_rows.append(
                        {
                            "null_kind": kind,
                            "recording": "r",
                            "label": "awake",
                            "horizon_role": "near_one_second",
                            "repetition": repetition,
                            "window_id": f"w{window}",
                            "skill_persistence": 0.1,
                            "skill_diagonal": repetition / 100,
                            "reference_similarity": 0.2,
                            "tracking_match_success": True,
                            "spectral_class": "no_resolved_rotation",
                        }
                    )
        observed = pd.DataFrame(
            {
                "arm": ["P"] * 3,
                "sensitivity": ["primary"] * 3,
                "horizon_role": ["gate"] * 3,
                "actual_horizon_seconds": [1.0] * 3,
                "recording": ["r"] * 3,
                "label": ["awake"] * 3,
                "window_id": ["w0", "w1", "w2"],
                "skill_persistence": [0.2] * 3,
                "skill_diagonal": [0.5] * 3,
            }
        )
        _, summary = summarize_negative_controls(pd.DataFrame(null_rows), observed, 95.0)
        self.assertTrue(summary["observed_exceeds_null_percentile"].all())
        self.assertTrue(summary["coordinated_mode_qualifier_cell"].all())

        observed["skill_diagonal"] = -0.01
        negative_null = pd.DataFrame(null_rows)
        negative_null["skill_diagonal"] = -0.1
        _, negative_summary = summarize_negative_controls(
            negative_null, observed, 95.0
        )
        self.assertTrue(negative_summary["observed_exceeds_null_percentile"].all())
        self.assertFalse(negative_summary["coordinated_mode_qualifier_cell"].any())


if __name__ == "__main__":
    unittest.main()
