"""Tests for EEG/EMG loading, synchronization, and display preparation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from src.funcnet import physiology as phys


def write_physiology_part(
    path: Path,
    eeg: np.ndarray,
    emg: np.ndarray | None = None,
    frame_trigger: np.ndarray | None = None,
    fs: float = 100.0,
) -> None:
    """Write the small classic-MAT subset used by the loader tests."""
    eeg = np.asarray(eeg, dtype=np.float32)
    if emg is None:
        emg = np.zeros_like(eeg)
    if frame_trigger is None:
        frame_trigger = np.zeros_like(eeg)
    payload = {
        "Fs": np.asarray([[fs]], dtype=np.int64),
        "EEG": eeg[np.newaxis, :],
        "EMG": np.asarray(emg, dtype=np.float32)[np.newaxis, :],
        "FrameTrigger": np.asarray(frame_trigger, dtype=np.float32)[np.newaxis, :],
        "ChannelNames": np.asarray(
            [["FrameTrigger", "EEG", "EMG"]],
            dtype=object,
        ),
        "ChannelUnits": np.asarray([["V", "mV", "mV"]], dtype=object),
    }
    savemat(path, payload)


def trigger_signal(n_samples: int, edge_groups: list[list[int]]) -> np.ndarray:
    values = np.zeros(n_samples, dtype=np.float32)
    for edge in [edge for group in edge_groups for edge in group]:
        values[edge : edge + 2] = 3.2
    return values


class PhysiologyLoadingTests(unittest.TestCase):
    def test_mouse06_parts_are_loaded_awake_then_anesthesia(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            awake = directory / "mouse06_ane_physiological_data_awake.mat"
            anesthesia = directory / "mouse06_ane_physiological_data_ane.mat"
            write_physiology_part(awake, np.ones(6))
            write_physiology_part(anesthesia, np.full(4, 2.0))

            recording = phys.load_physiology("mouse06_ane", directory)

        self.assertEqual(
            [path.name for path in recording.source_paths],
            [awake.name, anesthesia.name],
        )
        np.testing.assert_array_equal(recording.eeg, [1] * 6 + [2] * 4)
        np.testing.assert_array_equal(recording.part_boundaries, [6])

    def test_missing_required_channel_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mouse01_sleep_physiological_data.mat"
            write_physiology_part(path, np.ones(6))
            from scipy.io import loadmat

            payload = loadmat(path)
            payload.pop("EMG")
            savemat(
                path,
                {key: value for key, value in payload.items() if not key.startswith("__")},
            )

            with self.assertRaisesRegex(ValueError, "missing variables: EMG"):
                phys.load_physiology("mouse01_sleep", tmp)


class FrameTriggerAlignmentTests(unittest.TestCase):
    def test_short_primer_is_skipped_and_surplus_pulses_are_reported(self) -> None:
        groups = [
            [10, 20],
            [100, 110, 120, 130, 140],
            [250, 260, 270, 280, 290],
        ]
        frame_trigger = trigger_signal(320, groups)
        recording = phys.PhysiologyRecording(
            name="synthetic",
            fs=100.0,
            eeg=np.zeros(320, dtype=np.float32),
            emg=np.zeros(320, dtype=np.float32),
            frame_trigger=frame_trigger,
            part_boundaries=np.array([], dtype=np.int64),
            source_paths=(),
        )

        alignment = phys.align_frame_triggers(
            recording,
            n_frames=8,
            boundary_ind=np.array([4, 7]),
            imaging_fs=10.0,
        )

        self.assertEqual(alignment.detected_trigger_count, 12)
        self.assertEqual(alignment.mapped_trigger_count, 8)
        self.assertEqual(alignment.ignored_trigger_count, 4)
        self.assertEqual(
            [
                (
                    segment.frame_start,
                    segment.frame_stop,
                    segment.sample_start,
                    segment.sample_stop,
                    segment.trigger_group_size,
                )
                for segment in alignment.segments
            ],
            [(0, 5, 100, 150, 5), (5, 8, 250, 280, 5)],
        )

    def test_released_sleep_mapping_keeps_the_suffix_after_primer(self) -> None:
        groups = [[10, 20], [100, 110, 120, 130, 140]]
        recording = phys.PhysiologyRecording(
            name="mouse01_sleep",
            fs=100.0,
            eeg=np.zeros(160, dtype=np.float32),
            emg=np.zeros(160, dtype=np.float32),
            frame_trigger=trigger_signal(160, groups),
            part_boundaries=np.array([], dtype=np.int64),
            source_paths=(),
        )

        alignment = phys.align_frame_triggers(
            recording,
            n_frames=3,
            boundary_ind=np.array([2]),
            imaging_fs=10.0,
        )

        segment = alignment.segments[0]
        self.assertEqual(segment.trigger_group_index, 1)
        self.assertEqual(segment.trigger_start_offset, 2)
        self.assertEqual((segment.sample_start, segment.sample_stop), (120, 150))

    def test_released_mouse07_mapping_skips_the_middle_trigger_bout(self) -> None:
        groups = [
            [10, 20, 30],
            [100, 110, 120],
            [200, 210, 220],
        ]
        recording = phys.PhysiologyRecording(
            name="mouse07_ane",
            fs=100.0,
            eeg=np.zeros(240, dtype=np.float32),
            emg=np.zeros(240, dtype=np.float32),
            frame_trigger=trigger_signal(240, groups),
            part_boundaries=np.array([], dtype=np.int64),
            source_paths=(),
        )

        alignment = phys.align_frame_triggers(
            recording,
            n_frames=6,
            boundary_ind=np.array([2, 5]),
            imaging_fs=10.0,
        )

        self.assertEqual(
            [segment.trigger_group_index for segment in alignment.segments],
            [0, 2],
        )
        self.assertEqual(
            [segment.sample_start for segment in alignment.segments],
            [10, 200],
        )

    def test_loaded_release_rejects_an_unexpected_trigger_bout_signature(self) -> None:
        groups = [[10, 20], [100, 110, 120, 130, 140]]
        recording = phys.PhysiologyRecording(
            name="mouse01_sleep",
            fs=100.0,
            eeg=np.zeros(160, dtype=np.float32),
            emg=np.zeros(160, dtype=np.float32),
            frame_trigger=trigger_signal(160, groups),
            part_boundaries=np.array([], dtype=np.int64),
            source_paths=(Path("mouse01_sleep_physiological_data.mat"),),
        )

        with self.assertRaisesRegex(ValueError, "expected released FrameTrigger"):
            phys.align_frame_triggers(
                recording,
                n_frames=3,
                boundary_ind=np.array([2]),
                imaging_fs=10.0,
            )

    def test_source_file_boundary_forces_a_trigger_bout_split(self) -> None:
        frame_trigger = trigger_signal(100, [[10, 20, 30, 40, 50, 60]])
        recording = phys.PhysiologyRecording(
            name="synthetic",
            fs=100.0,
            eeg=np.zeros(100, dtype=np.float32),
            emg=np.zeros(100, dtype=np.float32),
            frame_trigger=frame_trigger,
            part_boundaries=np.array([35]),
            source_paths=(),
        )

        groups, period = phys.frame_trigger_groups(recording)

        self.assertEqual(period, 10)
        self.assertEqual([group.tolist() for group in groups], [[10, 20, 30], [40, 50, 60]])


class PhysiologyDisplayPreparationTests(unittest.TestCase):
    def test_spectrogram_recovers_frequency_and_uses_calcium_time(self) -> None:
        physiology_fs = 1000.0
        imaging_fs = 10.0
        n_frames = 200
        n_samples = 20_000
        time = np.arange(n_samples) / physiology_fs
        frame_edges = np.arange(0, n_samples, 100)
        frame_trigger = trigger_signal(n_samples, [frame_edges.tolist()])
        recording = phys.PhysiologyRecording(
            name="synthetic",
            fs=physiology_fs,
            eeg=np.sin(2 * np.pi * 10 * time).astype(np.float32),
            emg=np.zeros(n_samples, dtype=np.float32),
            frame_trigger=frame_trigger,
            part_boundaries=np.array([], dtype=np.int64),
            source_paths=(),
        )
        alignment = phys.align_frame_triggers(
            recording,
            n_frames=n_frames,
            boundary_ind=np.array([n_frames - 1]),
            imaging_fs=imaging_fs,
        )

        panels, limits = phys.prepare_eeg_spectrogram(
            recording,
            alignment,
            imaging_fs=imaging_fs,
            target_fs=100.0,
            max_frequency_hz=25.0,
            window_seconds=2.0,
        )

        self.assertEqual(len(panels), 1)
        panel = panels[0]
        peak_frequency = panel.frequency_hz[
            np.argmax(np.mean(panel.power_db_hz, axis=1))
        ]
        self.assertAlmostEqual(float(peak_frequency), 10.0, delta=0.6)
        self.assertGreaterEqual(float(panel.time_min.min()), 0.0)
        self.assertLessEqual(float(panel.time_min.max()), n_frames / imaging_fs / 60)
        self.assertLess(limits[0], limits[1])
        self.assertTrue(np.all(np.isfinite(panel.power_db_hz)))

    def test_emg_rms_envelope_is_nonnegative_and_retains_a_burst(self) -> None:
        sample_time = np.arange(1000) / 1000
        emg = np.zeros(1000, dtype=np.float32)
        burst = (sample_time >= 0.2) & (sample_time < 0.5)
        emg[burst] = np.sin(2 * np.pi * 100 * sample_time[burst])
        recording = phys.PhysiologyRecording(
            name="synthetic",
            fs=1000.0,
            eeg=np.zeros_like(emg),
            emg=emg,
            frame_trigger=trigger_signal(1000, [[0, 100, 200, 300, 400, 500, 600, 700, 800, 900]]),
            part_boundaries=np.array([], dtype=np.int64),
            source_paths=(),
        )
        alignment = phys.align_frame_triggers(
            recording,
            n_frames=10,
            boundary_ind=np.array([9]),
            imaging_fs=10.0,
        )

        panels, amplitude_limit = phys.prepare_emg_envelope(
            recording,
            alignment,
            imaging_fs=10.0,
            target_fs=500.0,
            filter_band_hz=(20.0, 200.0),
            rms_window_seconds=0.05,
            display_fs=100.0,
        )

        self.assertEqual(len(panels), 1)
        self.assertTrue(np.all(panels[0].amplitude >= 0))
        self.assertGreater(float(panels[0].amplitude.max()), 0.25)
        self.assertGreater(amplitude_limit, 0)

    def test_scoring_features_recover_delta_theta_and_four_second_emg(self) -> None:
        physiology_fs = 1000.0
        imaging_fs = 10.0
        duration_seconds = 12
        n_samples = int(duration_seconds * physiology_fs)
        n_frames = int(duration_seconds * imaging_fs)
        sample_time = np.arange(n_samples) / physiology_fs
        first_half = sample_time < duration_seconds / 2
        eeg = np.empty(n_samples, dtype=np.float32)
        eeg[first_half] = np.sin(2 * np.pi * 2 * sample_time[first_half])
        eeg[~first_half] = np.sin(2 * np.pi * 7 * sample_time[~first_half])
        emg = np.zeros(n_samples, dtype=np.float32)
        emg[sample_time < 4] = np.sin(
            2 * np.pi * 100 * sample_time[sample_time < 4]
        )
        frame_edges = np.arange(n_frames, dtype=np.int64) * 100
        recording = phys.PhysiologyRecording(
            name="synthetic",
            fs=physiology_fs,
            eeg=eeg,
            emg=emg,
            frame_trigger=trigger_signal(n_samples, [frame_edges.tolist()]),
            part_boundaries=np.array([], dtype=np.int64),
            source_paths=(),
        )
        alignment = phys.align_frame_triggers(
            recording,
            n_frames=n_frames,
            boundary_ind=np.array([n_frames - 1]),
            imaging_fs=imaging_fs,
        )

        panels, color_limit = phys.prepare_state_scoring_features(
            recording,
            alignment,
            imaging_fs=imaging_fs,
            analysis_fs=physiology_fs,
            chunk_windows=7,
        )

        self.assertEqual(len(panels), 1)
        panel = panels[0]
        self.assertEqual((panel.frame_index[0], panel.frame_index[-1]), (20, 100))
        early = np.argmin(np.abs(panel.frame_index - 30))
        late = np.argmin(np.abs(panel.frame_index - 90))
        self.assertGreater(panel.relative_delta[early], panel.relative_delta[late])
        self.assertGreater(
            panel.delta_theta_ratio[early],
            panel.delta_theta_ratio[late],
        )
        np.testing.assert_allclose(
            panel.delta_theta_ratio,
            panel.relative_delta * panel.raw_delta_theta_ratio,
            rtol=2e-6,
        )
        self.assertGreater(panel.emg_rms[early], panel.emg_rms[late])
        self.assertGreater(color_limit, 0)
        self.assertTrue(np.all(np.isfinite(panel.frequency_normalized_power_db)))


class SleepClassificationRuleTests(unittest.TestCase):
    def test_threshold_fit_ignores_quiet_awake_and_separates_emg(self) -> None:
        fit = phys.fit_emg_threshold(
            np.array([5.0, 4.0, 2.0, 1.0, 100.0]),
            np.array(["awake", "awake", "nrem", "rem", "quiet_awake"]),
        )

        self.assertGreater(fit.threshold, 2.0)
        self.assertLess(fit.threshold, 4.0)
        self.assertEqual(fit.balanced_accuracy, 1.0)
        self.assertEqual((fit.n_awake, fit.n_sleep), (2, 2))

    def test_sleep_rule_applies_emg_gate_before_delta_theta_boundary(self) -> None:
        predicted = phys.classify_sleep_windows(
            emg_rms=np.array([2.0, 0.1, 0.1]),
            delta_theta_ratio=np.array([10.0, 1.0, 0.1]),
            emg_threshold=1.0,
        )

        np.testing.assert_array_equal(predicted, ["awake", "nrem", "rem"])

    def test_sleep_rule_matches_archived_threshold_equalities(self) -> None:
        predicted = phys.classify_sleep_windows(
            emg_rms=np.array([1.0, 0.1]),
            delta_theta_ratio=np.array([0.1, 0.3]),
            emg_threshold=1.0,
        )

        np.testing.assert_array_equal(predicted, ["awake", "nrem"])

    def test_short_run_cleanup_is_deterministic_and_keeps_exact_minimum(self) -> None:
        same_neighbors = phys.merge_short_state_runs(
            np.array(["awake"] * 3 + ["rem"] + ["awake"] * 3),
            min_run_samples=2,
        )
        np.testing.assert_array_equal(same_neighbors, ["awake"] * 7)

        longer_following = phys.merge_short_state_runs(
            np.array(["awake"] * 2 + ["rem"] + ["nrem"] * 4),
            min_run_samples=2,
        )
        np.testing.assert_array_equal(
            longer_following,
            ["awake"] * 2 + ["nrem"] * 5,
        )

        exact_minimum = phys.merge_short_state_runs(
            np.array(["awake"] * 3 + ["rem"] * 2 + ["nrem"] * 3),
            min_run_samples=2,
        )
        np.testing.assert_array_equal(
            exact_minimum,
            ["awake"] * 3 + ["rem"] * 2 + ["nrem"] * 3,
        )


if __name__ == "__main__":
    unittest.main()
