"""Load and synchronize the v3.0 EEG/EMG recordings.

The physiology files added in RIKEN dataset 20260708-001 are classic MATLAB
files sampled at 5 kHz.  They contain more wall-clock time than the processed
calcium matrices, so array length or endpoint matching is not a valid
synchronization method.  This module detects ``FrameTrigger`` bouts, pairs them
with the acquisition segments in ``frame.boundary_ind``, and exposes compact
display products on the calcium recording's gap-free time axis.  It also
reconstructs the paper's 4-s EEG/EMG sleep-scoring features for the separate
label-verification analysis.

The inspection helpers apply display-oriented filtering and downsampling.  The
verification helper instead resamples the deposited amplifier outputs to the
paper-reported 1-kHz rate and computes its relative-delta, delta/theta, and EMG
RMS features without reusing those display filters.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import loadmat

from . import timeseries as ts
from .paths import PHYSIOLOGY_DIR


@dataclass(frozen=True)
class PhysiologyRecording:
    """EEG, EMG, and frame timing for one calcium-imaging recording."""

    name: str
    fs: float
    eeg: np.ndarray
    emg: np.ndarray
    frame_trigger: np.ndarray
    part_boundaries: np.ndarray
    source_paths: tuple[Path, ...]
    eeg_unit: str = "mV"
    emg_unit: str = "mV"

    @property
    def n_samples(self) -> int:
        return int(self.eeg.size)


@dataclass(frozen=True)
class AlignedPhysiologySegment:
    """One gap-free calcium segment paired with one frame-trigger bout."""

    frame_start: int
    frame_stop: int
    sample_start: int
    sample_stop: int
    trigger_group_index: int
    trigger_group_size: int
    trigger_start_offset: int
    trigger_samples: np.ndarray

    @property
    def n_frames(self) -> int:
        return self.frame_stop - self.frame_start


@dataclass(frozen=True)
class PhysiologyAlignment:
    """Validated mapping from physiological samples to calcium frames."""

    segments: tuple[AlignedPhysiologySegment, ...]
    detected_trigger_count: int
    mapped_trigger_count: int
    nominal_period_samples: float

    @property
    def ignored_trigger_count(self) -> int:
        return self.detected_trigger_count - self.mapped_trigger_count


@dataclass(frozen=True)
class EEGSpectrogramPanel:
    """A spectrogram block positioned on the gap-free recorded-time axis."""

    time_min: np.ndarray
    frequency_hz: np.ndarray
    power_db_hz: np.ndarray


@dataclass(frozen=True)
class EMGEnvelopePanel:
    """RMS EMG amplitude for one aligned acquisition segment."""

    time_min: np.ndarray
    amplitude: np.ndarray


@dataclass(frozen=True)
class StateScoringFeaturePanel:
    """Paper-motivated 4-s EEG/EMG features for one acquisition segment."""

    segment_index: int
    frame_index: np.ndarray
    trigger_sample: np.ndarray
    time_min: np.ndarray
    frequency_hz: np.ndarray
    frequency_normalized_power_db: np.ndarray
    relative_delta: np.ndarray
    delta_theta_ratio: np.ndarray
    raw_delta_theta_ratio: np.ndarray
    emg_rms: np.ndarray


@dataclass(frozen=True)
class EMGThresholdFit:
    """One label-calibrated approximation of an unpublished manual threshold."""

    threshold: float
    balanced_accuracy: float
    n_awake: int
    n_sleep: int


# The release README requires FrameTrigger synchronization but does not spell
# out which portions of surplus trigger bouts survived calcium preprocessing.
# These selections were resolved from boundary_ind, state transitions,
# StartTrigger placement, and cross-modal correspondence in the released data.
# Each tuple is (detected trigger-bout index, selection anchor).  ``end`` keeps
# the final L edges; ``start`` keeps the first L edges for a calcium segment of
# length L.  Names outside this fixed release manifest use validated sequential
# prefix matching, which keeps the pure helper useful for synthetic/new data.
_RELEASED_TRIGGER_SELECTIONS: dict[str, tuple[tuple[int, str], ...]] = {
    "mouse01_sleep": ((1, "end"),),
    "mouse02_sleep": ((1, "end"), (2, "end")),
    "mouse03_sleep": ((1, "end"), (2, "end")),
    "mouse04_day1_sleep": ((1, "end"), (2, "end")),
    "mouse04_day2_sleep": ((1, "end"), (2, "end"), (3, "end")),
    "mouse05_sleep": ((1, "end"), (2, "end")),
    "mouse03_ane": ((0, "start"), (1, "start")),
    "mouse05_ane": ((0, "start"), (1, "start"), (2, "end")),
    "mouse06_ane": ((0, "start"), (1, "start"), (2, "start")),
    "mouse07_ane": ((0, "start"), (2, "start")),
}

# Exact rising-edge bout counts provide a second guard against applying the
# selections above to a partial, corrupt, or subsequently changed release file.
# Synthetic objects leave ``source_paths`` empty and can still exercise the
# selection algorithm at small scale in unit tests.
_RELEASED_TRIGGER_GROUP_SIZES: dict[str, tuple[int, ...]] = {
    "mouse01_sleep": (100, 9500),
    "mouse02_sleep": (100, 9500, 9500),
    "mouse03_sleep": (100, 9500, 9501),
    "mouse04_day1_sleep": (100, 9500, 9501),
    "mouse04_day2_sleep": (100, 9500, 9500, 5001),
    "mouse05_sleep": (100, 9500, 9501),
    "mouse03_ane": (9500, 9500),
    "mouse05_ane": (9500, 9500, 9500),
    "mouse06_ane": (9500, 9500, 6164),
    "mouse07_ane": (9500, 9500, 9500),
}


def physiology_paths(
    recording_name: str,
    directory: str | Path = PHYSIOLOGY_DIR,
) -> tuple[Path, ...]:
    """Return the expected physiology path(s) in chronological order."""
    directory = Path(directory)
    if recording_name == "mouse06_ane":
        paths = (
            directory / "mouse06_ane_physiological_data_awake.mat",
            directory / "mouse06_ane_physiological_data_ane.mat",
        )
    else:
        paths = (directory / f"{recording_name}_physiological_data.mat",)

    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing EEG/EMG data: "
            f"{', '.join(missing)}. Run "
            "all cells in `scripts/00_download_data.ipynb`."
        )
    return paths


def _cell_strings(values: object) -> list[str]:
    """Normalize a squeezed MATLAB cell/string array to Python strings."""
    return [str(value).strip() for value in np.atleast_1d(values).ravel()]


def _load_part(path: Path) -> dict[str, object]:
    raw = loadmat(
        path,
        variable_names=(
            "Fs",
            "ChannelNames",
            "ChannelUnits",
            "EEG",
            "EMG",
            "FrameTrigger",
        ),
        squeeze_me=True,
        simplify_cells=True,
    )
    required = {"Fs", "ChannelNames", "ChannelUnits", "EEG", "EMG", "FrameTrigger"}
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(f"{path.name} is missing variables: {', '.join(missing)}")

    fs_values = np.asarray(raw["Fs"]).ravel()
    if fs_values.size != 1:
        raise ValueError(f"{path.name}: Fs must be a scalar")
    fs = float(fs_values[0])
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"{path.name}: Fs must be finite and positive")

    arrays = {
        name: np.asarray(raw[name], dtype=np.float32).ravel()
        for name in ("EEG", "EMG", "FrameTrigger")
    }
    lengths = {values.size for values in arrays.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError(f"{path.name}: EEG, EMG, and FrameTrigger lengths differ")
    for name, values in arrays.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path.name}: {name} contains non-finite samples")

    names = _cell_strings(raw["ChannelNames"])
    units = _cell_strings(raw["ChannelUnits"])
    if len(names) != len(units):
        raise ValueError(f"{path.name}: ChannelNames and ChannelUnits lengths differ")
    unit_by_name = dict(zip(names, units, strict=True))
    expected_units = {"FrameTrigger": "v", "EEG": "mv", "EMG": "mv"}
    for channel, expected_unit in expected_units.items():
        actual_unit = unit_by_name.get(channel)
        if actual_unit is None:
            raise ValueError(f"{path.name}: ChannelNames does not contain {channel}")
        if actual_unit.casefold() != expected_unit:
            raise ValueError(
                f"{path.name}: expected {channel} in {expected_unit}, "
                f"found {actual_unit!r}"
            )
    return {
        "fs": fs,
        **arrays,
        "eeg_unit": unit_by_name.get("EEG", "mV"),
        "emg_unit": unit_by_name.get("EMG", "mV"),
    }


def load_physiology(
    recording_name: str,
    directory: str | Path = PHYSIOLOGY_DIR,
) -> PhysiologyRecording:
    """Load only the channels needed for the synchronized inspection figure."""
    paths = physiology_paths(recording_name, directory)
    parts = [_load_part(path) for path in paths]
    fs = float(parts[0]["fs"])
    if any(not np.isclose(float(part["fs"]), fs) for part in parts[1:]):
        raise ValueError(f"{recording_name}: physiology parts use different Fs values")

    eeg_units = {str(part["eeg_unit"]) for part in parts}
    emg_units = {str(part["emg_unit"]) for part in parts}
    if len(eeg_units) != 1 or len(emg_units) != 1:
        raise ValueError(f"{recording_name}: physiology parts use different units")

    part_lengths = np.asarray([np.asarray(part["EEG"]).size for part in parts])
    part_boundaries = np.cumsum(part_lengths, dtype=np.int64)[:-1]

    def combine(name: str) -> np.ndarray:
        values = [np.asarray(part[name], dtype=np.float32) for part in parts]
        return values[0] if len(values) == 1 else np.concatenate(values)

    recording = PhysiologyRecording(
        name=recording_name,
        fs=fs,
        eeg=combine("EEG"),
        emg=combine("EMG"),
        frame_trigger=combine("FrameTrigger"),
        part_boundaries=part_boundaries,
        source_paths=paths,
        eeg_unit=eeg_units.pop(),
        emg_unit=emg_units.pop(),
    )
    if not (
        recording.eeg.size
        == recording.emg.size
        == recording.frame_trigger.size
    ):
        raise ValueError(f"{recording_name}: concatenated physiology lengths differ")
    return recording


def frame_trigger_edges(
    frame_trigger: np.ndarray,
    threshold_v: float = 1.0,
) -> np.ndarray:
    """Detect rising TTL edges using the documented trigger voltage units."""
    values = np.asarray(frame_trigger)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("frame_trigger must be a one-dimensional signal")
    if not np.all(np.isfinite(values)):
        raise ValueError("frame_trigger contains non-finite samples")

    if not np.isfinite(threshold_v):
        raise ValueError("threshold_v must be finite")
    if float(np.min(values)) >= threshold_v or float(np.max(values)) <= threshold_v:
        raise ValueError("frame_trigger does not cross threshold_v")
    high = values > threshold_v
    previous = np.r_[False, high[:-1]]
    edges = np.flatnonzero(high & ~previous).astype(np.int64)
    if edges.size < 2:
        raise ValueError("frame_trigger contains fewer than two rising edges")
    return edges


def frame_trigger_groups(
    recording: PhysiologyRecording,
    gap_factor: float = 2.0,
) -> tuple[tuple[np.ndarray, ...], float]:
    """Split rising edges at long acquisition gaps and source-file boundaries."""
    if not np.isfinite(gap_factor) or gap_factor <= 1:
        raise ValueError("gap_factor must be finite and greater than one")
    edges = frame_trigger_edges(recording.frame_trigger)
    periods = np.diff(edges)
    nominal_period = float(np.median(periods))
    split_after = periods > gap_factor * nominal_period
    if recording.part_boundaries.size:
        part_index = np.searchsorted(
            recording.part_boundaries,
            edges,
            side="right",
        )
        split_after |= np.diff(part_index) != 0
    groups = tuple(np.split(edges, np.flatnonzero(split_after) + 1))
    return groups, nominal_period


def align_frame_triggers(
    recording: PhysiologyRecording,
    n_frames: int,
    boundary_ind: np.ndarray,
    imaging_fs: float,
) -> PhysiologyAlignment:
    """Pair calcium acquisition segments with the released trigger bouts.

    The ten released recordings use the explicit, audited selections above.
    For an unknown recording name, short calibration bouts are skipped and the
    prefix of each next adequate bout is used.  Surplus pulses are always
    reported rather than silently treated as calcium frames.
    """
    if isinstance(n_frames, (bool, np.bool_)) or not isinstance(
        n_frames, (int, np.integer)
    ):
        raise TypeError("n_frames must be an integer")
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if not np.isfinite(imaging_fs) or imaging_fs <= 0:
        raise ValueError("imaging_fs must be finite and positive")

    frame_segments = ts.acquisition_segments(n_frames, boundary_ind)
    trigger_groups, nominal_period = frame_trigger_groups(recording)
    expected_period = recording.fs / imaging_fs
    if abs(nominal_period - expected_period) / expected_period > 0.02:
        raise ValueError(
            f"FrameTrigger period {nominal_period:.1f} samples is inconsistent "
            f"with physiology Fs={recording.fs:g} Hz and imaging Fs={imaging_fs:g} Hz"
        )

    selections = _RELEASED_TRIGGER_SELECTIONS.get(recording.name)
    expected_group_sizes = _RELEASED_TRIGGER_GROUP_SIZES.get(recording.name)
    observed_group_sizes = tuple(int(group.size) for group in trigger_groups)
    if (
        expected_group_sizes is not None
        and recording.source_paths
        and observed_group_sizes != expected_group_sizes
    ):
        raise ValueError(
            f"{recording.name}: expected released FrameTrigger bout sizes "
            f"{expected_group_sizes}, found {observed_group_sizes}"
        )
    if selections is not None and len(selections) != len(frame_segments):
        raise ValueError(
            f"{recording.name}: expected {len(selections)} calcium segments from "
            f"the release manifest, found {len(frame_segments)}"
        )

    aligned: list[AlignedPhysiologySegment] = []
    next_group_index = 0
    used_group_indices: set[int] = set()
    for segment_index, (frame_start, frame_stop) in enumerate(frame_segments):
        segment_frames = frame_stop - frame_start
        if selections is None:
            while (
                next_group_index < len(trigger_groups)
                and trigger_groups[next_group_index].size < segment_frames
            ):
                next_group_index += 1
            group_index = next_group_index
            anchor = "start"
            next_group_index += 1
        else:
            group_index, anchor = selections[segment_index]

        if group_index >= len(trigger_groups):
            sizes = [int(group.size) for group in trigger_groups]
            raise ValueError(
                f"{recording.name}: trigger selection requests bout {group_index}, "
                f"but detected bout sizes are {sizes}"
            )
        if group_index in used_group_indices:
            raise ValueError(f"Trigger bout {group_index} was selected more than once")
        used_group_indices.add(group_index)

        group = trigger_groups[group_index]
        if group.size < segment_frames:
            sizes = [int(candidate.size) for candidate in trigger_groups]
            raise ValueError(
                "Could not match calcium acquisition segments to FrameTrigger "
                f"bouts. Needed {segment_frames} frames; detected bout sizes {sizes}."
            )
        if anchor == "start":
            trigger_start_offset = 0
        elif anchor == "end":
            trigger_start_offset = int(group.size - segment_frames)
        else:
            raise ValueError(f"Unknown trigger selection anchor: {anchor!r}")
        selected = group[
            trigger_start_offset : trigger_start_offset + segment_frames
        ]
        group_period = (
            float(np.median(np.diff(group))) if group.size > 1 else nominal_period
        )
        sample_stop = min(
            recording.n_samples,
            int(round(float(selected[-1]) + group_period)),
        )
        aligned.append(
            AlignedPhysiologySegment(
                frame_start=frame_start,
                frame_stop=frame_stop,
                sample_start=int(selected[0]),
                sample_stop=sample_stop,
                trigger_group_index=group_index,
                trigger_group_size=int(group.size),
                trigger_start_offset=trigger_start_offset,
                trigger_samples=selected.copy(),
            )
        )

    detected_count = sum(int(group.size) for group in trigger_groups)
    return PhysiologyAlignment(
        segments=tuple(aligned),
        detected_trigger_count=detected_count,
        mapped_trigger_count=n_frames,
        nominal_period_samples=nominal_period,
    )


def _resample_to(values: np.ndarray, source_fs: float, target_fs: float) -> np.ndarray:
    if not np.isfinite(target_fs) or target_fs <= 0:
        raise ValueError("target_fs must be finite and positive")
    ratio = Fraction(float(target_fs) / float(source_fs)).limit_denominator(10_000)
    return signal.resample_poly(values, ratio.numerator, ratio.denominator)


def _recorded_time_min(
    aligned: AlignedPhysiologySegment,
    sample_offsets_s: np.ndarray,
    physiology_fs: float,
    imaging_fs: float,
) -> np.ndarray:
    """Warp physiological sample times through every selected frame trigger."""
    query_samples = aligned.sample_start + sample_offsets_s * physiology_fs
    sample_anchors = np.r_[aligned.trigger_samples, aligned.sample_stop]
    frame_anchors_s = np.r_[
        np.arange(aligned.frame_start, aligned.frame_stop) / imaging_fs,
        aligned.frame_stop / imaging_fs,
    ]
    return np.interp(query_samples, sample_anchors, frame_anchors_s) / 60


def prepare_eeg_spectrogram(
    recording: PhysiologyRecording,
    alignment: PhysiologyAlignment,
    imaging_fs: float,
    target_fs: float = 250.0,
    min_frequency_hz: float = 0.5,
    max_frequency_hz: float = 25.0,
    filter_high_hz: float | None = None,
    window_seconds: float = 4.0,
    overlap_fraction: float = 0.75,
) -> tuple[tuple[EEGSpectrogramPanel, ...], tuple[float, float]]:
    """Prepare segment-local EEG power without bridging acquisition gaps."""
    if not np.isfinite(target_fs) or target_fs <= 0:
        raise ValueError("target_fs must be finite and positive")
    if (
        not np.isfinite(min_frequency_hz)
        or not np.isfinite(max_frequency_hz)
        or min_frequency_hz <= 0
        or max_frequency_hz <= min_frequency_hz
    ):
        raise ValueError("EEG display frequencies must satisfy 0 < min < max")
    if filter_high_hz is None:
        filter_high_hz = min(
            0.45 * target_fs,
            max_frequency_hz + max(5.0, 0.2 * max_frequency_hz),
        )
    if (
        not np.isfinite(filter_high_hz)
        or filter_high_hz <= max_frequency_hz
        or target_fs <= 2 * filter_high_hz
    ):
        raise ValueError(
            "filter_high_hz must exceed max_frequency_hz and remain below Nyquist"
        )
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be finite and positive")
    if not np.isfinite(overlap_fraction) or not 0 <= overlap_fraction < 1:
        raise ValueError("overlap_fraction must be in [0, 1)")

    panels: list[EEGSpectrogramPanel] = []
    all_power: list[np.ndarray] = []
    requested_window = max(8, int(round(window_seconds * target_fs)))
    filter_sos = signal.butter(
        4,
        (min_frequency_hz, filter_high_hz),
        btype="bandpass",
        fs=target_fs,
        output="sos",
    )
    for aligned in alignment.segments:
        raw = recording.eeg[aligned.sample_start : aligned.sample_stop]
        downsampled = _resample_to(raw, recording.fs, target_fs)
        filtered = signal.sosfiltfilt(filter_sos, downsampled)
        nperseg = min(requested_window, filtered.size)
        if nperseg < 8:
            raise ValueError("An aligned EEG segment is too short for a spectrogram")
        noverlap = min(int(round(overlap_fraction * nperseg)), nperseg - 1)
        frequencies, seconds, power = signal.spectrogram(
            filtered,
            fs=target_fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
            mode="psd",
        )
        keep = (frequencies >= min_frequency_hz) & (
            frequencies <= max_frequency_hz
        )
        power = power[keep]
        floor = np.finfo(power.dtype).tiny
        power_db = 10 * np.log10(np.maximum(power, floor))

        time_min = _recorded_time_min(
            aligned,
            seconds,
            recording.fs,
            imaging_fs,
        )
        panel = EEGSpectrogramPanel(
            time_min=time_min,
            frequency_hz=frequencies[keep],
            power_db_hz=power_db,
        )
        panels.append(panel)
        all_power.append(power_db.ravel())

    finite_power = np.concatenate(all_power)
    finite_power = finite_power[np.isfinite(finite_power)]
    if finite_power.size == 0:
        raise ValueError("EEG spectrogram contains no finite power values")
    vmin, vmax = np.percentile(finite_power, (2.0, 99.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(finite_power.min()), float(finite_power.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    return tuple(panels), (float(vmin), float(vmax))


def prepare_emg_envelope(
    recording: PhysiologyRecording,
    alignment: PhysiologyAlignment,
    imaging_fs: float,
    target_fs: float = 500.0,
    filter_band_hz: tuple[float, float] = (20.0, 200.0),
    rms_window_seconds: float = 0.25,
    display_fs: float = 20.0,
) -> tuple[tuple[EMGEnvelopePanel, ...], float]:
    """Prepare a band-passed RMS EMG envelope for long-session display."""
    low_hz, high_hz = filter_band_hz
    if (
        not np.isfinite(low_hz)
        or not np.isfinite(high_hz)
        or low_hz <= 0
        or high_hz <= low_hz
        or target_fs <= 2 * high_hz
    ):
        raise ValueError("EMG filter band must be positive, ordered, and below Nyquist")
    if not np.isfinite(rms_window_seconds) or rms_window_seconds <= 0:
        raise ValueError("rms_window_seconds must be finite and positive")
    if not np.isfinite(display_fs) or display_fs <= 0 or display_fs > target_fs:
        raise ValueError("display_fs must be positive and no greater than target_fs")

    filter_sos = signal.butter(
        4,
        filter_band_hz,
        btype="bandpass",
        fs=target_fs,
        output="sos",
    )
    rms_samples = max(1, int(round(rms_window_seconds * target_fs)))
    rms_kernel = np.full(rms_samples, 1 / rms_samples)
    display_step = max(1, int(round(target_fs / display_fs)))
    panels: list[EMGEnvelopePanel] = []
    amplitudes: list[np.ndarray] = []

    for aligned in alignment.segments:
        raw = recording.emg[aligned.sample_start : aligned.sample_stop]
        downsampled = _resample_to(raw, recording.fs, target_fs)
        filtered = signal.sosfiltfilt(filter_sos, downsampled)
        mean_square = signal.convolve(
            np.square(filtered),
            rms_kernel,
            mode="same",
            method="auto",
        )
        rms = np.sqrt(np.maximum(mean_square, 0))
        display_indices = np.arange(0, rms.size, display_step, dtype=np.int64)
        amplitude = rms[display_indices]

        centers_s = display_indices / target_fs
        time_min = _recorded_time_min(
            aligned,
            centers_s,
            recording.fs,
            imaging_fs,
        )
        panels.append(
            EMGEnvelopePanel(
                time_min=time_min,
                amplitude=amplitude,
            )
        )
        amplitudes.append(amplitude)

    amplitude_limit = float(np.percentile(np.concatenate(amplitudes), 99.5))
    if not np.isfinite(amplitude_limit) or amplitude_limit <= 0:
        amplitude_limit = 1.0
    return tuple(panels), amplitude_limit


def _band_power(
    power: np.ndarray,
    frequency_hz: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    """Sum one open-interval band as in the cited archived scorer."""
    keep = (frequency_hz > low_hz) & (frequency_hz < high_hz)
    if np.count_nonzero(keep) < 2:
        raise ValueError(
            f"Frequency grid has fewer than two bins in {low_hz:g}-{high_hz:g} Hz"
        )
    return np.sum(power[..., keep], axis=-1)


def prepare_state_scoring_features(
    recording: PhysiologyRecording,
    alignment: PhysiologyAlignment,
    imaging_fs: float,
    analysis_fs: float = 1000.0,
    window_seconds: float = 4.0,
    display_frequency_hz: tuple[float, float] = (0.5, 25.0),
    delta_band_hz: tuple[float, float] = (1.0, 4.0),
    theta_band_hz: tuple[float, float] = (6.0, 9.0),
    total_band_hz: tuple[float, float] = (1.0, 50.0),
    chunk_windows: int = 512,
) -> tuple[tuple[StateScoringFeaturePanel, ...], float]:
    """Reconstruct the paper's 4-s sleep-scoring features at imaging frames.

    Windows are centered on the exact selected ``FrameTrigger`` edges and are
    dropped when they would cross an acquisition boundary.  The deposited
    amplifier outputs are resampled to the paper-reported 1-kHz analysis rate,
    but are not digitally band-pass filtered a second time.  A Hann periodogram
    supplies delta, theta, and total EEG power.  The paper defines normalized
    delta as D/N and normalized theta as T/D; its archived cited implementation
    then calculates "delta/theta" as (D/N)/(T/D).  The plain D/T band-power
    ratio is retained separately as a sensitivity feature.  EMG RMS is
    calculated directly from the amplifier output, matching that archive.

    The spectrogram is expressed as dB change from each frequency's median over
    the complete recording.  This normalization affects display only; relative
    delta and delta/theta features use the unnormalized power spectrum before
    the explicit ratios above are formed.
    """
    numeric_settings = {
        "imaging_fs": imaging_fs,
        "analysis_fs": analysis_fs,
        "window_seconds": window_seconds,
    }
    for name, value in numeric_settings.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if isinstance(chunk_windows, (bool, np.bool_)) or not isinstance(
        chunk_windows,
        (int, np.integer),
    ):
        raise TypeError("chunk_windows must be an integer")
    if chunk_windows <= 0:
        raise ValueError("chunk_windows must be positive")

    bands = {
        "display": display_frequency_hz,
        "delta": delta_band_hz,
        "theta": theta_band_hz,
        "total": total_band_hz,
    }
    for name, band in bands.items():
        if len(band) != 2:
            raise ValueError(f"{name} band must be a (low, high) pair")
        low_hz, high_hz = map(float, band)
        if (
            not np.isfinite(low_hz)
            or not np.isfinite(high_hz)
            or low_hz < 0
            or high_hz <= low_hz
        ):
            raise ValueError(f"{name} band must satisfy 0 <= low < high")
    required_high_hz = max(
        display_frequency_hz[1],
        delta_band_hz[1],
        theta_band_hz[1],
        total_band_hz[1],
        300.0,
    )
    if analysis_fs <= 2 * required_high_hz:
        raise ValueError(
            "analysis_fs must exceed twice the highest EEG/EMG frequency "
            f"({required_high_hz:g} Hz)"
        )

    requested_window = int(round(window_seconds * analysis_fs))
    half_window = requested_window // 2
    window_samples = 2 * half_window
    if window_samples < 8:
        raise ValueError("window_seconds is too short for spectral analysis")
    frequency_hz = np.fft.rfftfreq(window_samples, d=1 / analysis_fs)
    display_keep = (frequency_hz >= display_frequency_hz[0]) & (
        frequency_hz <= display_frequency_hz[1]
    )
    if np.count_nonzero(display_keep) < 2:
        raise ValueError("Display band contains fewer than two frequency bins")

    hann = signal.windows.hann(window_samples, sym=True).astype(np.float32)
    periodogram_scale = float(analysis_fs * np.sum(np.square(hann)))
    sample_offsets = np.arange(-half_window, half_window, dtype=np.int64)
    tiny = np.finfo(np.float32).tiny
    raw_panels: list[dict[str, object]] = []

    for segment_index, aligned in enumerate(alignment.segments):
        eeg = _resample_to(
            recording.eeg[aligned.sample_start : aligned.sample_stop],
            recording.fs,
            analysis_fs,
        ).astype(np.float32, copy=False)
        emg = _resample_to(
            recording.emg[aligned.sample_start : aligned.sample_stop],
            recording.fs,
            analysis_fs,
        ).astype(np.float32, copy=False)
        center_samples = np.rint(
            (aligned.trigger_samples - aligned.sample_start)
            * analysis_fs
            / recording.fs
        ).astype(np.int64)
        valid = (center_samples - half_window >= 0) & (
            center_samples + half_window <= min(eeg.size, emg.size)
        )
        if not np.any(valid):
            raise ValueError(
                f"Aligned segment {segment_index} is shorter than one centered "
                f"{window_seconds:g}-s feature window"
            )

        centers = center_samples[valid]
        frame_index = np.arange(
            aligned.frame_start,
            aligned.frame_stop,
            dtype=np.int64,
        )[valid]
        trigger_sample = aligned.trigger_samples[valid].astype(np.int64, copy=True)
        n_windows = centers.size
        display_power_db = np.empty(
            (np.count_nonzero(display_keep), n_windows),
            dtype=np.float32,
        )
        relative_delta = np.empty(n_windows, dtype=np.float32)
        delta_theta_ratio = np.empty(n_windows, dtype=np.float32)
        raw_delta_theta_ratio = np.empty(n_windows, dtype=np.float32)
        emg_rms = np.empty(n_windows, dtype=np.float32)

        for start in range(0, n_windows, int(chunk_windows)):
            stop = min(start + int(chunk_windows), n_windows)
            indices = centers[start:stop, np.newaxis] + sample_offsets

            eeg_windows = eeg[indices]
            spectrum = np.fft.rfft(eeg_windows * hann, axis=1)
            power = np.square(np.abs(spectrum)) / periodogram_scale
            if power.shape[1] > 2:
                power[:, 1:-1] *= 2

            delta = _band_power(
                power,
                frequency_hz,
                *delta_band_hz,
            )
            theta = _band_power(
                power,
                frequency_hz,
                *theta_band_hz,
            )
            total = _band_power(
                power,
                frequency_hz,
                *total_band_hz,
            )
            relative_delta_chunk = delta / np.maximum(total, tiny)
            raw_ratio_chunk = delta / np.maximum(theta, tiny)
            relative_delta[start:stop] = relative_delta_chunk
            raw_delta_theta_ratio[start:stop] = raw_ratio_chunk
            delta_theta_ratio[start:stop] = relative_delta_chunk * raw_ratio_chunk
            display_power_db[:, start:stop] = (
                10 * np.log10(np.maximum(power[:, display_keep], tiny))
            ).T

            emg_windows = emg[indices]
            emg_rms[start:stop] = np.sqrt(
                np.mean(np.square(emg_windows), axis=1)
            )

        raw_panels.append(
            {
                "segment_index": segment_index,
                "frame_index": frame_index,
                "trigger_sample": trigger_sample,
                "time_min": frame_index / imaging_fs / 60,
                "frequency_hz": frequency_hz[display_keep].copy(),
                "power_db": display_power_db,
                "relative_delta": relative_delta,
                "delta_theta_ratio": delta_theta_ratio,
                "raw_delta_theta_ratio": raw_delta_theta_ratio,
                "emg_rms": emg_rms,
            }
        )

    all_power_db = np.concatenate(
        [np.asarray(panel["power_db"]) for panel in raw_panels],
        axis=1,
    )
    frequency_baseline_db = np.nanmedian(all_power_db, axis=1, keepdims=True)
    normalized_panels = [
        np.asarray(panel["power_db"]) - frequency_baseline_db
        for panel in raw_panels
    ]
    finite_change = np.concatenate(
        [np.ravel(values) for values in normalized_panels]
    )
    finite_change = finite_change[np.isfinite(finite_change)]
    if finite_change.size == 0:
        raise ValueError("Frequency-normalized spectrogram has no finite values")
    color_limit = float(np.percentile(np.abs(finite_change), 98.5))
    if not np.isfinite(color_limit) or color_limit <= 0:
        color_limit = 1.0

    panels = tuple(
        StateScoringFeaturePanel(
            segment_index=int(panel["segment_index"]),
            frame_index=np.asarray(panel["frame_index"]),
            trigger_sample=np.asarray(panel["trigger_sample"]),
            time_min=np.asarray(panel["time_min"]),
            frequency_hz=np.asarray(panel["frequency_hz"]),
            frequency_normalized_power_db=np.asarray(normalized_power),
            relative_delta=np.asarray(panel["relative_delta"]),
            delta_theta_ratio=np.asarray(panel["delta_theta_ratio"]),
            raw_delta_theta_ratio=np.asarray(panel["raw_delta_theta_ratio"]),
            emg_rms=np.asarray(panel["emg_rms"]),
        )
        for panel, normalized_power in zip(
            raw_panels,
            normalized_panels,
            strict=True,
        )
    )
    return panels, color_limit


def fit_emg_threshold(
    emg_rms: np.ndarray,
    reference_labels: np.ndarray,
) -> EMGThresholdFit:
    """Calibrate the missing manual EMG threshold from W/NREM/REM labels.

    Quiet-awake and any other labels are deliberately ignored.  The chosen
    threshold maximizes balanced accuracy for awake versus NREM/REM.  This is a
    label-calibrated reconstruction of the unpublished manual setting, not an
    independent validation of the deposited labels.
    """
    values = np.asarray(emg_rms, dtype=float)
    labels = np.asarray(reference_labels, dtype=str)
    if values.ndim != 1 or labels.ndim != 1 or values.size != labels.size:
        raise ValueError("emg_rms and reference_labels must be aligned 1-D arrays")
    eligible = np.isfinite(values) & (values >= 0) & np.isin(
        labels,
        ("awake", "nrem", "rem"),
    )
    values = values[eligible]
    labels = labels[eligible]
    awake = labels == "awake"
    sleep = np.isin(labels, ("nrem", "rem"))
    if not np.any(awake) or not np.any(sleep):
        raise ValueError("Threshold fitting requires both awake and sleep windows")

    unique = np.unique(values)
    if unique.size == 1:
        candidates = unique.copy()
    else:
        positive = unique > 0
        if np.all(positive):
            middle = np.sqrt(unique[:-1] * unique[1:])
        else:
            middle = (unique[:-1] + unique[1:]) / 2
        candidates = np.r_[
            np.nextafter(unique[0], -np.inf),
            middle,
            np.nextafter(unique[-1], np.inf),
        ]
    scores = np.asarray(
        [
            0.5
            * (
                np.mean(values[awake] >= threshold)
                + np.mean(values[sleep] < threshold)
            )
            for threshold in candidates
        ]
    )
    best = np.flatnonzero(np.isclose(scores, np.max(scores), rtol=0, atol=1e-12))
    selected = int(best[best.size // 2])
    return EMGThresholdFit(
        threshold=float(candidates[selected]),
        balanced_accuracy=float(scores[selected]),
        n_awake=int(np.count_nonzero(awake)),
        n_sleep=int(np.count_nonzero(sleep)),
    )


def classify_sleep_windows(
    emg_rms: np.ndarray,
    delta_theta_ratio: np.ndarray,
    emg_threshold: float,
    ratio_boundary: float = 0.3,
) -> np.ndarray:
    """Apply the paper's automatic W/NREM/REM decision sequence."""
    emg_rms = np.asarray(emg_rms, dtype=float)
    ratio = np.asarray(delta_theta_ratio, dtype=float)
    if emg_rms.ndim != 1 or ratio.ndim != 1 or emg_rms.shape != ratio.shape:
        raise ValueError("emg_rms and delta_theta_ratio must be aligned 1-D arrays")
    if np.any(~np.isfinite(emg_rms)) or np.any(emg_rms < 0):
        raise ValueError("emg_rms must contain finite non-negative values")
    if np.any(~np.isfinite(ratio)) or np.any(ratio < 0):
        raise ValueError("delta_theta_ratio must contain finite non-negative values")
    if not np.isfinite(emg_threshold) or emg_threshold < 0:
        raise ValueError("emg_threshold must be finite and non-negative")
    if not np.isfinite(ratio_boundary) or ratio_boundary <= 0:
        raise ValueError("ratio_boundary must be finite and positive")

    predicted = np.full(emg_rms.size, "rem", dtype="<U5")
    high_emg = emg_rms >= emg_threshold
    predicted[~high_emg & (ratio >= ratio_boundary)] = "nrem"
    predicted[high_emg] = "awake"
    return predicted


def merge_short_state_runs(
    labels: np.ndarray,
    min_run_samples: int,
) -> np.ndarray:
    """Merge runs shorter than a declared duration using a label-free rule.

    Matching neighbors take precedence.  Otherwise the short run joins the
    longer adjacent run, with ties assigned to the preceding state.  The paper
    did not disclose how it resolved unlike neighboring states, so callers must
    present this deterministic cleanup as an approximation.
    """
    values = np.asarray(labels, dtype=str)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    if isinstance(min_run_samples, (bool, np.bool_)) or not isinstance(
        min_run_samples,
        (int, np.integer),
    ):
        raise TypeError("min_run_samples must be an integer")
    if min_run_samples <= 0:
        raise ValueError("min_run_samples must be positive")

    merged = values.copy()
    while True:
        changes = np.flatnonzero(merged[1:] != merged[:-1]) + 1
        starts = np.r_[0, changes]
        stops = np.r_[changes, merged.size]
        short = np.flatnonzero((stops - starts) < min_run_samples)
        if short.size == 0 or starts.size == 1:
            break
        run_index = int(short[np.argmin((stops - starts)[short])])
        if run_index == 0:
            replacement = merged[starts[run_index + 1]]
        elif run_index == starts.size - 1:
            replacement = merged[starts[run_index - 1]]
        else:
            previous = merged[starts[run_index - 1]]
            following = merged[starts[run_index + 1]]
            if previous == following:
                replacement = previous
            else:
                previous_length = stops[run_index - 1] - starts[run_index - 1]
                following_length = stops[run_index + 1] - starts[run_index + 1]
                replacement = (
                    following if following_length > previous_length else previous
                )
        merged[starts[run_index] : stops[run_index]] = replacement
    return merged


__all__ = [
    "AlignedPhysiologySegment",
    "EEGSpectrogramPanel",
    "EMGThresholdFit",
    "EMGEnvelopePanel",
    "PhysiologyAlignment",
    "PhysiologyRecording",
    "StateScoringFeaturePanel",
    "align_frame_triggers",
    "classify_sleep_windows",
    "fit_emg_threshold",
    "frame_trigger_edges",
    "frame_trigger_groups",
    "load_physiology",
    "merge_short_state_runs",
    "physiology_paths",
    "prepare_eeg_spectrogram",
    "prepare_emg_envelope",
    "prepare_state_scoring_features",
]
