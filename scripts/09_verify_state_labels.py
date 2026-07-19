# %% [markdown]
# # 09 · Can the published EEG/EMG rules reproduce the state labels?
#
# This tutorial is deliberately separate from the inspection figure.  It tests
# agreement between the state annotations deposited by Kiyooka & Oomoto et al.
# and the numerical portion of their published sleep-scoring method:
#
# - 4-s sliding windows;
# - awake when EMG RMS exceeds a manually defined threshold (approximated per
#   recording here);
# - otherwise NREM when the paper's normalized delta/theta feature exceeds 0.3
#   and REM when it is below;
# - manual correction of quiet wakefulness; and
# - removal/merging of state episodes shorter than 12 s.
#
# The manual EMG thresholds, precise spectral estimator, sliding-window step,
# cleanup tie rule, and quiet-awake decisions were not released.  Therefore,
# this is an agreement audit—not an independent reconstruction of ground truth.
# The script calibrates the missing EMG threshold from training blocks and
# evaluates held-out temporal blocks.  Quiet awake is never used for fitting and
# is reported separately because the paper says it was corrected manually.
#
# Anesthesia is different.  The final paper operationally defines it as the
# 0.6%-isoflurane condition after 20–60 min of wake recording.  The release says
# annotations were manually inspected against EEG/EMG, but no anesthesia EEG or
# EMG classifier is given.  For anesthesia sessions, this script consequently
# reports physiological separation without inventing a classification rule.
#
# Primary source: Kiyooka & Oomoto et al., Cell Reports (2026), STAR Methods,
# “In vivo two-photon calcium imaging…” and “EEG and EMG recording and analysis”
# https://doi.org/10.1016/j.celrep.2025.116902
# Formula cross-check: the archived scorer for cited reference 101 computes
# (delta / 1–50 Hz) / (theta / delta), not the simpler raw delta/theta ratio:
# https://doi.org/10.5281/zenodo.14591051
# That earlier scorer used non-overlapping epochs and a particular causal
# cleanup.  We use it only to resolve the feature algebra: the Kiyooka paper
# instead says "sliding window" and does not release its step or cleanup code.

# %% Step 0 — imports
import gc
import os
import sys
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from pymatreader import read_mat
from scipy import stats

from src.funcnet import dataio, physiology as physio, visualization as viz
from src.funcnet.paths import FIG_DIR, RAW_DIR, RESULTS_DIR

FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Step 1 — fixed settings
#
# The physiological release stores 5-kHz signals, whereas the paper reports
# 1-kHz digitization.  `ANALYSIS_FS = 1000` matches the stated method by
# polyphase-downsampling the deposited amplifier outputs inside each acquisition
# segment.  We do not add another EEG/EMG band-pass filter: the paper says the
# signals were already analog-filtered at acquisition (EEG 0.1–100 Hz; EMG
# 5–300 Hz), and an undocumented second filter would change the verification.
#
# The paper says “4-s sliding window” without its step or centering convention.
# We use one centered window per selected imaging-frame trigger, discard the
# first/last 2 s of every acquisition, and exclude transition-straddling windows
# from the primary metrics.  The fixed 0.3 rule below is never optimized.

# %%
SLEEP_RECORDINGS = (
    "mouse01_sleep",
    "mouse02_sleep",
    "mouse03_sleep",
    "mouse04_day1_sleep",
    "mouse04_day2_sleep",
    "mouse05_sleep",
)
ANESTHESIA_RECORDINGS = (
    "mouse03_ane",
    "mouse05_ane",
    "mouse06_ane",
    "mouse07_ane",
)

ANALYSIS_FS = 1000.0
WINDOW_SECONDS = 4.0
DELTA_THETA_BOUNDARY = 0.3
SHORT_RUN_SECONDS = 12.0
CV_BLOCK_SECONDS = 60.0
CV_FOLDS = 5
DISPLAY_FREQUENCY_HZ = (0.5, 25.0)
SHOW_FIGURES = True

SCORABLE_SLEEP_LABELS = ("awake", "nrem", "rem")
STATE_ORDER = ("awake", "quiet_awake", "nrem", "rem", "anesthesia")


@dataclass(frozen=True)
class TimingMetadata:
    """Small calcium-file subset needed for physiology-label verification."""

    name: str
    data_info: str
    fs: float
    state: np.ndarray
    boundary_ind: np.ndarray
    used_frame: tuple[np.ndarray, ...]
    codes: dict[float, str]

    @property
    def n_frames(self) -> int:
        return int(self.state.size)


def load_timing_metadata(recording_name: str) -> TimingMetadata:
    """Read state/timing fields without loading the three large calcium arrays."""
    path = RAW_DIR / f"{recording_name}.mat"
    raw = read_mat(path, variable_names=("data_info", "state", "frame"))
    data_info = str(raw["data_info"])
    if data_info not in {"sleep", "ane"}:
        raise ValueError(f"Unexpected data_info={data_info!r} in {path.name}")
    state = np.asarray(raw["state"], dtype=float).ravel()
    frame = raw["frame"]
    boundary_ind = (
        np.atleast_1d(np.asarray(frame.get("boundary_ind", []))).ravel().astype(int)
        - 1
    )
    used_raw = frame.get("used_frame", [])
    used_list = list(used_raw) if isinstance(used_raw, (list, tuple)) else [used_raw]
    used_frame = tuple(
        np.atleast_1d(np.asarray(values)).ravel().astype(int) - 1
        for values in used_list
    )
    codes = (
        dict(dataio.SLEEP_STATE_CODES)
        if data_info == "sleep"
        else dict(dataio.ANE_STATE_CODES)
    )
    unknown = np.setdiff1d(np.unique(state), list(codes))
    if unknown.size:
        raise ValueError(f"{path.name}: unknown state codes {unknown.tolist()}")
    return TimingMetadata(
        name=recording_name,
        data_info=data_info,
        fs=dataio.FS_HZ,
        state=state,
        boundary_ind=boundary_ind,
        used_frame=used_frame,
        codes=codes,
    )


# %% [markdown]
# ## Step 2 — window labels, temporal folds, and metrics
#
# Threshold calibration is the only label-dependent part of the reconstructed
# rule.  Five interleaved **60-s contiguous blocks** are used as folds.  The
# first/last 2 s of every block are purged so 4-s raw-signal windows in training
# and test folds do not overlap.  This is stricter than fitting the unpublished
# threshold on the whole session and then reporting the same labels as a test.

# %%
def stable_window_mask(
    state: np.ndarray,
    frame_index: np.ndarray,
    frame_start: int,
    frame_stop: int,
    fs: float,
    window_seconds: float,
) -> np.ndarray:
    """Mark centered windows whose deposited state is constant throughout."""
    half_frames = int(np.ceil(window_seconds * fs / 2))
    stable = np.empty(frame_index.size, dtype=bool)
    for index, center in enumerate(frame_index):
        start = max(frame_start, int(center) - half_frames)
        stop = min(frame_stop, int(center) + half_frames + 1)
        stable[index] = np.all(np.isclose(state[start:stop], state[int(center)]))
    return stable


def feature_table(
    timing: TimingMetadata,
    alignment: physio.PhysiologyAlignment,
    panels: tuple[physio.StateScoringFeaturePanel, ...],
) -> pd.DataFrame:
    """Assemble one auditable row per valid centered 4-s feature window."""
    rows: list[pd.DataFrame] = []
    for panel, aligned in zip(panels, alignment.segments, strict=True):
        labels = np.asarray(
            [timing.codes[float(timing.state[frame])] for frame in panel.frame_index]
        )
        stable = stable_window_mask(
            timing.state,
            panel.frame_index,
            aligned.frame_start,
            aligned.frame_stop,
            timing.fs,
            WINDOW_SECONDS,
        )
        local_seconds = (panel.frame_index - aligned.frame_start) / timing.fs
        block = np.floor(local_seconds / CV_BLOCK_SECONDS).astype(int)
        block_start = block * CV_BLOCK_SECONDS
        segment_seconds = (aligned.frame_stop - aligned.frame_start) / timing.fs
        block_stop = np.minimum(block_start + CV_BLOCK_SECONDS, segment_seconds)
        fold_interior = (local_seconds - block_start >= WINDOW_SECONDS / 2) & (
            block_stop - local_seconds >= WINDOW_SECONDS / 2
        )
        rows.append(
            pd.DataFrame(
                {
                    "recording": timing.name,
                    "animal": timing.name.split("_")[0],
                    "paradigm": timing.data_info,
                    "segment": panel.segment_index,
                    "frame": panel.frame_index,
                    "trigger_sample": panel.trigger_sample,
                    "time_min": panel.time_min,
                    "reference_label": labels,
                    "stable_window": stable,
                    "cv_block": block,
                    "cv_fold": block % CV_FOLDS,
                    "fold_interior": fold_interior,
                    "relative_delta": panel.relative_delta,
                    "delta_theta_ratio": panel.delta_theta_ratio,
                    "raw_delta_theta_ratio": panel.raw_delta_theta_ratio,
                    "emg_rms_mv": panel.emg_rms,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def confusion_and_metrics(
    reference: np.ndarray,
    predicted: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return a W/NREM/REM confusion matrix and prevalence-aware metrics."""
    reference = np.asarray(reference, dtype=str)
    predicted = np.asarray(predicted, dtype=str)
    if reference.shape != predicted.shape or reference.ndim != 1:
        raise ValueError("reference and predicted labels must be aligned 1-D arrays")
    matrix = np.zeros(
        (len(SCORABLE_SLEEP_LABELS), len(SCORABLE_SLEEP_LABELS)),
        dtype=np.int64,
    )
    for row, actual in enumerate(SCORABLE_SLEEP_LABELS):
        for column, estimate in enumerate(SCORABLE_SLEEP_LABELS):
            matrix[row, column] = np.count_nonzero(
                (reference == actual) & (predicted == estimate)
            )
    total = int(matrix.sum())
    if total == 0:
        raise ValueError("No scorable predictions were supplied")
    diagonal = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1).astype(float)
    predicted_count = matrix.sum(axis=0).astype(float)
    recall = np.divide(
        diagonal,
        support,
        out=np.full_like(diagonal, np.nan),
        where=support > 0,
    )
    precision = np.divide(
        diagonal,
        predicted_count,
        out=np.zeros_like(diagonal),
        where=predicted_count > 0,
    )
    f1 = np.full_like(diagonal, np.nan)
    reference_present = support > 0
    f1_denominator = precision + recall
    f1[reference_present] = np.divide(
        2 * precision[reference_present] * recall[reference_present],
        f1_denominator[reference_present],
        out=np.zeros(np.count_nonzero(reference_present), dtype=float),
        where=f1_denominator[reference_present] > 0,
    )
    accuracy = float(diagonal.sum() / total)
    expected = float(np.dot(support, predicted_count) / total**2)
    kappa = (accuracy - expected) / (1 - expected) if expected < 1 else np.nan
    numerator = diagonal.sum() * total - np.dot(support, predicted_count)
    denominator = np.sqrt(
        (total**2 - np.dot(predicted_count, predicted_count))
        * (total**2 - np.dot(support, support))
    )
    metrics = {
        "n_windows": total,
        "accuracy": accuracy,
        "balanced_accuracy": float(np.nanmean(recall)),
        "macro_f1": float(np.nanmean(f1)),
        "cohen_kappa": float(kappa),
        "multiclass_mcc": float(numerator / denominator)
        if denominator > 0
        else np.nan,
    }
    for index, label in enumerate(SCORABLE_SLEEP_LABELS):
        metrics[f"{label}_recall"] = float(recall[index])
        metrics[f"{label}_precision"] = float(precision[index])
        metrics[f"{label}_f1"] = float(f1[index])
    return matrix, metrics


def clean_prediction_chunks(table: pd.DataFrame, column: str) -> np.ndarray:
    """Apply the disclosed <12-s approximation within full acquisitions."""
    values = table[column].astype(str).to_numpy(copy=True)
    available = np.isin(values, SCORABLE_SLEEP_LABELS)
    positions = np.flatnonzero(available)
    if positions.size == 0:
        return values
    table_frames = table["frame"].to_numpy()
    table_segments = table["segment"].to_numpy()
    cuts = np.flatnonzero(
        (np.diff(positions) != 1)
        | (np.diff(table_frames[positions]) != 1)
        | (np.diff(table_segments[positions]) != 0)
    ) + 1
    min_samples = int(np.ceil(SHORT_RUN_SECONDS * dataio.FS_HZ))
    for chunk in np.split(positions, cuts):
        values[chunk] = physio.merge_short_state_runs(
            values[chunk],
            min_run_samples=min_samples,
        )
    return values


def blocked_sleep_predictions(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, physio.EMGThresholdFit, list[float]]:
    """Fit missing EMG thresholds on training blocks and predict held-out blocks."""
    table = table.copy()
    stable = table["stable_window"].to_numpy(dtype=bool)
    interior = table["fold_interior"].to_numpy(dtype=bool)
    labels = table["reference_label"].astype(str).to_numpy()
    scorable = np.isin(labels, SCORABLE_SLEEP_LABELS)
    emg = table["emg_rms_mv"].to_numpy(dtype=float)
    ratio = table["delta_theta_ratio"].to_numpy(dtype=float)
    folds = table["cv_fold"].to_numpy(dtype=int)

    full_fit = physio.fit_emg_threshold(emg[stable & scorable], labels[stable & scorable])
    full_raw = physio.classify_sleep_windows(
        emg,
        ratio,
        full_fit.threshold,
        DELTA_THETA_BOUNDARY,
    )
    table["calibrated_prediction_raw"] = full_raw
    calibrated_clean = full_raw.copy()
    for _segment, group in table.groupby("segment", sort=False):
        positions = group.index.to_numpy()
        calibrated_clean[positions] = physio.merge_short_state_runs(
            calibrated_clean[positions],
            min_run_samples=int(np.ceil(SHORT_RUN_SECONDS * dataio.FS_HZ)),
        )
    table["calibrated_prediction_clean"] = calibrated_clean

    oof = np.full(table.shape[0], "unscored", dtype="<U8")
    fold_thresholds: list[float] = []
    for fold in range(CV_FOLDS):
        training = stable & interior & scorable & (folds != fold)
        evaluation_test = stable & interior & scorable & (folds == fold)
        prediction_rows = folds == fold
        if not np.any(evaluation_test):
            continue
        try:
            fitted = physio.fit_emg_threshold(emg[training], labels[training])
        except ValueError:
            continue
        fold_thresholds.append(fitted.threshold)
        oof[prediction_rows] = physio.classify_sleep_windows(
            emg[prediction_rows],
            ratio[prediction_rows],
            fitted.threshold,
            DELTA_THETA_BOUNDARY,
        )
    table["oof_prediction_raw"] = oof
    table["oof_prediction_clean"] = clean_prediction_chunks(
        table,
        "oof_prediction_raw",
    )
    return table, full_fit, fold_thresholds


def binary_auc(values: np.ndarray, positive: np.ndarray) -> float:
    """Probability that a random positive window exceeds a random negative."""
    values = np.asarray(values, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    if not np.any(positive) or np.all(positive):
        return np.nan
    ranks = stats.rankdata(values)
    n_positive = int(np.count_nonzero(positive))
    n_negative = int(positive.size - n_positive)
    rank_sum = float(ranks[positive].sum())
    return (
        rank_sum - n_positive * (n_positive + 1) / 2
    ) / (n_positive * n_negative)


# %% [markdown]
# ## Step 3 — the requested verification figure
#
# The spectrogram is normalized independently at each frequency: zero means the
# session median for that frequency, and color is dB change from that median.
# This makes temporal shifts visible without changing the band powers used by
# the classifier.  The next three panels show exactly the requested 4-s
# relative-delta, normalized delta/theta, and EMG-RMS traces.  Here the paper's
# “delta/theta” is `(D / P[1–50]) / (T / D)`, as confirmed by the archived code
# for its cited method.  The simpler `D/T` ratio is retained in the output table
# as `raw_delta_theta_ratio`, but is not silently substituted for the published
# 0.3 feature.  The bottom comparison adds the deposited labels, the descriptive
# whole-session-calibrated rule, and the primary blocked out-of-fold prediction.

# %%
def plot_state_comparison(
    ax,
    timing: TimingMetadata,
    table: pd.DataFrame,
) -> None:
    """Draw deposited and reconstructed labels as frame-aligned categorical rows."""
    category_index = {label: index for index, label in enumerate(STATE_ORDER)}
    colors = [viz.DEFAULT_STATE_COLORS[label] for label in STATE_ORDER]
    rows: list[np.ndarray] = []
    row_labels: list[str] = []

    deposited = np.full(timing.n_frames, np.nan)
    for code, label in timing.codes.items():
        deposited[np.isclose(timing.state, code)] = category_index[label]
    rows.append(deposited)
    row_labels.append("deposited")

    if timing.data_info == "sleep":
        for column, label in (
            ("calibrated_prediction_clean", "calibrated + <12-s"),
            ("oof_prediction_clean", "blocked OOF + <12-s"),
        ):
            reconstructed = np.full(timing.n_frames, np.nan)
            for state_label in SCORABLE_SLEEP_LABELS:
                selected = table[column].astype(str).to_numpy() == state_label
                reconstructed[table.loc[selected, "frame"].to_numpy(dtype=int)] = (
                    category_index[state_label]
                )
            rows.append(reconstructed)
            row_labels.append(label)

    image_values = np.vstack(rows)
    masked = np.ma.masked_invalid(image_values)
    cmap = ListedColormap(colors)
    cmap.set_bad("white")
    ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=(0, timing.n_frames / timing.fs / 60, len(rows), 0),
        cmap=cmap,
        vmin=-0.5,
        vmax=len(STATE_ORDER) - 0.5,
    )
    ax.set_yticks(np.arange(len(rows)) + 0.5, row_labels)
    ax.set_xlabel("recorded time (min)")
    ax.set_ylabel("state")
    for spine in ax.spines.values():
        spine.set_visible(False)
    present = [label for label in STATE_ORDER if label in timing.codes.values()]
    ax.legend(
        handles=[
            Patch(
                facecolor=viz.DEFAULT_STATE_COLORS[label],
                label=viz.DEFAULT_STATE_LABELS[label],
            )
            for label in present
        ],
        loc="center left",
        bbox_to_anchor=(1.005, 0.5),
        frameon=False,
        fontsize=8,
    )


def make_verification_figure(
    timing: TimingMetadata,
    panels: tuple[physio.StateScoringFeaturePanel, ...],
    color_limit: float,
    table: pd.DataFrame,
    full_fit: physio.EMGThresholdFit | None,
    summary: dict[str, object],
):
    """Plot normalized EEG, classifier features, and label agreement."""
    timeline_view = {
        "n_frames": timing.n_frames,
        "fs": timing.fs,
        "duration_min": timing.n_frames / timing.fs / 60,
        "time_limits_min": (0.0, timing.n_frames / timing.fs / 60),
        "state": timing.state,
        "codes": timing.codes,
        "boundary_minutes": [
            (int(boundary) + 1) / timing.fs / 60
            for boundary in timing.boundary_ind
            if 0 <= int(boundary) < timing.n_frames - 1
        ],
    }
    fig = plt.figure(figsize=(14.5, 11), constrained_layout=True)
    grid = fig.add_gridspec(
        5,
        2,
        width_ratios=(1, 0.025),
        height_ratios=(2.5, 1, 1, 1, 0.8),
        hspace=0.06,
        wspace=0.04,
    )
    spectrum_ax = fig.add_subplot(grid[0, 0])
    delta_ax = fig.add_subplot(grid[1, 0], sharex=spectrum_ax)
    ratio_ax = fig.add_subplot(grid[2, 0], sharex=spectrum_ax)
    emg_ax = fig.add_subplot(grid[3, 0], sharex=spectrum_ax)
    state_ax = fig.add_subplot(grid[4, 0], sharex=spectrum_ax)
    colorbar_ax = fig.add_subplot(grid[0, 1])

    image = None
    for panel in panels:
        image = spectrum_ax.pcolormesh(
            panel.time_min,
            panel.frequency_hz,
            panel.frequency_normalized_power_db,
            shading="auto",
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
            rasterized=True,
        )
        delta_ax.plot(panel.time_min, panel.relative_delta, color="tab:purple", lw=0.55)
        ratio_ax.plot(panel.time_min, panel.delta_theta_ratio, color="tab:green", lw=0.5)
        emg_ax.plot(panel.time_min, panel.emg_rms, color="0.08", lw=0.5)
    if image is None:
        raise RuntimeError("No state-scoring feature panels were prepared")

    spectrum_ax.set_ylim(*DISPLAY_FREQUENCY_HZ)
    spectrum_ax.set_ylabel("EEG frequency (Hz)")
    spectrum_ax.tick_params(axis="x", labelbottom=False)
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("power change\n(dB vs frequency median)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    delta_ax.set_ylabel("relative delta\nP(1–4)/P(1–50)")
    delta_ax.set_ylim(0, min(1.0, table["relative_delta"].quantile(0.995) * 1.15))
    delta_ax.tick_params(axis="x", labelbottom=False)

    positive_ratio = table.loc[table["delta_theta_ratio"] > 0, "delta_theta_ratio"]
    ratio_limits = np.maximum(positive_ratio.quantile([0.005, 0.995]).to_numpy(), 1e-3)
    ratio_ax.set_yscale("log")
    ratio_ax.set_ylim(ratio_limits[0] / 1.25, ratio_limits[1] * 1.25)
    ratio_ax.axhline(
        DELTA_THETA_BOUNDARY,
        color="tab:red",
        ls="--",
        lw=1,
        label="paper boundary = 0.3",
    )
    ratio_ax.set_ylabel("normalized EEG delta/theta\n[D/P(1–50)] / [T/D]")
    ratio_ax.legend(loc="upper right", frameon=False, fontsize=8)
    ratio_ax.tick_params(axis="x", labelbottom=False)

    positive_emg = table.loc[table["emg_rms_mv"] > 0, "emg_rms_mv"]
    emg_limits = np.maximum(positive_emg.quantile([0.005, 0.995]).to_numpy(), 1e-8)
    emg_ax.set_yscale("log")
    emg_ax.set_ylim(emg_limits[0] / 1.5, emg_limits[1] * 1.5)
    if full_fit is not None:
        emg_ax.axhline(
            full_fit.threshold,
            color="tab:orange",
            ls="--",
            lw=1,
            label=f"descriptive calibrated threshold = {full_fit.threshold:.4g} mV",
        )
        emg_ax.legend(loc="upper right", frameon=False, fontsize=8)
    emg_ax.set_ylabel("4-s EMG RMS (mV)")
    emg_ax.tick_params(axis="x", labelbottom=False)

    for axis in (spectrum_ax, delta_ax, ratio_ax, emg_ax):
        viz.mark_acquisition_boundaries(axis, timeline_view)
        axis.set_xlim(*timeline_view["time_limits_min"])
    plot_state_comparison(state_ax, timing, table)
    viz.mark_acquisition_boundaries(state_ax, timeline_view)

    if timing.data_info == "sleep":
        subtitle = (
            "blocked OOF balanced accuracy raw/clean = "
            f"{summary['oof_raw_balanced_accuracy']:.3f}/"
            f"{summary['oof_clean_balanced_accuracy']:.3f}; "
            "quiet awake excluded from fitting"
        )
    else:
        subtitle = (
            "0.6% isoflurane protocol label; physiology shown descriptively—"
            "no published anesthesia classifier"
        )
    fig.suptitle(
        f"{timing.name} · paper-motivated 4-s EEG/EMG state verification\n{subtitle}",
        fontsize=14,
    )
    return fig


# %% [markdown]
# ## Step 4 — reproduce the six sleep-label sequences
#
# Two predictions are retained:
#
# - `calibrated_prediction_*`: one threshold fitted to the complete stable
#   W/NREM/REM sequence.  This is useful for a readable descriptive timeline but
#   its agreement is resubstitution, not held-out performance.
# - `oof_prediction_*`: each 60-s block predicted with a threshold fitted on the
#   other folds.  These blocked out-of-fold metrics are the primary check.
#
# The 12-s cleanup uses a declared deterministic approximation when unlike
# states surround a short run: merge into the longer neighbor, with ties going
# to the preceding state.  The paper did not state this tie rule, so raw and
# cleaned results are both saved.

# %%
window_tables: list[pd.DataFrame] = []
sleep_summary_rows: list[dict[str, object]] = []
confusion_rows: list[dict[str, object]] = []

for recording_name in SLEEP_RECORDINGS:
    print(f"\n=== {recording_name}: sleep-label reproduction ===")
    timing = load_timing_metadata(recording_name)
    physiology = physio.load_physiology(recording_name)
    alignment = physio.align_frame_triggers(
        physiology,
        timing.n_frames,
        timing.boundary_ind,
        timing.fs,
    )
    panels, color_limit = physio.prepare_state_scoring_features(
        physiology,
        alignment,
        imaging_fs=timing.fs,
        analysis_fs=ANALYSIS_FS,
        window_seconds=WINDOW_SECONDS,
        display_frequency_hz=DISPLAY_FREQUENCY_HZ,
    )
    table = feature_table(timing, alignment, panels)
    table, full_fit, fold_thresholds = blocked_sleep_predictions(table)

    labels = table["reference_label"].astype(str).to_numpy()
    oof_available = np.isin(
        table["oof_prediction_raw"].astype(str).to_numpy(),
        SCORABLE_SLEEP_LABELS,
    )
    if not fold_thresholds:
        raise RuntimeError(f"{recording_name}: no temporal CV fold could be fitted")
    summary: dict[str, object] = {
        "recording": recording_name,
        "animal": recording_name.split("_")[0],
        "n_feature_windows": table.shape[0],
        "n_stable_windows": int(table["stable_window"].sum()),
        "emg_threshold_full_mv": full_fit.threshold,
        "emg_gate_calibration_balanced_accuracy": full_fit.balanced_accuracy,
        "cv_threshold_min_mv": float(np.min(fold_thresholds)),
        "cv_threshold_median_mv": float(np.median(fold_thresholds)),
        "cv_threshold_max_mv": float(np.max(fold_thresholds)),
        "oof_coverage_of_all_feature_windows": float(np.mean(oof_available)),
    }
    for mode, column in (
        ("calibrated_raw", "calibrated_prediction_raw"),
        ("calibrated_clean", "calibrated_prediction_clean"),
        ("oof_raw", "oof_prediction_raw"),
        ("oof_clean", "oof_prediction_clean"),
    ):
        predicted = table[column].astype(str).to_numpy()
        eligible = (
            table["stable_window"].to_numpy(dtype=bool)
            & np.isin(labels, SCORABLE_SLEEP_LABELS)
            & np.isin(predicted, SCORABLE_SLEEP_LABELS)
        )
        if mode.startswith("oof"):
            eligible &= table["fold_interior"].to_numpy(dtype=bool)
        matrix, metrics = confusion_and_metrics(labels[eligible], predicted[eligible])
        summary.update({f"{mode}_{key}": value for key, value in metrics.items()})
        for row, actual in enumerate(SCORABLE_SLEEP_LABELS):
            for column_index, estimate in enumerate(SCORABLE_SLEEP_LABELS):
                confusion_rows.append(
                    {
                        "recording": recording_name,
                        "mode": mode,
                        "reference_label": actual,
                        "predicted_label": estimate,
                        "count": int(matrix[row, column_index]),
                    }
                )

    quiet = (labels == "quiet_awake") & table["stable_window"].to_numpy(dtype=bool)
    summary["quiet_awake_windows"] = int(np.count_nonzero(quiet))
    for predicted_label in SCORABLE_SLEEP_LABELS:
        summary[f"quiet_awake_predicted_{predicted_label}_fraction"] = (
            float(
                np.mean(
                    table.loc[quiet, "calibrated_prediction_raw"].astype(str)
                    == predicted_label
                )
            )
            if np.any(quiet)
            else np.nan
        )
    for state_label in ("awake", "quiet_awake", "nrem", "rem"):
        selected = (labels == state_label) & table["stable_window"].to_numpy(dtype=bool)
        for feature in (
            "relative_delta",
            "delta_theta_ratio",
            "raw_delta_theta_ratio",
            "emg_rms_mv",
        ):
            summary[f"median_{feature}_{state_label}"] = (
                float(table.loc[selected, feature].median())
                if np.any(selected)
                else np.nan
            )

    print(
        f"full EMG threshold = {full_fit.threshold:.5g} mV; "
        f"blocked OOF accuracy = {summary['oof_raw_accuracy']:.3f}; "
        f"balanced accuracy = {summary['oof_raw_balanced_accuracy']:.3f}; "
        f"macro-F1 = {summary['oof_raw_macro_f1']:.3f}"
    )
    figure = make_verification_figure(
        timing,
        panels,
        color_limit,
        table,
        full_fit,
        summary,
    )
    figure_path = FIG_DIR / f"09_state_verification_{recording_name}.png"
    figure.savefig(figure_path, dpi=160, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(figure)
    print("saved ->", figure_path)

    window_tables.append(table)
    sleep_summary_rows.append(summary)
    del timing, physiology, alignment, panels, table, figure
    gc.collect()


# %% [markdown]
# ## Step 5 — anesthesia is a protocol condition, not the sleep classifier
#
# In the final paper, mice received 0.6% isoflurane after 20–60 min of awake
# recording.  The paper gives no EEG/EMG boundary for declaring anesthesia.
# For network analysis, it excluded the first 1,000 imaging frames after the
# transition as potentially unstable.  Here we verify that deposited
# `used_frame[anesthesia]` starts exactly 1,000 frames after the deposited state
# transition, then report the EEG/EMG features and their descriptive AUCs.
# An AUC near 0.5 means that feature alone does not separate the protocol
# conditions; distance from 0.5 is descriptive and is not classifier accuracy.
# These summaries cover all stable 4-s windows in each protocol label, including
# the first 1,000 anesthesia frames that the network analysis later excluded;
# they describe induction plus maintenance, not only the paper's analysis subset.

# %%
anesthesia_summary_rows: list[dict[str, object]] = []

for recording_name in ANESTHESIA_RECORDINGS:
    print(f"\n=== {recording_name}: descriptive anesthesia check ===")
    timing = load_timing_metadata(recording_name)
    physiology = physio.load_physiology(recording_name)
    alignment = physio.align_frame_triggers(
        physiology,
        timing.n_frames,
        timing.boundary_ind,
        timing.fs,
    )
    panels, color_limit = physio.prepare_state_scoring_features(
        physiology,
        alignment,
        imaging_fs=timing.fs,
        analysis_fs=ANALYSIS_FS,
        window_seconds=WINDOW_SECONDS,
        display_frequency_hz=DISPLAY_FREQUENCY_HZ,
    )
    table = feature_table(timing, alignment, panels)
    table["calibrated_prediction_raw"] = "not_applicable"
    table["calibrated_prediction_clean"] = "not_applicable"
    table["oof_prediction_raw"] = "not_applicable"
    table["oof_prediction_clean"] = "not_applicable"

    transition_frames = np.flatnonzero(np.diff(timing.state) != 0) + 1
    if transition_frames.size != 1:
        raise ValueError(
            f"{recording_name}: expected one awake-to-anesthesia transition, "
            f"found {transition_frames.tolist()}"
        )
    transition_frame = int(transition_frames[0])
    first_used_anesthesia = (
        int(timing.used_frame[1][0]) if len(timing.used_frame) > 1 else -1
    )
    stable = table["stable_window"].to_numpy(dtype=bool)
    labels = table["reference_label"].astype(str).to_numpy()
    anesthesia = labels == "anesthesia"
    summary = {
        "recording": recording_name,
        "animal": recording_name.split("_")[0],
        "transition_frame": transition_frame,
        "transition_time_min": transition_frame / timing.fs / 60,
        "first_used_anesthesia_frame": first_used_anesthesia,
        "first_used_minus_transition_frames": first_used_anesthesia
        - transition_frame,
        "matches_paper_1000_frame_stabilization_exclusion": first_used_anesthesia
        - transition_frame
        == 1000,
        "physiology_summary_scope": "whole_protocol_stable_windows",
    }
    for feature in (
        "relative_delta",
        "delta_theta_ratio",
        "raw_delta_theta_ratio",
        "emg_rms_mv",
    ):
        values = table[feature].to_numpy(dtype=float)
        awake_selected = stable & (labels == "awake")
        anesthesia_selected = stable & anesthesia
        summary[f"median_{feature}_awake"] = float(np.median(values[awake_selected]))
        summary[f"median_{feature}_anesthesia"] = float(
            np.median(values[anesthesia_selected])
        )
        summary[f"auc_anesthesia_greater_{feature}"] = binary_auc(
            values[stable],
            anesthesia[stable],
        )

    print(
        f"transition frame {transition_frame:,}; first analysis anesthesia frame "
        f"{first_used_anesthesia:,} (+{first_used_anesthesia-transition_frame:,}); "
        f"EMG AUC = {summary['auc_anesthesia_greater_emg_rms_mv']:.3f}"
    )
    figure = make_verification_figure(
        timing,
        panels,
        color_limit,
        table,
        full_fit=None,
        summary=summary,
    )
    figure_path = FIG_DIR / f"09_state_verification_{recording_name}.png"
    figure.savefig(figure_path, dpi=160, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(figure)
    print("saved ->", figure_path)

    window_tables.append(table)
    anesthesia_summary_rows.append(summary)
    del timing, physiology, alignment, panels, table, figure
    gc.collect()


# %% [markdown]
# ## Step 6 — save auditable outputs
#
# `09_state_scoring_windows.csv.gz` contains every retained 4-s window and lets
# a reader recalculate each metric.  Sleep summaries distinguish descriptive
# whole-session calibration from primary blocked out-of-fold agreement.
# Mouse04 contributes two sessions but only one animal; any later across-animal
# inference must first average those two days so that the paper's sleep sample
# size remains five animals rather than six sessions.

# %%
sleep_summary = pd.DataFrame(sleep_summary_rows)
anesthesia_summary = pd.DataFrame(anesthesia_summary_rows)
confusion_table = pd.DataFrame(confusion_rows)
all_windows = pd.concat(window_tables, ignore_index=True)

sleep_summary_path = RESULTS_DIR / "09_sleep_label_reproduction_summary.csv"
anesthesia_summary_path = RESULTS_DIR / "09_anesthesia_physiology_summary.csv"
confusion_path = RESULTS_DIR / "09_sleep_confusion_matrices.csv"
windows_path = RESULTS_DIR / "09_state_scoring_windows.csv.gz"

sleep_summary.to_csv(sleep_summary_path, index=False, float_format="%.6g")
anesthesia_summary.to_csv(anesthesia_summary_path, index=False, float_format="%.6g")
confusion_table.to_csv(confusion_path, index=False)
all_windows.to_csv(windows_path, index=False, compression="gzip", float_format="%.7g")

print("\nBlocked out-of-fold sleep-label agreement:")
print(
    sleep_summary[
        [
            "recording",
            "oof_raw_accuracy",
            "oof_raw_balanced_accuracy",
            "oof_raw_macro_f1",
            "oof_clean_balanced_accuracy",
        ]
    ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
)
print("\nAnesthesia protocol/stability check:")
print(
    anesthesia_summary[
        [
            "recording",
            "first_used_minus_transition_frames",
            "matches_paper_1000_frame_stabilization_exclusion",
            "auc_anesthesia_greater_relative_delta",
            "auc_anesthesia_greater_delta_theta_ratio",
            "auc_anesthesia_greater_emg_rms_mv",
        ]
    ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
)
print("\nsaved ->", sleep_summary_path)
print("saved ->", anesthesia_summary_path)
print("saved ->", confusion_path)
print("saved ->", windows_path)

# %% [markdown]
# ## Interpretation boundary
#
# High agreement would show that the published numerical rule is broadly
# consistent with the deposited annotations.  It would not independently
# validate those annotations: both reference labels and reconstructed features
# originate from the same EEG/EMG recordings, and the missing EMG threshold is
# estimated from deposited labels.  Conversely, disagreements can reflect the
# unreported manual threshold, quiet-awake correction, spectral implementation,
# window alignment, cleanup convention, or genuinely ambiguous physiology.
#
# The frame-trigger manifest used here was fixed before this verification and is
# never re-optimized for label agreement.  Its original audit did use
# cross-modal/state correspondence for ambiguous release segments, so temporal
# alignment is also not wholly independent of the annotations.
