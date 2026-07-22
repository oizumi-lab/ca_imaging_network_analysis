# %% [markdown]
# # 02 · Visualizing population activity across brain states
#
# Before estimating a functional network, first look at the activity itself.
# This tutorial makes four complementary views:
#
# 1. **Raw ΔF/F activity**, either as stacked traces from a reproducible random
#    sample of 100 neurons or as a display-aggregated heatmap to which every
#    neuron and frame contributes.
# 2. **An all-neuron deconvolved-spike raster** made from `spike_deconv > 0`.
#    Every neuron is retained; only the time axis is binned to about one second
#    for display. A population-active-fraction trace makes changes in overall
#    activity easier to see when thousands of rows are compressed onto the page.
# 3. **A brain-region-grouped spike raster** containing the same neurons and
#    events as the activity-ranked raster. Brain regions follow the atlas
#    labels supplied with the recording, and neurons remain activity-ranked
#    within each region so the two orderings can be compared directly.
# 4. **An active-neuron Rastermap view**. This tutorial passes only neurons above
#    an explicit activity criterion to the official MouseLand implementation,
#    then shows the one-row-per-neuron raster and adjacent-neuron superneurons.
# Sleep and anesthesia come from separate recording sessions, so they are shown
# as separate timelines. The colored strip under every panel is aligned to the
# original frame-by-frame state vector. In the sleep recording it includes
# awake, quiet awake, NREM sleep, and REM sleep rather than joining disjoint
# state-specific analysis epochs into an artificial continuous time series.
# Dashed vertical lines mark microscope acquisition breaks; time across such a
# line is cumulative recorded time, not an uninterrupted acquisition.
# By default, all figures show the **entire recorded sequence**. The settings
# below also let you zoom to a chosen time interval without changing the data.

# %%
import gc
import os
import sys
from contextlib import ExitStack

# add the repo root (parent of scripts/) to the path so `src.funcnet` is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from src.funcnet import (
    dataio,
    rastermap_tools as rmt,
    timeseries as ts,
    visualization as viz,
)
from src.funcnet.paths import FIG_DIR, RESULTS_DIR

FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Settings
#
# These are the settings you are expected to edit:
#
# - `RECORDING_MODE`: use `"all"` to plot every available recording session
#   (six sleep sessions and four anesthesia sessions). Each session becomes one
#   page in a condition-specific PDF. Use `"representative"` for the faster
#   two-session preview used earlier in the tutorial.
# - `SLEEP_RECORDINGS`, `ANESTHESIA_RECORDINGS`, and
#   `REPRESENTATIVE_RECORDINGS`: define which files belong to those two modes.
#   Mouse 4 has two sleep days, so the six sleep recordings correspond to five
#   sleep animals rather than six independent animals.
# - `VIEW_WINDOWS_MIN`: `None` shows every frame (the default). To show a window,
#   replace a recording's value with `(start_min, stop_min)`, for example
#   `(5, 12)`. Each interval uses that recording's original time axis; a stop
#   beyond the recording is clipped to its final frame. Unlisted recordings also
#   default to the full sequence. This changes the visible x-range, while keeping
#   full-session sampling, ordering, and color scales directly comparable.
# - `DFF_VIEW`: independently controls the neurons shown *within each session*.
#   Use `"sample"` for individually readable stacked traces or `"all"` for an
#   all-neuron ΔF/F heatmap. This is separate from `RECORDING_MODE`, which
#   controls whether all recording sessions are plotted.
# - `N_TRACE_NEURONS` and `RANDOM_SEED`: control the reproducible random sample
#   and are used only when `DFF_VIEW == "sample"`.
#   They change only the stacked-trace panel; the raster still includes every
#   neuron in the recording.
# - `DISPLAY_BIN_SECONDS`: shared temporal display resolution of the all-neuron
#   ΔF/F heatmap and spike raster. `POPULATION_SMOOTH_SECONDS` controls only the
#   summary line.
# - `RUN_RASTERMAP`: fit and display the official Rastermap ordering. Unlike the
#   two baseline rasters, this view deliberately uses only active neurons.
# - `RASTERMAP_SELECTION_MODE`: the default `"dataset_active"` uses every
#   finite/nonconstant row in the supplied `nonzero_ROI` mask. That mask is the
#   dataset's documented activity filter from the publication's complete
#   network-analysis windows. The optional
#   `"dataset_active_and_minimum_positive_bin_rate"` adds a fixed support floor;
#   `"minimum_positive_bin_rate"` omits the dataset mask; and
#   `"minimum_positive_bins"` uses a duration-dependent numerical-support rule.
# - `RASTERMAP_MIN_POSITIVE_BIN_RATE_PER_SECOND`: minimum number of positive
#   OASIS deconvolution samples per primary-state second. It is a numerical
#   support proxy, not a calibrated physiological firing rate. The paper reports
#   application cutoffs of 0.1--0.25 Hz, but its public Figure 3 input does not
#   reveal the exact upstream selection. Tutorials 03--05 audit why a literal
#   0.10-positive-bin/s conversion is unsuitable across these recordings.
# - The remaining `RASTERMAP_*` values control the official model. The lag is
#   expressed in seconds and converted to frames for each session. Cache files
#   are reused only when all data signatures, settings, and selection metadata
#   match.
# - `SHOW_FIGURES`: when `True`, display every ΔF/F, activity-ranked raster,
#   brain-region-grouped raster, and active-neuron Rastermap figure in the
#   interactive window. Figures are shown and released one session at a time so
#   displaying all sessions does not keep all plot objects in memory.
#
# Both baseline rasters and the `"all"` ΔF/F view retain every recorded neuron
# and never apply `nonzero_ROI` or another neuron filter. Rastermap is the
# intentional exception: it uses the explicit active-neuron population above.
# The default retains 7,693 of 7,843 rows in `mouse01_sleep` and 3,013 of 4,337
# in `mouse07_ane`. Tutorial 06 holds this dataset-active population fixed while
# repeating time-block allocations and fit seeds, so "active-only" should not
# be read as a calibrated physiological firing-rate threshold.
# Full-session mode requires all files from `download_data.py`, not only
# `download_data.py --example`. It streams one recording at a time so the ten
# large data files are never held in memory simultaneously.

# %%
RECORDING_MODE = (
    "representative"  # "all" = all 10 sessions; "representative" = quick preview
)

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
REPRESENTATIVE_RECORDINGS = ("mouse01_sleep", "mouse07_ane")

VIEW_WINDOWS_MIN = {
    "mouse01_sleep": None,  # None = this session's entire sequence
    "mouse02_sleep": None,
    "mouse03_sleep": None,
    "mouse04_day1_sleep": None,
    "mouse04_day2_sleep": None,
    "mouse05_sleep": None,
    "mouse03_ane": None,
    "mouse05_ane": None,
    "mouse06_ane": None,
    "mouse07_ane": None,
}
DFF_VIEW = "sample"  # "sample" = line traces; "all" = all-neuron heatmap
N_TRACE_NEURONS = 100
RANDOM_SEED = 7
DISPLAY_BIN_SECONDS = 1.0
POPULATION_SMOOTH_SECONDS = 5.0
SHOW_FIGURES = True

RUN_RASTERMAP = True
RASTERMAP_SELECTION_MODE = "dataset_active"
RASTERMAP_MIN_POSITIVE_BIN_RATE_PER_SECOND = 0.020
RASTERMAP_MIN_POSITIVE_BINS = 50
RASTERMAP_N_CLUSTERS = 100
RASTERMAP_N_PCS = 128
RASTERMAP_LOCALITY = 0.0
# Five frames in the paper's 3.2-Hz recording span 1.5625 seconds.
RASTERMAP_TIME_LAG_SECONDS = 5 / 3.2
RASTERMAP_TIME_BIN = 1
RASTERMAP_MEAN_TIME = True
RASTERMAP_SUPERNEURON_SIZE = 50
RASTERMAP_RANDOM_SEED = 0
RASTERMAP_DISPLAY_VMIN = 0.0
RASTERMAP_DISPLAY_VMAX = 1.5
RASTERMAP_CACHE_DIR = RESULTS_DIR / "cache" / "rastermap"


# %% [markdown]
# ## Helpers for an exact categorical state timeline
#
# The numeric code `1` means NREM in a sleep file but anesthesia in an
# anesthesia file. We therefore choose the code-to-label mapping from
# `rec.data_info`, then explicitly map those codes to categorical colors. A
# continuous colormap would incorrectly imply that the states are ordered
# measurements.
#
# Reusable timeline helpers live in ``src/funcnet/visualization.py``. That module
# validates time limits, draws categorical state bouts, and marks microscope
# breaks. ``dataio.state_codes`` supplies the recording-specific code legend.
# The tutorial keeps only loading and file-output orchestration here:
#
# - `load_view` loads one large session, extracts compact plotting arrays, and
#   immediately releases the original matrices.
# - the explicit Step 6 cell shows every compact view interactively and streams
#   it into multipage PDFs, one full recording per page.


# %% [markdown]
# ## Helpers for explicit active-neuron Rastermap fitting
#
# These helpers keep selection and model fitting separate so the population can
# be audited before interpreting its order:
#
# - `rastermap_selection` returns the Boolean active-neuron mask, a short figure
#   label, and complete cache provenance. It never samples neurons randomly.
# - `primary_state_positive_bin_rates` measures support over awake + NREM for
#   sleep or awake + anesthesia for anesthesia without joining those epochs into
#   a new activity matrix.
# - `fit_or_load_rastermap` converts the mask to original ROI row numbers, checks
#   an exact cache match, and otherwise calls the official MouseLand package.
#   The returned order is mapped back to original ROI numbers by
#   `rastermap_tools.fit_selected_neurons`.
#
# The lagged similarity calculation treats adjacent matrix columns as adjacent
# time points. Rastermap 1.0 has no segment mask for microscope breaks, so a few
# lag pairs can cross a break. Tutorial 04 quantifies this fraction and performs
# the stronger held-out checks; the figure here remains a descriptive view.


# %%
def primary_state_positive_bin_rates(rec, chunk_frames=2048):
    """Return positive-bin rates over the two primary states and their labels."""
    code_legend = dataio.state_codes(rec)
    primary_labels = (
        {"awake", "nrem"} if rec.data_info == "sleep" else {"awake", "anesthesia"}
    )
    primary_codes = tuple(
        code for code, label in code_legend.items() if label in primary_labels
    )
    selected_frames = np.isin(rec.state, primary_codes)
    n_selected_frames = int(np.count_nonzero(selected_frames))
    if n_selected_frames == 0:
        raise ValueError(f"{rec.name}: no primary-state frames were found")

    positive_counts = np.zeros(rec.n_neurons, dtype=np.int64)
    for start in range(0, rec.n_frames, chunk_frames):
        stop = min(rec.n_frames, start + chunk_frames)
        local_selection = selected_frames[start:stop]
        if np.any(local_selection):
            positive_counts += np.count_nonzero(
                rec.spike_deconv[:, start:stop][:, local_selection] > 0,
                axis=1,
            )
    rates = positive_counts.astype(np.float64) * rec.fs / n_selected_frames
    return rates, primary_codes, n_selected_frames


def rastermap_selection(rec):
    """Return an explicit active-neuron mask, display label, and provenance."""
    if RASTERMAP_SELECTION_MODE == "dataset_active":
        if rec.nonzero_ROI is None:
            raise ValueError(
                f"{rec.name}: the selected Rastermap mode requires nonzero_ROI"
            )
        mask = rmt.valid_activity_rows(rec.spike_deconv) & rec.nonzero_ROI
        label = "dataset-active (nonzero_ROI)"
        metadata = {
            "definition": "dataset_nonzero_roi_and_finite_nonconstant",
            "dataset_nonzero_roi_neurons": int(np.count_nonzero(rec.nonzero_ROI)),
        }
    elif RASTERMAP_SELECTION_MODE in {
        "dataset_active_and_minimum_positive_bin_rate",
        "minimum_positive_bin_rate",
    }:
        rates, primary_codes, primary_frame_count = primary_state_positive_bin_rates(
            rec
        )
        mask = rmt.valid_activity_rows(rec.spike_deconv) & (
            rates >= RASTERMAP_MIN_POSITIVE_BIN_RATE_PER_SECOND
        )
        uses_dataset_mask = (
            RASTERMAP_SELECTION_MODE == "dataset_active_and_minimum_positive_bin_rate"
        )
        if uses_dataset_mask:
            if rec.nonzero_ROI is None:
                raise ValueError(
                    f"{rec.name}: the selected Rastermap mode requires nonzero_ROI"
                )
            mask &= rec.nonzero_ROI
        label = ("dataset-active " if uses_dataset_mask else "active ") + (
            f"(≥{RASTERMAP_MIN_POSITIVE_BIN_RATE_PER_SECOND:g} positive bins/s "
            "over primary states)"
        )
        metadata = {
            "definition": (
                "dataset_nonzero_roi_and_finite_nonconstant_and_minimum_"
                "primary_state_positive_bin_rate_per_second"
                if uses_dataset_mask
                else "finite_nonconstant_and_minimum_primary_state_positive_"
                "bin_rate_per_second"
            ),
            "minimum_positive_bin_rate_per_second": (
                RASTERMAP_MIN_POSITIVE_BIN_RATE_PER_SECOND
            ),
            "primary_state_codes": list(primary_codes),
            "primary_state_frames": primary_frame_count,
        }
    elif RASTERMAP_SELECTION_MODE == "minimum_positive_bins":
        mask = rmt.active_deconvolution_count_rows(
            rec.spike_deconv,
            min_positive_bins=RASTERMAP_MIN_POSITIVE_BINS,
        )
        effective_rate = RASTERMAP_MIN_POSITIVE_BINS * rec.fs / rec.n_frames
        label = f"active (≥{RASTERMAP_MIN_POSITIVE_BINS} positive bins)"
        metadata = {
            "definition": "finite_nonconstant_and_minimum_positive_bins",
            "minimum_positive_bins": RASTERMAP_MIN_POSITIVE_BINS,
            "effective_positive_bin_rate_per_second": effective_rate,
        }
    else:
        raise ValueError(
            "RASTERMAP_SELECTION_MODE must be "
            "'dataset_active', 'dataset_active_and_minimum_positive_bin_rate', "
            "'minimum_positive_bin_rate', or 'minimum_positive_bins'"
        )

    metadata["selection_label"] = label
    metadata["selected_neurons"] = int(np.count_nonzero(mask))
    return mask, label, metadata


def fit_or_load_rastermap(rec):
    """Fit official Rastermap to all selected active rows or reuse an exact cache."""
    active_mask, selection_label, selection_metadata = rastermap_selection(rec)
    selected_rows = np.flatnonzero(active_mask)
    if selected_rows.size < 2:
        raise ValueError(
            f"{rec.name}: the Rastermap activity criterion retained fewer than "
            "two usable neurons"
        )

    lag_frames = max(
        0,
        round(RASTERMAP_TIME_LAG_SECONDS * rec.fs / RASTERMAP_TIME_BIN),
    )
    fit_parameters = {
        "n_clusters": RASTERMAP_N_CLUSTERS,
        "n_PCs": RASTERMAP_N_PCS,
        "locality": RASTERMAP_LOCALITY,
        "time_lag_window": lag_frames,
        "mean_time": RASTERMAP_MEAN_TIME,
        "time_bin": RASTERMAP_TIME_BIN,
        "superneuron_size": RASTERMAP_SUPERNEURON_SIZE,
        "random_state": RASTERMAP_RANDOM_SEED,
    }
    display_parameters = {
        **fit_parameters,
        "selection_mode": RASTERMAP_SELECTION_MODE,
        "selection_label": selection_label,
    }
    source_path = dataio.RAW_DIR / f"{rec.name}.mat"
    metadata = rmt.make_cache_metadata(
        recording_name=rec.name,
        n_neurons=rec.n_neurons,
        n_frames=rec.n_frames,
        fs=rec.fs,
        parameters=fit_parameters,
        neuron_selection=selection_metadata,
        source_path=source_path,
    )
    cache_path = RASTERMAP_CACHE_DIR / f"{rec.name}.npz"
    result = rmt.load_cache(cache_path, metadata)
    cached = result is not None
    print(
        f"  Rastermap population: {selected_rows.size:,}/{rec.n_neurons:,} "
        f"{selection_label} ...",
        flush=True,
    )
    if result is None:
        result = rmt.fit_selected_neurons(
            rec.spike_deconv,
            selected_rows,
            **fit_parameters,
        )
        rmt.save_cache(cache_path, result, metadata)
        print(
            f"  fitted Rastermap in {result.runtime_seconds:.2f} s; "
            f"cached -> {cache_path}",
            flush=True,
        )
    else:
        print(
            f"  reused exact Rastermap cache ({result.runtime_seconds:.2f} s "
            "original fit)",
            flush=True,
        )
    return result, cached, display_parameters


# %% [markdown]
# ## Build lightweight plotting data
#
# The loader intentionally returns all three full `N × T` activity matrices, so
# one recording can occupy several gigabytes. We load sessions one at a time
# and retain only sampled traces or a binned ΔF/F heatmap, plus a compact spike
# raster, population active fraction, and two lightweight neuron-order arrays.
# In all-session mode, even this compact view is released after its PDF pages
# are written.
#
# `spike_deconv` is an OASIS deconvolved spike estimate, not a direct electrical
# measurement of action potentials. A black raster mark means that a neuron had
# at least one **positive deconvolution sample** in that approximately one-second
# display bin. We do not collapse consecutive positive samples into custom event
# onsets. The population trace is the percentage of neurons positive in each
# original 7.65 Hz frame, smoothed over five seconds.
#
# The reusable preparation helpers have distinct jobs and live in
# ``visualization.py`` or ``timeseries.py``:
#
# - `viz.select_trace_neurons` chooses the reproducible sample used by the line view.
# - `ts.acquisition_segments` prevents plots or bins from crossing microscope
#   discontinuities; optional extra splits also keep state bins separate.
# - `viz.binned_spike_raster` builds the compact all-neuron spike display.
# - `viz.binned_dff_heatmaps` builds the optional all-neuron ΔF/F display. It keeps
#   one row per neuron and only averages neighboring time frames for display.
# - `ts.segmented_moving_average` smooths the population summary within, never
#   across, acquisition segments.
# - `viz.brain_region_order` groups the existing activity-ranked ROI order by
#   exact atlas region while retaining activity rank within each group.
# - `prepare_view` runs the relevant helpers while the large recording is loaded
#   and returns a smaller dictionary used by the plotting functions.


# %%
def prepare_view(rec, seed) -> viz.ActivityView:
    """Extract only the arrays needed by the tutorial figures."""
    if DFF_VIEW not in {"sample", "all"}:
        raise ValueError("DFF_VIEW must be 'sample' or 'all'")
    duration_min = rec.n_frames / rec.fs / 60
    time_limits_min = viz.resolve_time_limits(
        duration_min,
        VIEW_WINDOWS_MIN.get(rec.name),
    )

    print(
        f"  building a deconvolved-spike raster for all {rec.n_neurons:,} neurons ...",
        flush=True,
    )
    raster, active_counts, bin_frames, neuron_order, bin_centers = (
        viz.binned_spike_raster(
            rec.spike_deconv,
            rec.fs,
            DISPLAY_BIN_SECONDS,
            rec.boundary_ind,
            rec.state,
        )
    )

    if DFF_VIEW == "sample":
        trace_ids = viz.select_trace_neurons(rec.n_neurons, N_TRACE_NEURONS, seed)
        print(
            f"  copying raw ΔF/F for {trace_ids.size} selected neurons ...", flush=True
        )
        dff = rec.dFF[trace_ids].copy()
        dff_heatmaps = None
        dff_color_limit = None
        dff_bin_frames = None
    else:
        trace_ids = None
        dff = None
        print(f"  binning raw ΔF/F for all {rec.n_neurons:,} neurons ...", flush=True)
        dff_heatmaps, dff_color_limit, dff_bin_frames = viz.binned_dff_heatmaps(
            rec.dFF,
            neuron_order,
            rec.fs,
            DISPLAY_BIN_SECONDS,
            rec.boundary_ind,
            rec.state,
        )

    active_fraction = 100 * active_counts.astype(float) / rec.n_neurons
    active_fraction = ts.segmented_moving_average(
        active_fraction,
        round(POPULATION_SMOOTH_SECONDS * rec.fs),
        rec.boundary_ind,
    )
    boundary_minutes = [
        (int(boundary) + 1) / rec.fs / 60
        for boundary in np.asarray(rec.boundary_ind).ravel()
        if 0 <= int(boundary) < rec.n_frames - 1
    ]
    atlas_labels = (
        rec.atlas
        if rec.atlas is not None
        else np.full(rec.n_neurons, None, dtype=object)
    )
    brain_regions = viz.brain_region_labels(atlas_labels)
    region_order = viz.brain_region_order(brain_regions, neuron_order)

    view: viz.ActivityView = {
        "name": rec.name,
        "data_info": rec.data_info,
        "n_neurons": rec.n_neurons,
        "n_frames": rec.n_frames,
        "fs": rec.fs,
        "duration_min": duration_min,
        "time_limits_min": time_limits_min,
        "state": rec.state.astype(np.float32, copy=True),
        "codes": dict(dataio.state_codes(rec)),
        "trace_ids": trace_ids,
        "dff": dff,
        "dff_view": DFF_VIEW,
        "dff_heatmaps": dff_heatmaps,
        "dff_color_limit": dff_color_limit,
        "dff_bin_frames": dff_bin_frames,
        "raster": raster,
        "population_active_fraction": active_fraction,
        "bin_frames": bin_frames,
        "bin_centers_min": bin_centers / rec.fs / 60,
        "neuron_order": neuron_order,
        "brain_regions": brain_regions,
        "brain_region_order": region_order,
        "boundary_minutes": boundary_minutes,
        "acquisition_segments": ts.acquisition_segments(
            rec.n_frames,
            rec.boundary_ind,
        ),
    }
    if RUN_RASTERMAP:
        rastermap_result, cached, parameters = fit_or_load_rastermap(rec)
        view.update(
            {
                "rastermap_X_embedding": rastermap_result.X_embedding,
                "rastermap_embedding": rastermap_result.embedding,
                "rastermap_isort": rastermap_result.isort,
                "rastermap_valid_rows": rastermap_result.valid_rows,
                "rastermap_runtime_seconds": rastermap_result.runtime_seconds,
                "rastermap_cached": cached,
                "rastermap_display_selected_only": True,
                "rastermap_stop_min": (
                    rastermap_result.X_embedding.shape[1]
                    * RASTERMAP_TIME_BIN
                    / rec.fs
                    / 60
                ),
                "rastermap_parameters": parameters,
                "rastermap_version": rmt.installed_rastermap_version(),
            }
        )
    return view


# %% [markdown]
# ## Figure 1 — ΔF/F as sampled traces or an all-neuron heatmap
#
# With `DFF_VIEW = "sample"`, each selected trace is median-centered and moved
# vertically by the **same fixed spacing**. There is no per-neuron normalization
# or z-scoring, so relative ΔF/F amplitudes are preserved.
#
# With `DFF_VIEW = "all"`, every neuron remains one heatmap row and every frame
# enters a temporal display-bin mean. Rows use the same whole-session activity
# ranking as the spike raster. Antialiased rendering compresses thousands of
# rows onto the page, so use the sampled mode when individual traces matter.


# %%
def make_dff_figure(views, page_label=None):
    """Build the ΔF/F figure for one or more compact recording views."""
    fig = plt.figure(figsize=(15, 9 * len(views)), constrained_layout=True)
    grid = fig.add_gridspec(
        2 * len(views),
        1,
        height_ratios=[value for _view in views for value in (9, 0.42)],
    )
    for row, view in enumerate(views):
        trace_ax = fig.add_subplot(grid[2 * row])
        state_ax = fig.add_subplot(grid[2 * row + 1], sharex=trace_ax)
        if view["dff_view"] == "sample":
            viz.plot_stacked_dff(trace_ax, view)
        else:
            image = viz.plot_all_dff_heatmap(trace_ax, view)
            colorbar = fig.colorbar(image, ax=trace_ax, pad=0.01, fraction=0.025)
            colorbar.set_label("ΔF/F (display-bin mean; median-centered)")
        viz.plot_state_strip(state_ax, view)

    if DFF_VIEW == "sample":
        title = "Raw single-neuron calcium activity across brain states"
    else:
        title = "All-neuron calcium activity across brain states"
    if page_label is not None:
        title = f"{title} · {page_label}"
    fig.suptitle(title, fontsize=15)
    return fig


# %% [markdown]
# ## Figure 2 — deconvolved-spike raster for every neuron
#
# Every row is one neuron; neurons are sorted once by their total number of
# positive deconvolution samples over the whole session (most active at the
# top). This ordering uses no state labels. The raster is binary: it shows
# whether at least one positive sample occurred inside each display bin, not the
# deconvolved spike amplitude.
#
# Every occupied neuron/bin is drawn as a point, so all neurons contribute to the
# rendered overview instead of being nearest-neighbor sampled as an image. Rows
# still cannot be individually resolved on a normal page when N is in the
# thousands. The line above each raster is the percentage of neurons with a
# positive estimate in each original frame, smoothed over five seconds.


# %%
def make_raster_figure(views, page_label=None):
    """Build the all-neuron raster figure for compact recording views."""
    fig = plt.figure(figsize=(15, 6 * len(views)), constrained_layout=True)
    grid = fig.add_gridspec(
        3 * len(views),
        1,
        height_ratios=[value for _view in views for value in (1.0, 3.8, 0.34)],
    )
    for row, view in enumerate(views):
        rate_ax = fig.add_subplot(grid[3 * row])
        raster_ax = fig.add_subplot(grid[3 * row + 1], sharex=rate_ax)
        state_ax = fig.add_subplot(grid[3 * row + 2], sharex=rate_ax)
        viz.plot_population_fraction(rate_ax, view)
        viz.plot_spike_raster(raster_ax, view)
        viz.plot_state_strip(state_ax, view)

    title = "All-neuron OASIS deconvolved-spike rasters and population activity"
    if page_label is not None:
        title = f"{title} · {page_label}"
    fig.suptitle(title, fontsize=15)
    return fig


# %% [markdown]
# ## Figure 3 — deconvolved-spike raster grouped by brain region
#
# This view contains exactly the same neurons, positive deconvolution samples,
# temporal display bins, and population summary as Figure 2. Only the row order
# changes. Neurons are grouped by the recording's row-aligned atlas acronyms,
# following the shared anatomical color order. Within every region they
# retain the whole-session activity rank used in Figure 2.
#
# The narrow strip at the left contains one categorical region color for every
# neuron row. Thin horizontal lines mark region boundaries. The layer `2/3`
# suffix is removed for display, but supplied area acronyms remain distinct.
# Source ``root`` is shown as ``Unassigned`` because it names no specific area;
# `Other` handles an unexpected valid acronym and `Unknown` a missing label. No
# neuron is filtered or averaged. Related families share hues: motor greens,
# somatosensory warm colors, retrosplenial blues, and visual purples.


# %%
def make_region_raster_figure(views, page_label=None):
    """Build all-neuron spike rasters grouped by exact atlas region."""
    fig = plt.figure(figsize=(15, 6 * len(views) + 0.8), constrained_layout=True)
    grid = fig.add_gridspec(
        3 * len(views) + 1,
        2,
        width_ratios=(0.22, 20),
        height_ratios=[value for _view in views for value in (1.0, 3.8, 0.34)] + [0.65],
    )
    shown_brain_regions = set()
    for row, view in enumerate(views):
        rate_ax = fig.add_subplot(grid[3 * row, 1])
        area_ax = fig.add_subplot(grid[3 * row + 1, 0])
        spike_ax = fig.add_subplot(grid[3 * row + 1, 1], sharex=rate_ax)
        state_ax = fig.add_subplot(grid[3 * row + 2, 1], sharex=rate_ax)
        viz.plot_population_fraction(rate_ax, view)
        viz.plot_brain_region_spike_raster(spike_ax, view)
        region_order = view["brain_region_order"]
        viz.plot_brain_region_strip(
            area_ax,
            view["brain_regions"],
            region_order,
        )
        spike_ax.set_ylabel("")
        area_ax.set_ylabel("all neurons\n(grouped by atlas region)", labelpad=8)
        shown_brain_regions.update(view["brain_regions"].tolist())
        viz.plot_state_strip(state_ax, view)

    region_handles = viz.brain_region_legend_handles(
        [region for region in viz.BRAIN_REGION_COLORS if region in shown_brain_regions]
    )
    region_legend_ax = fig.add_subplot(grid[-1, :])
    region_legend_ax.axis("off")
    region_legend_ax.legend(
        handles=region_handles,
        loc="center",
        ncol=min(9, len(region_handles)),
        frameon=False,
        title="Atlas region (activity-ranked within each region)",
    )
    title = "Brain-region-grouped deconvolved-spike rasters and population activity"
    if page_label is not None:
        title = f"{title} · {page_label}"
    fig.suptitle(title, fontsize=15)
    return fig


# %% [markdown]
# ## Figure 4 — active-neuron raster sorted by Rastermap
#
# Rastermap is fit to every neuron that passes the configured activity rule;
# there is no random neuron selection. The upper raster keeps one row per
# selected neuron, while the lower heatmap averages adjacent sorted neurons in
# groups of at most `RASTERMAP_SUPERNEURON_SIZE` for a paper-style overview.
# Both panels retain the complete recorded sequence. The narrow strip at left
# supplies cortical-region context only; spatial localization is not used to
# fit or sort the neurons.
#
# This is a visualization, not proof that the fine neuron order is unique.
# Tutorials 03 and 04 check normalization, PCA, cluster sorting, seed/tuning
# sensitivity, time-block transfer, and synthetic controls before making that
# distinction.


# %%
def make_rastermap_figure(views, page_label=None):
    """Build active-only Rastermap rasters and superneuron views."""
    if not RUN_RASTERMAP:
        raise ValueError("Set RUN_RASTERMAP=True before building this figure")

    fig = plt.figure(figsize=(15, 9 * len(views) + 0.8), constrained_layout=True)
    grid = fig.add_gridspec(
        4 * len(views) + 1,
        2,
        width_ratios=(0.22, 20),
        height_ratios=[value for _view in views for value in (1.0, 3.4, 3.4, 0.34)]
        + [0.65],
    )
    region_handles_by_label = {}
    for row, view in enumerate(views):
        rate_ax = fig.add_subplot(grid[4 * row, 1])
        area_ax = fig.add_subplot(grid[4 * row + 1, 0])
        spike_ax = fig.add_subplot(grid[4 * row + 1, 1], sharex=rate_ax)
        embedding_ax = fig.add_subplot(grid[4 * row + 2, 1], sharex=rate_ax)
        state_ax = fig.add_subplot(grid[4 * row + 3, 1], sharex=rate_ax)

        viz.plot_population_fraction(rate_ax, view)
        rate_ax.set_ylabel("% all recorded neurons\nactive (5-s mean)")
        viz.plot_rastermap_spike_raster(spike_ax, view)
        rastermap_order = np.asarray(view["rastermap_isort"], dtype=np.int64)
        _, region_handles = viz.plot_cortical_region_strip(
            area_ax,
            view["brain_regions"],
            rastermap_order,
        )
        for handle in region_handles:
            region_handles_by_label[handle.get_label()] = handle
        image = viz.plot_rastermap_embedding(
            embedding_ax,
            view,
            vmin=RASTERMAP_DISPLAY_VMIN,
            vmax=RASTERMAP_DISPLAY_VMAX,
        )
        colorbar = fig.colorbar(
            image,
            ax=embedding_ax,
            pad=0.01,
            fraction=0.025,
        )
        colorbar.set_label("normalized superneuron activity")
        viz.plot_state_strip(state_ax, view)

    legend_ax = fig.add_subplot(grid[-1, :])
    legend_ax.axis("off")
    stable_handles = [
        region_handles_by_label[region]
        for region in viz.CORTICAL_REGION_COLORS
        if region in region_handles_by_label
    ]
    legend_ax.legend(
        handles=stable_handles,
        loc="center",
        ncol=min(6, len(stable_handles)),
        frameon=False,
        title="Cortical area of active selected neurons (annotation only)",
    )

    title = "Active-neuron Rastermap ordering across complete recordings"
    if page_label is not None:
        title = f"{title} · {page_label}"
    fig.suptitle(title, fontsize=15)
    return fig


# %% [markdown]
# ## Load and save the selected recording set
#
# `RECORDING_MODE = "all"` writes one full recording per PDF page. With
# `RUN_RASTERMAP = True`, eight files are produced: ΔF/F, activity-ranked raster,
# brain-region-grouped raster, and active-neuron Rastermap PDFs for sleep and
# anesthesia. Turning Rastermap off produces the other six. Streaming keeps peak
# memory near the cost of one recording, and separate pages keep the complete
# time axes legible. With `SHOW_FIGURES = True`, every page is also displayed in
# the interactive window as soon as it is created.
#
# `RECORDING_MODE = "representative"` instead loads the two preview sessions.
# The activity-ranked, region-grouped, and Rastermap rasters are each saved as
# separate sleep and anesthesia PNGs so every timeline has a full-size panel.
# Only the ΔF/F overview retains its combined two-session layout. Changing
# `VIEW_WINDOWS_MIN` to a tuple only zooms the named page; every `None` entry
# above retains that session's complete sequence.
#
# The small helpers below handle repeated loading/name choices rather than
# scientific transformations:
#
# - `load_view` loads and compacts one recording, then frees its large matrices.
# - `dff_output_stem` gives sampled traces and the optional all-neuron heatmap
#   distinct filenames.
#
# The actual execution is deliberately *not* wrapped in a `main()` function.
# Run the cells below in order: load the compact views, inspect each figure, and
# optionally stream the complete ten-session set to PDFs.

# %%
ALL_RECORDINGS = SLEEP_RECORDINGS + ANESTHESIA_RECORDINGS
TRACE_SEED_BY_RECORDING = {
    name: RANDOM_SEED + index for index, name in enumerate(ALL_RECORDINGS)
}
CONDITION_NAME_BY_DATA_INFO = {"sleep": "sleep", "ane": "anesthesia"}


def load_view(recording_name):
    """Load one session and return its compact plotting representation."""
    print(f"Loading {recording_name} ...", flush=True)
    recording = dataio.load_recording(recording_name)
    try:
        print(recording)
        seed = TRACE_SEED_BY_RECORDING.get(recording_name, RANDOM_SEED)
        return prepare_view(recording, seed)
    finally:
        del recording
        gc.collect()


def dff_output_stem():
    """Return an output stem that distinguishes the two ΔF/F display modes."""
    if DFF_VIEW == "sample":
        return "02_dff_traces"
    if DFF_VIEW == "all":
        return "02_dff_all_neurons"
    raise ValueError("DFF_VIEW must be 'sample' or 'all'")


# %% [markdown]
# ### Step 1 — load the two-session interactive preview
#
# This cell performs data loading only. The four following cells build the
# figures one at a time, leaving their figure objects in the workspace so axes
# and arrays can be inspected interactively.

# %%
if RECORDING_MODE not in {"all", "representative"}:
    raise ValueError("RECORDING_MODE must be 'all' or 'representative'")

representative_views = []
if RECORDING_MODE == "representative":
    representative_views = [
        load_view(recording_name) for recording_name in REPRESENTATIVE_RECORDINGS
    ]
else:
    print("RECORDING_MODE='all': skip to Step 6 for streamed all-session figures")


# %% [markdown]
# ### Step 2 — inspect raw ΔF/F

# %%
dff_fig = None
if representative_views:
    dff_fig = make_dff_figure(representative_views)
    dff_path = FIG_DIR / f"{dff_output_stem()}.png"
    dff_fig.savefig(dff_path, dpi=150, bbox_inches="tight")
    print("saved ->", dff_path)
    if SHOW_FIGURES:
        plt.show()


# %% [markdown]
# ### Step 3 — inspect the activity-ranked binary spike raster
#
# This is the simple baseline ordering: neurons with more positive deconvolved
# samples over the complete session appear near the top. Sleep and anesthesia
# are intentionally separate figure objects and output files here: their time
# axes and neuron counts differ, and neither panel is compressed to make room
# for the other.

# %%
raster_figs = {}
raster_paths = {}
if representative_views:
    for view in representative_views:
        condition = CONDITION_NAME_BY_DATA_INFO[view["data_info"]]
        raster_figs[condition] = make_raster_figure((view,))
        raster_paths[condition] = (
            FIG_DIR / f"02_spike_rasters_by_activity_{condition}.png"
        )
        raster_figs[condition].savefig(
            raster_paths[condition],
            dpi=150,
            bbox_inches="tight",
        )
        print("saved ->", raster_paths[condition])
    if SHOW_FIGURES:
        plt.show()

# Named aliases are convenient when running the tutorial cell by cell.
sleep_raster_fig = raster_figs.get("sleep")
anesthesia_raster_fig = raster_figs.get("anesthesia")


# %% [markdown]
# ### Step 4 — inspect the brain-region-grouped binary spike raster
#
# The binary events are identical to Step 3; only the neuron order changes.
# Every atlas-region block remains activity-ranked internally. As in Step 3,
# the sleep and anesthesia sessions are saved as independent figures.

# %%
region_raster_figs = {}
region_raster_paths = {}
if representative_views:
    for view in representative_views:
        condition = CONDITION_NAME_BY_DATA_INFO[view["data_info"]]
        region_raster_figs[condition] = make_region_raster_figure((view,))
        region_raster_paths[condition] = (
            FIG_DIR / f"02_spike_rasters_by_region_{condition}.png"
        )
        region_raster_figs[condition].savefig(
            region_raster_paths[condition],
            dpi=150,
            bbox_inches="tight",
        )
        print("saved ->", region_raster_paths[condition])
    if SHOW_FIGURES:
        plt.show()

sleep_region_raster_fig = region_raster_figs.get("sleep")
anesthesia_region_raster_fig = region_raster_figs.get("anesthesia")


# %% [markdown]
# ### Step 5 — inspect the active-neuron Rastermap order
#
# Unlike Steps 3--4, this view excludes neurons below the explicit activity
# threshold. The order uses the official model and every recorded frame; the
# cortical strip is annotation, not an input to Rastermap. Sleep and anesthesia
# are again kept in separate figures.

# %%
rastermap_figs = {}
rastermap_paths = {}
if representative_views and RUN_RASTERMAP:
    for view in representative_views:
        condition = CONDITION_NAME_BY_DATA_INFO[view["data_info"]]
        rastermap_figs[condition] = make_rastermap_figure((view,))
        rastermap_paths[condition] = FIG_DIR / f"02_rastermap_{condition}.png"
        rastermap_figs[condition].savefig(
            rastermap_paths[condition],
            dpi=150,
            bbox_inches="tight",
        )
        print("saved ->", rastermap_paths[condition])
    if SHOW_FIGURES:
        plt.show()

sleep_rastermap_fig = rastermap_figs.get("sleep")
anesthesia_rastermap_fig = rastermap_figs.get("anesthesia")


# %% [markdown]
# ### Step 6 — optionally stream every sleep and anesthesia session
#
# Set `RECORDING_MODE = "all"` in the settings cell, then run this cell. Only one
# large recording is resident at a time. Each page is saved and shown before
# the corresponding compact view is released.

# %%
if RECORDING_MODE == "all":
    recording_groups = (
        ("sleep", SLEEP_RECORDINGS),
        ("anesthesia", ANESTHESIA_RECORDINGS),
    )
    for condition, recording_names in recording_groups:
        dff_path = FIG_DIR / f"{dff_output_stem()}_{condition}_all_sessions.pdf"
        raster_path = (
            FIG_DIR / f"02_spike_rasters_by_activity_{condition}_all_sessions.pdf"
        )
        region_raster_path = (
            FIG_DIR / f"02_spike_rasters_by_region_{condition}_all_sessions.pdf"
        )
        rastermap_path = FIG_DIR / f"02_rastermap_{condition}_all_sessions.pdf"
        metadata = {
            "Title": f"Population activity: all {condition} recording sessions",
            "Subject": "One complete calcium-imaging recording per page",
        }

        with ExitStack() as outputs:
            dff_pdf = outputs.enter_context(PdfPages(dff_path, metadata=metadata))
            raster_pdf = outputs.enter_context(PdfPages(raster_path, metadata=metadata))
            region_raster_pdf = outputs.enter_context(
                PdfPages(region_raster_path, metadata=metadata)
            )
            rastermap_pdf = (
                outputs.enter_context(PdfPages(rastermap_path, metadata=metadata))
                if RUN_RASTERMAP
                else None
            )
            for page, recording_name in enumerate(recording_names, start=1):
                print(
                    f"{condition} session {page}/{len(recording_names)}: "
                    f"{recording_name}",
                    flush=True,
                )
                view = load_view(recording_name)
                page_label = (
                    f"{condition.title()} session {page}/{len(recording_names)}"
                )

                session_dff_fig = make_dff_figure((view,), page_label=page_label)
                dff_pdf.savefig(session_dff_fig, dpi=150, bbox_inches="tight")
                if SHOW_FIGURES:
                    plt.show()
                plt.close(session_dff_fig)

                session_raster_fig = make_raster_figure(
                    (view,),
                    page_label=page_label,
                )
                raster_pdf.savefig(
                    session_raster_fig,
                    dpi=150,
                    bbox_inches="tight",
                )
                if SHOW_FIGURES:
                    plt.show()
                plt.close(session_raster_fig)

                session_region_raster_fig = make_region_raster_figure(
                    (view,),
                    page_label=page_label,
                )
                region_raster_pdf.savefig(
                    session_region_raster_fig,
                    dpi=150,
                    bbox_inches="tight",
                )
                if SHOW_FIGURES:
                    plt.show()
                plt.close(session_region_raster_fig)

                if rastermap_pdf is not None:
                    session_rastermap_fig = make_rastermap_figure(
                        (view,),
                        page_label=page_label,
                    )
                    rastermap_pdf.savefig(
                        session_rastermap_fig,
                        dpi=150,
                        bbox_inches="tight",
                    )
                    if SHOW_FIGURES:
                        plt.show()
                    plt.close(session_rastermap_fig)

                del view
                gc.collect()

        print("saved ->", dff_path)
        print("saved ->", raster_path)
        print("saved ->", region_raster_path)
        if RUN_RASTERMAP:
            print("saved ->", rastermap_path)


# %% [markdown]
# ## What to look for
#
# - In the ΔF/F panel, ask whether events become more synchronous, more sparse,
#   or change amplitude around state transitions. Sampled traces show individual
#   examples; the all-neuron heatmap is a compressed population overview.
# - In the raster, vertical concentrations of black marks indicate many neurons
#   becoming active at similar times. Differences from top to bottom reflect
#   neurons' overall positive-sample counts because the rows are activity-ranked.
# - In the brain-region figure, compare each atlas block with Figure 2. The
#   events are identical, and neurons remain activity-ranked inside each block;
#   only the between-region grouping changes. Look for region-specific periods
#   of dense or sparse activity around state transitions.
# - In the Rastermap figure, look for smooth sequences or repeated bands that
#   become clearer after activity selection and ordering. The lower panel is an
#   adjacent-neuron average, so confirm any apparent pattern in the one-row-per-
#   selected-neuron raster above it. A striking image alone does not establish a
#   unique neuron order; tutorial 04 tests that order out of sample.
# - Use the population-active-fraction line to compare overall activity with
#   the exact state strip. Treat it as descriptive, not a statistical state test.
# - The next tutorials use long, stable `used_frame` epochs for quantitative
#   state comparisons. Here we retain the original time axis and all state
#   labels; with the default `None` windows, every transition remains visible.
