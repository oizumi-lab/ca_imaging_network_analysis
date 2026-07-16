# %% [markdown]
# # 02 · Visualizing population activity across brain states
#
# Before estimating a functional network, first look at the activity itself.
# This tutorial makes two complementary views:
#
# 1. **Raw ΔF/F activity**, either as stacked traces from a reproducible random
#    sample of 100 neurons or as a display-aggregated heatmap to which every
#    neuron and frame contributes.
# 2. **An all-neuron deconvolved-spike raster** made from `spike_deconv > 0`.
#    Every neuron is retained; only the time axis is binned to about one second
#    for display. A population-active-fraction trace makes changes in overall
#    activity easier to see when thousands of rows are compressed onto the page.
#
# Sleep and anesthesia come from separate recording sessions, so they are shown
# as separate timelines. The colored strip under every panel is aligned to the
# original frame-by-frame state vector. In the sleep recording it includes
# awake, quiet awake, NREM sleep, and REM sleep rather than joining disjoint
# state-specific analysis epochs into an artificial continuous time series.
# Dashed vertical lines mark microscope acquisition breaks; time across such a
# line is cumulative recorded time, not an uninterrupted acquisition.
# By default, both figures show the **entire recorded sequence**. The settings
# below also let you zoom to a chosen time interval without changing the data.

# %%
import gc
import os
import sys

# add the repo root (parent of scripts/) to the path so `src.funcnet` is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from src.funcnet import dataio, timeseries as ts, visualization as viz
from src.funcnet.paths import FIG_DIR

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
# - `SHOW_FIGURES`: when `True`, display every ΔF/F and raster figure in the
#   interactive window. Figures are shown and released one session at a time so
#   displaying all sessions does not keep all large plot objects in memory.
#
# The raster and the `"all"` ΔF/F view never apply `nonzero_ROI` or another
# neuron filter. Full-session mode requires all files from `download_data.py`,
# not only `download_data.py --example`. It streams one recording at a time so
# that the ten large data files are never held in memory simultaneously.

# %%
RECORDING_MODE = "all"  # "all" = all 10 sessions; "representative" = quick preview

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
    "mouse01_sleep": None,       # None = this session's entire sequence
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
DFF_VIEW = "sample"            # "sample" = line traces; "all" = all-neuron heatmap
N_TRACE_NEURONS = 100
RANDOM_SEED = 7
DISPLAY_BIN_SECONDS = 1.0
POPULATION_SMOOTH_SECONDS = 5.0
SHOW_FIGURES = True


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
# - `save_all_session_figures` shows every compact view interactively and also
#   streams it into the multipage PDFs, one full recording per page.


# %% [markdown]
# ## Build lightweight plotting data
#
# The loader intentionally returns all three full `N × T` activity matrices, so
# one recording can occupy several gigabytes. We load sessions one at a time
# and retain only sampled traces or a binned ΔF/F heatmap, plus a compact spike
# raster and population active fraction. In all-session mode, even this compact
# view is released as soon as its two PDF pages have been written.
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
# - `prepare_view` runs the relevant helpers while the large recording is loaded
#   and returns a smaller dictionary used by the plotting functions.

# %%
def prepare_view(rec, seed) -> viz.ActivityView:
    """Extract only the arrays needed by the two tutorial figures."""
    if DFF_VIEW not in {"sample", "all"}:
        raise ValueError("DFF_VIEW must be 'sample' or 'all'")
    duration_min = rec.n_frames / rec.fs / 60
    time_limits_min = viz.resolve_time_limits(
        duration_min,
        VIEW_WINDOWS_MIN.get(rec.name),
    )

    print(f"  building a deconvolved-spike raster for all {rec.n_neurons:,} neurons ...", flush=True)
    raster, active_counts, bin_frames, neuron_order, bin_centers = viz.binned_spike_raster(
        rec.spike_deconv,
        rec.fs,
        DISPLAY_BIN_SECONDS,
        rec.boundary_ind,
        rec.state,
    )

    if DFF_VIEW == "sample":
        trace_ids = viz.select_trace_neurons(rec.n_neurons, N_TRACE_NEURONS, seed)
        print(f"  copying raw ΔF/F for {trace_ids.size} selected neurons ...", flush=True)
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

    return {
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
        "boundary_minutes": boundary_minutes,
        "acquisition_segments": ts.acquisition_segments(
            rec.n_frames,
            rec.boundary_ind,
        ),
    }

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
# ## Load and save the selected recording set
#
# `RECORDING_MODE = "all"` writes one full recording per PDF page. Four files
# are produced: ΔF/F and spike-raster PDFs for sleep and anesthesia. Streaming
# keeps peak memory near the cost of one recording, and separate pages keep the
# complete time axes legible. With `SHOW_FIGURES = True`, every page is also
# displayed in the interactive window as soon as it is created.
#
# `RECORDING_MODE = "representative"` instead loads the two preview sessions,
# places them in the original combined PNG figures, and displays them. Changing
# `VIEW_WINDOWS_MIN` to a tuple only zooms the named page; every `None` entry
# above retains that session's complete sequence.
#
# The helpers below handle output rather than scientific transformations:
#
# - `load_view` loads and compacts one recording, then frees its large matrices.
# - `dff_output_stem` gives sampled traces and the optional all-neuron heatmap
#   distinct filenames.
# - `save_representative_figures` writes the fast two-session PNG preview.
# - `save_all_session_figures` displays every session and writes the four
#   all-session multipage PDFs.

# %%
ALL_RECORDINGS = SLEEP_RECORDINGS + ANESTHESIA_RECORDINGS
TRACE_SEED_BY_RECORDING = {
    name: RANDOM_SEED + index
    for index, name in enumerate(ALL_RECORDINGS)
}


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


def save_representative_figures():
    """Save the two-session PNG preview and optionally display it."""
    views = []
    dff_fig = None
    raster_fig = None
    try:
        views.extend(load_view(name) for name in REPRESENTATIVE_RECORDINGS)

        dff_fig = make_dff_figure(views)
        dff_path = FIG_DIR / f"{dff_output_stem()}.png"
        dff_fig.savefig(dff_path, dpi=150, bbox_inches="tight")
        print("saved ->", dff_path)

        raster_fig = make_raster_figure(views)
        raster_path = FIG_DIR / "02_spike_rasters.png"
        raster_fig.savefig(raster_path, dpi=150, bbox_inches="tight")
        print("saved ->", raster_path)
        if SHOW_FIGURES:
            plt.show()
    finally:
        if dff_fig is not None:
            plt.close(dff_fig)
        if raster_fig is not None:
            plt.close(raster_fig)
        views.clear()
        gc.collect()


def save_all_session_figures():
    """Display and stream every recording into condition-specific PDFs."""
    groups = (
        ("sleep", SLEEP_RECORDINGS),
        ("anesthesia", ANESTHESIA_RECORDINGS),
    )
    for condition, recording_names in groups:
        dff_path = FIG_DIR / f"{dff_output_stem()}_{condition}_all_sessions.pdf"
        raster_path = FIG_DIR / f"02_spike_rasters_{condition}_all_sessions.pdf"
        metadata = {
            "Title": f"Population activity: all {condition} recording sessions",
            "Subject": "One complete calcium-imaging recording per page",
        }

        with (
            PdfPages(dff_path, metadata=metadata) as dff_pdf,
            PdfPages(raster_path, metadata=metadata) as raster_pdf,
        ):
            for page, recording_name in enumerate(recording_names, start=1):
                print(
                    f"{condition} session {page}/{len(recording_names)}: "
                    f"{recording_name}",
                    flush=True,
                )
                view = load_view(recording_name)
                page_label = f"{condition.title()} session {page}/{len(recording_names)}"

                dff_fig = make_dff_figure((view,), page_label=page_label)
                try:
                    dff_pdf.savefig(dff_fig, dpi=150, bbox_inches="tight")
                    if SHOW_FIGURES:
                        plt.show()
                finally:
                    plt.close(dff_fig)
                del dff_fig

                raster_fig = make_raster_figure((view,), page_label=page_label)
                try:
                    raster_pdf.savefig(raster_fig, dpi=150, bbox_inches="tight")
                    if SHOW_FIGURES:
                        plt.show()
                finally:
                    plt.close(raster_fig)
                del raster_fig

                del view
                gc.collect()

        print("saved ->", dff_path)
        print("saved ->", raster_path)


if RECORDING_MODE == "representative":
    save_representative_figures()
elif RECORDING_MODE == "all":
    save_all_session_figures()
else:
    raise ValueError("RECORDING_MODE must be 'all' or 'representative'")


# %% [markdown]
# ## What to look for
#
# - In the ΔF/F panel, ask whether events become more synchronous, more sparse,
#   or change amplitude around state transitions. Sampled traces show individual
#   examples; the all-neuron heatmap is a compressed population overview.
# - In the raster, vertical concentrations of black marks indicate many neurons
#   becoming active at similar times. Differences from top to bottom reflect
#   neurons' overall positive-sample counts because the rows are activity-ranked.
# - Use the population-active-fraction line to compare overall activity with
#   the exact state strip. Treat it as descriptive, not a statistical state test.
# - The next tutorials use long, stable `used_frame` epochs for quantitative
#   state comparisons. Here we retain the original time axis and all state
#   labels; with the default `None` windows, every transition remains visible.
